"""Triangles from the volume's zero crossing: surface nets, one block at a time.

Surface nets put one vertex in every voxel cell whose corners change sign (at the mean of the
edge crossings) and join the four cells around every sign-changing voxel edge into a quad. No
case tables, smooth output, about half the triangles of marching cubes: plenty for a surface
known to a few centimetres.

Each block owns the edges that start on its own voxels, so neighbouring blocks never emit the
same quad; the cells around those edges reach one voxel into the neighbours, whose data is
gathered alongside. A block's mesh therefore changes when a neighbour changes: callers re-mesh
the neighbours of every changed block.
"""

import numpy as np

from .volume import BLOCK

R = BLOCK + 2  # region: voxels -1 .. BLOCK of a block
_OFFSET = np.arange(R) - 1
_BLOCK_OF = _OFFSET // BLOCK + 1
_LOCAL_OF = _OFFSET % BLOCK
CORNERS = np.array([(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)])
EDGES = [
    (a, b) for a in range(8) for b in range(a + 1, 8) if np.abs(CORNERS[a] - CORNERS[b]).sum() == 1
]


def _gather(volume, slots):
    nb = volume.neighbourhood(slots)
    bi, bj, bk = np.meshgrid(_BLOCK_OF, _BLOCK_OF, _BLOCK_OF, indexing="ij")
    li, lj, lk = np.meshgrid(_LOCAL_OF, _LOCAL_OF, _LOCAL_OF, indexing="ij")
    region = nb[:, bi, bj, bk]  # (M, R, R, R) slot of each region voxel
    sdf = volume.sdf[region, li, lj, lk]
    weight = volume.weight[region, li, lj, lk]
    colour = volume.colour[region, li, lj, lk]
    colour_weight = volume.colour_weight[region, li, lj, lk]
    return sdf, weight, colour, colour_weight


def extract(volume, slots, min_weight=0.25):
    """Meshes of these blocks: a list of (key, positions (V,3) f32, normals (V,3) f32,
    colours (V,4) u8 with alpha 0 where no camera frame coloured it, triangles (T,3) u32)."""
    slots = np.asarray(slots, np.int64)
    if not len(slots):
        return []
    sdf, weight, colour, colour_weight = _gather(volume, slots)
    m = len(slots)
    n = R - 1  # cells per axis
    corner = (
        lambda a, c=None: (  # noqa: E731
            (sdf if c is None else c)[:, a[0] : a[0] + n, a[1] : a[1] + n, a[2] : a[2] + n]
        )
    )
    s = np.stack([corner(c) for c in CORNERS], -1)  # (M, n, n, n, 8)
    valid = np.stack([corner(c, weight) >= min_weight for c in CORNERS], -1).all(-1)
    negative = s < 0
    active = valid & negative.any(-1) & (~negative).any(-1)

    # Vertex positions: mean of the edge crossings, in cell units.
    total = np.zeros(active.shape + (3,), np.float32)
    count = np.zeros(active.shape, np.float32)
    for a, b in EDGES:
        sa, sb = s[..., a], s[..., b]
        crossing = (sa < 0) != (sb < 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(crossing, sa / (sa - sb), 0)
        point = CORNERS[a] + t[..., None] * (CORNERS[b] - CORNERS[a])
        total += np.where(crossing[..., None], point, 0)
        count += crossing
    inside = total / np.maximum(count, 1)[..., None]

    # Normal: the signed distance's gradient over the cell, pointing to free space.
    sign = CORNERS * 2 - 1  # (8, 3)
    gradient = np.einsum("...c,cd->...d", s, sign.astype(np.float32))
    gradient /= np.maximum(np.linalg.norm(gradient, axis=-1, keepdims=True), 1e-9)

    cw = np.stack([corner(c, colour_weight) for c in CORNERS], -1)  # (M,n,n,n,8)
    rgb = np.stack(
        [colour[:, c[0] : c[0] + n, c[1] : c[1] + n, c[2] : c[2] + n] for c in CORNERS], -2
    )  # (M,n,n,n,8,3)
    cws = cw.sum(-1)
    mean = (rgb * cw[..., None]).sum(-2) / np.maximum(cws, 1e-6)[..., None]

    flat = active.reshape(m, -1)
    per_block = flat.sum(1)
    starts = np.concatenate([[0], np.cumsum(per_block)])
    vid = np.cumsum(flat.ravel()) - 1
    vid = np.where(flat.ravel(), vid, -1).reshape(active.shape)

    cell = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1)  # (n,n,n,3)
    origin = volume.keys[slots] * volume.block_size  # block's voxel 0 corner region
    voxel = volume.voxel
    # Cell c's corner 0 is voxel c-1, whose centre is at (c - 1 + 0.5) voxels.
    local = (cell[None] - 0.5 + inside) * voxel  # (M,n,n,n,3)
    positions = origin[:, None, None, None, :] + local

    triangles = []
    tri_block = []
    v = np.arange(1, BLOCK + 1)  # the block's own voxels, in region indices
    for axis in range(3):
        b, c = [x for x in range(3) if x != axis]
        idx = np.stack(np.meshgrid(v, v, v, indexing="ij"), -1)  # (8,8,8,3) region voxel
        nxt = idx.copy()
        nxt[..., axis] += 1
        s0 = sdf[:, idx[..., 0], idx[..., 1], idx[..., 2]]
        s1 = sdf[:, nxt[..., 0], nxt[..., 1], nxt[..., 2]]
        ok0 = weight[:, idx[..., 0], idx[..., 1], idx[..., 2]] >= min_weight
        ok1 = weight[:, nxt[..., 0], nxt[..., 1], nxt[..., 2]] >= min_weight
        crossing = ok0 & ok1 & ((s0 < 0) != (s1 < 0))
        blk, i, j, k = np.nonzero(crossing)
        if not len(blk):
            continue
        base = idx[i, j, k]  # (E, 3) region voxel index == cell index along `axis`
        quad = []
        for db, dc in ((-1, -1), (0, -1), (0, 0), (-1, 0)):
            cidx = base.copy()
            cidx[:, b] += db
            cidx[:, c] += dc
            quad.append(vid[blk, cidx[:, 0], cidx[:, 1], cidx[:, 2]])
        quad = np.stack(quad, 1)
        flip = s0[blk, i, j, k] < 0
        quad = np.where(flip[:, None], quad[:, ::-1], quad)
        good = (quad >= 0).all(1)
        quad, blk = quad[good], blk[good]
        triangles.append(np.concatenate([quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]]))
        tri_block.append(np.concatenate([blk, blk]))
    if triangles:
        triangles = np.concatenate(triangles)
        tri_block = np.concatenate(tri_block)
    else:
        triangles = np.zeros((0, 3), np.int64)
        tri_block = np.zeros(0, np.int64)

    pos_flat = positions.reshape(-1, 3)[flat.ravel()].astype(np.float32)
    nrm_flat = gradient.reshape(-1, 3)[flat.ravel()].astype(np.float32)
    col_flat = mean.reshape(-1, 3)[flat.ravel()]
    alpha = (cws.reshape(-1)[flat.ravel()] > 0).astype(np.float32) * 255
    rgba = np.concatenate([col_flat, alpha[:, None]], 1).clip(0, 255).astype(np.uint8)
    order = np.argsort(tri_block, kind="stable")
    triangles, tri_block = triangles[order], tri_block[order]
    tri_starts = np.searchsorted(tri_block, np.arange(m + 1))
    out = []
    for i, slot in enumerate(slots):
        a, b = starts[i], starts[i + 1]
        tris = triangles[tri_starts[i] : tri_starts[i + 1]] - a
        key = tuple(int(x) for x in volume.keys[slot])
        if not len(tris):
            out.append((key, None, None, None, None))
            continue
        out.append((key, pos_flat[a:b], nrm_flat[a:b], rgba[a:b], tris.astype(np.uint32)))
    return out
