"""Triangles from the volume's zero crossing: surface nets, one block at a time.

Surface nets put one vertex in every voxel cell whose corners change sign (at the mean of the
edge crossings) and join the four cells around every sign-changing voxel edge into a quad. No
case tables, smooth output, about half the triangles of marching cubes: plenty for a surface
known to a few centimetres.

Each block owns the edges that start on its own voxels, so neighbouring blocks never emit the
same quad; the cells around those edges reach one voxel into the neighbours, whose data is
gathered alongside. A block's mesh therefore changes when a neighbour changes: callers re-mesh
the neighbours of every changed block.

Speed: a surface is a sheet through a volume, so only a few percent of a block's cells hold a
vertex. Signs and weights are read for the whole region (cheap), but vertex positions, normals
and colours are computed for those active cells only: re-meshing is the worker's biggest cost
and what makes the mesh lag behind the ToF.
"""

import numpy as np

from .volume import BLOCK

R = BLOCK + 2  # region: voxels -1 .. BLOCK of a block
N = R - 1  # cells per axis; cell c has corners at region voxels c and c + 1
_OFFSET = np.arange(R) - 1
_BLOCK_OF = _OFFSET // BLOCK + 1
_LOCAL_OF = _OFFSET % BLOCK
CORNERS = np.array([(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)])
EDGES = [
    (a, b) for a in range(8) for b in range(a + 1, 8) if np.abs(CORNERS[a] - CORNERS[b]).sum() == 1
]
_BI, _BJ, _BK = np.meshgrid(_BLOCK_OF, _BLOCK_OF, _BLOCK_OF, indexing="ij")
# Index of each region voxel inside its block, flattened (i * 64 + j * 8 + k).
_LOCAL_FLAT = np.add.reduce(
    np.meshgrid(_LOCAL_OF * BLOCK * BLOCK, _LOCAL_OF * BLOCK, _LOCAL_OF, indexing="ij")
)
_SIGN = (CORNERS * 2 - 1).astype(np.float32)  # (8, 3)
# Offset of each cell corner from the cell's corner 0, in flattened (R, R, R) region indices.
_CORNER_OFFSET = (CORNERS[:, 0] * R + CORNERS[:, 1]) * R + CORNERS[:, 2]


def _corners(a):
    """How many of every cell's eight corners are set in `a` (M, R, R, R) bool: (M, N, N, N)."""
    c = a.view(np.uint8)
    c = c[:, :-1] + c[:, 1:]
    c = c[:, :, :-1] + c[:, :, 1:]
    return c[:, :, :, :-1] + c[:, :, :, 1:]


def extract(volume, slots, min_weight=0.1):
    """Meshes of these blocks: a list of (key, positions (V,3) f32, normals (V,3) f32,
    colours (V,4) u8 with alpha 0 where no camera frame coloured it, triangles (T,3) u32)."""
    slots = np.asarray(slots, np.int64)
    if not len(slots):
        return []
    m = len(slots)
    # Flat voxel index of every region voxel; a missing neighbour is the volume's empty row.
    region = volume.neighbourhood(slots)[:, _BI, _BJ, _BK]  # (M, R, R, R) slot
    flat = region * BLOCK**3 + _LOCAL_FLAT
    sdf = volume.sdf.reshape(-1)[flat]
    valid = volume.weight.reshape(-1)[flat] >= min_weight
    negative = sdf < 0

    # Active cells: all eight corners weighted, signs mixed (counted over each 2x2x2 of corners,
    # one axis at a time).
    valid_corners = _corners(valid)
    negative_corners = _corners(negative)
    active = (valid_corners == 8) & (negative_corners > 0) & (negative_corners < 8)
    blk, ci, cj, ck = np.nonzero(active)  # block-major, so each block's vertices are contiguous
    count = len(blk)
    vid = np.full(active.shape, -1, np.int64)
    vid[blk, ci, cj, ck] = np.arange(count)

    # The eight corners of every active cell (one gather: corner 0's place in the region plus
    # each corner's offset).
    corner0 = ((blk * R + ci) * R + cj) * R + ck
    corner_flat = flat.reshape(-1)[corner0[:, None] + _CORNER_OFFSET[None, :]]  # (A, 8)
    s = sdf.reshape(-1)[corner0[:, None] + _CORNER_OFFSET[None, :]]

    # Vertex positions: mean of the edge crossings, in cell units. An edge runs from corner a
    # (0 on its axis) to b (1 on it): its crossing is corner a's other two coordinates and t along
    # the axis. Summed per coordinate in float32, edge by edge (the same float32 sums as adding
    # whole points: float64 has room for every float32 sum to round only once).
    total = np.zeros((3, count), np.float32)
    crossings = np.zeros(count, np.float32)
    for a, b in EDGES:
        sa, sb = s[:, a], s[:, b]
        crossing = (sa < 0) != (sb < 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(crossing, sa / (sa - sb), np.float32(0))
        axis = int(np.argmax(CORNERS[b] - CORNERS[a]))
        for c in range(3):
            if c == axis:
                total[c] += t
            elif CORNERS[a][c]:
                total[c] += crossing
        crossings += crossing
    inside = total.T / np.maximum(crossings, 1)[:, None]

    # Normal: the signed distance's gradient over the cell, pointing to free space.
    gradient = s @ _SIGN
    gradient /= np.maximum(np.linalg.norm(gradient, axis=-1, keepdims=True), 1e-9)

    cw = volume.colour_weight.reshape(-1)[corner_flat]  # (A, 8)
    cws = cw.sum(-1)
    # Colour-weighted mean of the corners' colours, summed corner by corner (as a sum over that
    # axis of an (A, 8, 3) product does, without making it).
    rgb = volume.colour.reshape(-1, 3)
    mean = rgb[corner_flat[:, 0]] * cw[:, :1]
    for c in range(1, 8):
        mean += rgb[corner_flat[:, c]] * cw[:, c : c + 1]
    mean /= np.maximum(cws, 1e-6)[:, None]

    # Cell c's corner 0 is voxel c-1, whose centre is at (c - 1 + 0.5) voxels.
    cell = np.stack([ci, cj, ck], 1)
    origin = volume.keys[slots] * volume.block_size
    positions = origin[blk] + (cell - 0.5 + inside) * volume.voxel

    triangles = []
    tri_block = []
    own = slice(1, BLOCK + 1)  # the block's own voxels, in region indices
    for axis in range(3):
        b, c = [x for x in range(3) if x != axis]
        nxt = [own, own, own]
        nxt[axis] = slice(2, BLOCK + 2)
        here = (slice(None), own, own, own)
        there = (slice(None), *nxt)
        crossing = valid[here] & valid[there] & (negative[here] != negative[there])
        tb, i, j, k = np.nonzero(crossing)
        if not len(tb):
            continue
        base = np.stack([i, j, k], 1) + 1  # (E, 3) region voxel index == cell index on `axis`
        quad = []
        for db, dc in ((-1, -1), (0, -1), (0, 0), (-1, 0)):
            cidx = base.copy()
            cidx[:, b] += db
            cidx[:, c] += dc
            quad.append(vid[tb, cidx[:, 0], cidx[:, 1], cidx[:, 2]])
        quad = np.stack(quad, 1)
        flip = negative[here][tb, i, j, k]
        quad = np.where(flip[:, None], quad[:, ::-1], quad)
        good = (quad >= 0).all(1)
        quad, tb = quad[good], tb[good]
        triangles.append(np.concatenate([quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]]))
        tri_block.append(np.concatenate([tb, tb]))
    if triangles:
        triangles = np.concatenate(triangles)
        tri_block = np.concatenate(tri_block)
    else:
        triangles = np.zeros((0, 3), np.int64)
        tri_block = np.zeros(0, np.int64)

    pos_flat = positions.astype(np.float32)
    nrm_flat = gradient.astype(np.float32)
    known = cws > 0
    mean = _fill(mean, known, triangles)
    alpha = known.astype(np.float32) * 255
    rgba = np.concatenate([mean, alpha[:, None]], 1).clip(0, 255).astype(np.uint8)
    starts = np.searchsorted(blk, np.arange(m + 1))
    order = np.argsort(tri_block, kind="stable")
    triangles, tri_block = triangles[order], tri_block[order]
    tri_starts = np.searchsorted(tri_block, np.arange(m + 1))
    out = []
    for i, key in enumerate(map(tuple, volume.keys[slots].tolist())):
        a, b = starts[i], starts[i + 1]
        tris = triangles[tri_starts[i] : tri_starts[i + 1]] - a
        if not len(tris):
            out.append((key, None, None, None, None))
            continue
        out.append((key, pos_flat[a:b], nrm_flat[a:b], rgba[a:b], tris.astype(np.uint32)))
    return out


FILL_ROUNDS = 12


def _fill(colour, known, triangles):
    """Vertices no camera frame coloured (their voxels never painted: seen only at a grazing
    angle, or hidden behind a bump) take their painted neighbours' colour along the mesh, ring by
    ring, so the renderer has a colour for them instead of grey. Updates `known` in place."""
    if known.all() or not known.any() or not len(triangles):
        return colour
    colour = np.where(known[:, None], colour, 0.0)
    a = triangles.reshape(-1)
    b = np.roll(triangles, 1, axis=1).reshape(-1)
    a, b = np.concatenate([a, b]), np.concatenate([b, a])  # every edge, both ways
    keep = ~known[a]  # only edges into uncoloured vertices matter
    a, b = a[keep], b[keep]
    for _ in range(FILL_ROUNDS):
        source = known[b]
        if not source.any():
            break
        into, src = a[source], b[source]
        # Per vertex reached: how many coloured neighbours, and their colours' sum (bincount adds
        # in input order, so the same sums as binning by unique vertex).
        counts = np.bincount(into, minlength=len(colour))
        vertices = np.flatnonzero(counts)
        count = counts[vertices]
        total = np.zeros((len(vertices), 3))
        for c in range(3):
            total[:, c] = np.bincount(into, weights=colour[src, c], minlength=len(colour))[vertices]
        colour[vertices] = total / count[:, None]
        known[vertices] = True
        left = ~known[a]
        a, b = a[left], b[left]
        if not len(a):
            break
    return colour
