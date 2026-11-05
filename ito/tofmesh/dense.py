"""The latest dense map as triangles, plus persistent geometry outside its observed rays.

A depth discontinuity cuts triangle *connections*, never a neighbourhood of valid pixels.
The full-resolution map stays live while normal-distance TSDF keyframes retain the room.
"""

import numpy as np


def mesh(mapped, camera, image):
    """On the GPU, from the map's tensors (depth.DepthMap.gpu, in the depth process): a few ms,
    where NumPy took ~75 there."""
    import torch

    depth, xyz, normals, inside = mapped.gpu
    h, w = depth.shape
    camera = torch.as_tensor(camera.astype(np.float32)).to(depth.device)
    world = xyz.reshape(-1, 3) @ camera[:3, :3].T + camera[:3, 3]
    normals = normals.reshape(-1, 3) @ camera[:3, :3].T
    a = (
        torch.arange(h - 1, device=depth.device)[:, None] * w
        + torch.arange(w - 1, device=depth.device)
    ).reshape(-1)
    triangles = torch.cat(
        (torch.stack((a, a + w, a + 1), -1), torch.stack((a + 1, a + w, a + w + 1), -1))
    )
    d = depth.reshape(-1)[triangles]
    low, high = d.min(1).values, d.max(1).values
    valid = torch.isfinite(d).all(1) & (low > 0.05) & (high < 12.0)
    # Camera pixels can connect along a sloping plane; an abrupt 20% jump is an occlusion.
    valid &= high <= 1.2 * low
    weights = torch.where(inside, 1.0, mapped.outside).reshape(-1)
    valid &= (weights[triangles] > 0).all(1)
    # Only cells with both triangles intact occlude the retained surface below them.
    safe = valid[: len(a)] & valid[len(a) :]
    covered = torch.zeros((h, w), dtype=torch.bool, device=depth.device)
    covered[:-1, :-1] = safe.reshape(h - 1, w - 1)
    mapped.covered = covered.cpu().numpy()
    world, normals = world.cpu().numpy(), normals.cpu().numpy()
    triangles = triangles[valid].to(torch.int32).cpu().numpy().view(np.uint32)
    colours = np.column_stack((image.reshape(-1, 3), np.full(h * w, 255, np.uint8)))
    blocks = np.zeros((len(world), 4), np.float32)
    blocks[:, 3] = -1e6  # no atlas: newest camera projector, captured RGB if occluded later
    return world, normals, colours, triangles, blocks


def uncovered(part, mapped, camera):
    """Keep retained triangles only where the live depth map has no surface.

    This is duplicate-surface removal, not hiding holes: if a live triangle was cut at an
    occlusion, the TSDF there remains visible. Test all three vertices rather than a centre
    so a persistent triangle crossing the camera border is not silently discarded.
    """
    pos, normal, colour, tri, blocks = part
    inv = np.linalg.inv(camera)
    local = pos @ inv[:3, :3].T + inv[:3, 3]
    v, u, ok, _ = mapped._pixels(local)
    covered = ok & mapped.covered[v, u]
    tri = tri[~covered[tri].all(1)]
    return pos, normal, colour, tri, blocks
