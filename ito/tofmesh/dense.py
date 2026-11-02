"""The latest dense map as triangles, plus persistent geometry outside its observed rays.

A depth discontinuity cuts triangle *connections*, never a neighbourhood of valid pixels.
The full-resolution map stays live while normal-distance TSDF keyframes retain the room.
"""

import numpy as np


def mesh(mapped, camera, image):
    camera = camera.astype(np.float32)
    h, w = mapped.depth.shape
    local = mapped.xyz.reshape(-1, 3)
    world = (local @ camera[:3, :3].T + camera[:3, 3]).astype(np.float32)
    normals = (mapped.normals.reshape(-1, 3) @ camera[:3, :3].T).astype(np.float32)
    a = (np.arange(h - 1)[:, None] * w + np.arange(w - 1)).ravel()
    triangles = np.concatenate(
        (np.stack((a, a + w, a + 1), -1), np.stack((a + 1, a + w, a + w + 1), -1))
    )
    d = mapped.depth.ravel()[triangles]
    valid = np.isfinite(d).all(1) & (d.min(1) > 0.05) & (d.max(1) < 12.0)
    # Camera pixels can connect along a sloping plane; an abrupt 20% jump is an occlusion.
    valid &= d.max(1) <= 1.2 * d.min(1)
    weights = np.where(mapped.inside, 1.0, mapped.outside).ravel()
    valid &= (weights[triangles] > 0).all(1)
    # Only cells with both triangles intact occlude the retained surface below them.
    safe = valid[: len(a)] & valid[len(a) :]
    mapped.covered = np.zeros((h, w), bool)
    mapped.covered[:-1, :-1] = safe.reshape(h - 1, w - 1)
    colours = np.column_stack((image.reshape(-1, 3), np.full(h * w, 255, np.uint8)))
    blocks = np.zeros((len(world), 4), np.float32)
    blocks[:, 3] = -1e6  # no atlas: newest camera projector, captured RGB if occluded later
    return world, normals, colours, triangles[valid].astype(np.uint32), blocks


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
