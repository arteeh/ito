"""A causal 8x8, 45-degree ToF intervention; no invented dense object boundaries."""

import numpy as np


def tof(source_rows, source, row, k, origin, rng):
    # The 15 Hz sensor integrates before its 100 ms transport delay. Use only that prefix.
    stamp = np.floor((row["captured"] - 0.100) * 15) / 15
    captured = np.array([r["captured"] - origin for r in source_rows])
    index = np.searchsorted(captured, stamp, side="right") - 1
    result = np.zeros((k.height, k.width), np.float32)
    if index < 0:
        return result, None
    sample = np.load(source / source_rows[index]["file"])
    # Each square angular zone reports a robust axial depth over its footprint.
    edges = np.linspace(-np.tan(np.deg2rad(22.5)), np.tan(np.deg2rad(22.5)), 9)
    camera = np.linalg.inv(np.load(source / row["file"])["camera"]) @ sample["camera"]
    for y in range(8):
        for x in range(8):
            u0, u1 = np.round(k.cx + k.fx * edges[x : x + 2]).astype(int)
            v0, v1 = np.round(k.cy + k.fy * edges[y : y + 2]).astype(int)
            if not (0 <= (u0 + u1) / 2 < k.width and 0 <= (v0 + v1) / 2 < k.height):
                continue
            patch = sample["depth"][max(0, v0) : min(k.height, v1), max(0, u0) : min(k.width, u1)]
            valid = patch[np.isfinite(patch) & (patch > 0) & (patch <= 4)]
            if not len(valid):
                continue
            z = float(np.median(valid))
            z = max(0.05, z + rng.normal(0, 0.020 + 0.01 * z))
            point = np.array(
                [(edges[x] + edges[x + 1]) / 2 * z, -(edges[y] + edges[y + 1]) / 2 * z, -z, 1]
            )
            point = camera @ point
            if point[2] >= -0.01:
                continue
            u, v = np.round(
                [k.fx * point[0] / -point[2] + k.cx, k.cy - k.fy * point[1] / -point[2]]
            ).astype(int)
            if 0 <= u < k.width and 0 <= v < k.height:
                result[v, u] = -point[2]
    return result, int(source_rows[index]["source_frame_id"])
