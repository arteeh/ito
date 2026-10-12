"""Use training-only triangulated depths to calibrate MASt3R's monocular depth scale."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
run = sys.argv[1]
out = ROOT / "out" / run
poses = np.load(out / "data.npz")["poses"]
mast = np.load(out / "true-extracted.npz")
sparse = np.load(out / "triangulated.npz")
results = []
for i in np.unique(sparse["point_frames"]):
    a = mast["points"][mast["point_frames"] == i]
    b = sparse["points"][sparse["point_frames"] == i]
    if len(a) < 5 or len(b) < 4:
        continue
    pose = poses[i]
    a = (a - pose[:3, 3]) @ pose[:3, :3]
    b = (b - pose[:3, 3]) @ pose[:3, :3]
    distance, nearest = cKDTree(a[:, :2] / a[:, 2:]).query(b[:, :2] / b[:, 2:])
    ratio = b[:, 2] / a[nearest, 2]
    valid = (distance < 0.04) & (ratio > 0.1) & (ratio < 100)
    if valid.sum() < 4:
        continue
    values = ratio[valid]
    results.append(
        {
            "frame": int(i),
            "ratio": float(np.median(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
            "matches": int(valid.sum()),
        }
    )
print(json.dumps(results[:10], indent=2))
print(
    "median depth ratio",
    np.median([x["ratio"] for x in results]),
    "range",
    np.quantile([x["ratio"] for x in results], [0.1, 0.9]),
    "frames",
    len(results),
    flush=True,
)
(out / "init-depth-scale.json").write_text(json.dumps(results, indent=2))
# Piecewise-linear interpolation from independently triangulated training-frame scales.
ids = mast["point_frames"]
factors = np.interp(ids, [x["frame"] for x in results], [x["ratio"] for x in results])
centers = poses[ids, :3, 3]
p = centers + (mast["points"] - centers) * factors[:, None]
np.savez(
    out / "calibrated.npz",
    points=p.astype(np.float32),
    colors=mast["colors"],
    point_frames=ids,
)
