# ruff: noqa: E402
"""Depth scale of a MASt3R cloud against training-only triangulated points (true poses).
true-extracted: the oracle (L1) cloud, placed with true poses, frontend scale never corrected.
online-extracted: Ito's own fused cloud (fused poses, fused scale), camera frame from the
aligned fused estimate of the same frame."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "repo/e2e"))
from truth import align  # noqa: E402

run = sys.argv[1]
out = ROOT / "out" / run
truth = np.load(out / "data.npz")["poses"]
sparse = np.load(out / "triangulated.npz")
fused = np.load(out / "fused-extracted.npz")
est = np.linalg.inv(align(fused["estimates"][0], truth[fused["ids"][0]])) @ fused["estimates"]
fused_pose = dict(zip(fused["ids"].tolist(), est, strict=True))
report = {}
for name, poses in [("true-extracted", None), ("online-extracted", fused_pose)]:
    cloud = np.load(out / f"{name}.npz")
    ratios = []
    frames = np.unique(cloud["point_frames"])
    for i in np.unique(sparse["point_frames"]):
        # Nearest frame (within 3) that has cloud points; compare in that camera.
        j = int(frames[np.argmin(np.abs(frames - i))])
        if abs(j - i) > 3 or (poses is not None and j not in poses):
            continue
        a = cloud["points"][cloud["point_frames"] == j]
        b = sparse["points"][sparse["point_frames"] == i]
        if len(a) < 5 or len(b) < 4:
            continue
        pa = truth[j] if poses is None else poses[j]
        a = (a - pa[:3, 3]) @ pa[:3, :3]
        b = (b - truth[j][:3, 3]) @ truth[j][:3, :3]
        distance, nearest = cKDTree(a[:, :2] / a[:, 2:]).query(b[:, :2] / b[:, 2:])
        ratio = b[:, 2] / a[nearest, 2]
        valid = (distance < 0.04) & (ratio > 0.1) & (ratio < 100)
        if valid.sum() >= 4:
            ratios.append(float(np.median(ratio[valid])))
    report[name] = {
        "frames": len(ratios),
        "triangulated_over_cloud_depth_p10_p50_p90": np.percentile(ratios, [10, 50, 90])
        .round(3)
        .tolist()
        if ratios
        else None,
    }
(out / "scale-check.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
