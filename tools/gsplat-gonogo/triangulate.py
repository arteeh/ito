"""Training-only SIFT triangulation with known camera poses; no pose estimation or GPU."""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
cv2.setNumThreads(2)
run = sys.argv[1]
source = ROOT.parent / "rf-run/out" / run
out = ROOT / "out" / run
rows = [json.loads(x) for x in (source / "rec/frames.jsonl").read_text().splitlines()]
poses = np.load(out / "data.npz")["poses"] @ np.diag([1, -1, -1, 1])
views = np.linalg.inv(poses)
intr = json.loads((source / "rec/camera.json").read_text())
K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]])
train = [i for i in range(0, len(rows), 6) if (i + 1) % 8]
sift = cv2.SIFT_create(nfeatures=3500)
matcher = cv2.BFMatcher()
cache = {}


def features(i):
    if i not in cache:
        im = cv2.imread(str(source / "rec" / rows[i]["name"]))
        k, d = sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None)
        cache[i] = (np.array([x.pt for x in k]), d, im[:, :, ::-1])
    return cache[i]


points = []
colors = []
frames = []
errors = []
begin = time.perf_counter()
pairs = 0
for j, i in enumerate(train[::2]):
    pa, da, im = features(i)
    for delta in [12, 30, 60]:
        b = i + delta
        if b >= len(rows) or (b + 1) % 8 == 0:
            continue
        if np.linalg.norm(poses[i, :3, 3] - poses[b, :3, 3]) < 0.015:
            continue
        pb, db, _ = features(b)
        if da is None or db is None:
            continue
        matches = [
            a for a, bmatch in matcher.knnMatch(da, db, k=2) if a.distance < 0.72 * bmatch.distance
        ]
        if len(matches) < 8:
            continue
        a = np.array([pa[m.queryIdx] for m in matches])
        bxy = np.array([pb[m.trainIdx] for m in matches])
        P1 = K @ views[i, :3]
        P2 = K @ views[b, :3]
        q = cv2.triangulatePoints(P1, P2, a.T, bxy.T)
        q = (q[:3] / q[3]).T
        x = q @ views[i, :3, :3].T + views[i, :3, 3]
        y = q @ views[b, :3, :3].T + views[b, :3, 3]
        ax = x @ K.T
        by = y @ K.T
        err = np.maximum(
            np.linalg.norm(ax[:, :2] / ax[:, 2:] - a, axis=1),
            np.linalg.norm(by[:, :2] / by[:, 2:] - bxy, axis=1),
        )
        ray1 = q - poses[i, :3, 3]
        ray2 = q - poses[b, :3, 3]
        cosine = (ray1 * ray2).sum(1) / (
            np.linalg.norm(ray1, axis=1) * np.linalg.norm(ray2, axis=1)
        )
        valid = (
            np.isfinite(q).all(1)
            & (err < 1.5)
            & (x[:, 2] > 0.08)
            & (y[:, 2] > 0.08)
            & (x[:, 2] < 12)
            & (y[:, 2] < 12)
            & (cosine < np.cos(np.radians(0.5)))
        )
        q = q[valid]
        xy = np.rint(a[valid]).astype(int)
        color = im[xy[:, 1], xy[:, 0]]
        points.append(q)
        colors.append(color)
        frames.extend([i] * len(q))
        errors.extend(err[valid])
        pairs += 1
    if j % 25 == 0:
        print(
            run,
            j,
            "pairs",
            pairs,
            "points",
            len(frames),
            "seconds",
            round(time.perf_counter() - begin, 1),
            flush=True,
        )
    # Keep a bounded image/features cache.
    if j % 20 == 0:
        cache = {k: v for k, v in cache.items() if k >= i}
np.savez(
    out / "triangulated.npz",
    points=np.concatenate(points).astype(np.float32),
    colors=np.concatenate(colors),
    point_frames=frames,
)
report = {
    "points": len(frames),
    "pairs": pairs,
    "seconds": time.perf_counter() - begin,
    "reprojection_error_px_p50": float(np.median(errors)),
    "reprojection_error_px_p95": float(np.quantile(errors, 0.95)),
    "train_only": True,
}
(out / "triangulation.json").write_text(json.dumps(report, indent=2))
print("DONE", report, flush=True)
