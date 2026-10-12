# ruff: noqa: E402
"""Read-only source recordings/runtime; experiment artifacts stay in gs."""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["CUPY_CACHE_DIR"] = str(ROOT / "cache/cupy")
os.environ["CUDA_CACHE_PATH"] = str(ROOT / "cache/cuda-windows")
os.environ["TORCH_HOME"] = str(ROOT / "cache/torch")
sys.path[:0] = [str(ROOT / "repo"), str(ROOT / "repo/e2e")]
import numpy as np
from PIL import Image
from truth import Truth

from ito.reconstruction import mast3r_runtime

mast3r_runtime.MODELS = ROOT.parent / "rf/models"
from ito.protocol import Intrinsics, RobotDescription
from ito.reconstruction.ring import MOTION_STATES
from ito.reconstruction.slam import SLAMBackend

run = sys.argv[1]
mode = sys.argv[2]
source = ROOT.parent / "rf-run/out" / run
out = ROOT / "out" / run
out.mkdir(exist_ok=True, parents=True)
rows = [json.loads(x) for x in (source / "rec/frames.jsonl").read_text().splitlines()]
motion = [json.loads(x) for x in (source / "rec/motion.jsonl").read_text().splitlines()]
intr = Intrinsics.model_validate_json((source / "rec/camera.json").read_text())
robot = RobotDescription.model_validate_json((source / "rec/robot.json").read_text())
times = np.array([r["captured"] for r in rows])
if not (out / "data.npz").exists():
    truth = Truth(source / "truth.jsonl", motion)
    poses = []
    errors = []
    exposures = []
    for r in rows:
        found = truth.match(
            np.asarray(Image.open(source / "rec" / r["name"]).convert("RGB")),
            r["captured"],
        )
        if found is None:
            raise RuntimeError("Unmatched truth")
        t, p, e = found
        poses.append(p)
        errors.append(e)
        exposures.append(t)
    np.savez(out / "data.npz", poses=poses, times=times, exposures=exposures, errors=errors)
    print(
        "TRUTH",
        len(rows),
        "turns",
        truth.turns,
        "errors",
        np.percentile(errors, [50, 95]),
        "size",
        Image.open(source / "rec" / rows[0]["name"]).size,
        flush=True,
    )
poses = np.load(out / "data.npz")["poses"]
bytime = {t: i for i, t in enumerate(times)}
points = []
colors = []
indices = []
estimates = []
ids = []


class Capture(SLAMBackend):
    def place(self, cloud, pose, when):
        if mode in ("true", "online"):
            p = cloud.points[cloud.valid].cpu().numpy()
            c = cloud.colors[cloud.valid].cpu().numpy()
            take = np.arange(0, len(p), max(1, len(p) // 1500))
            points.append((p[take] @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32))
            colors.append(c[take])
            indices.extend([index] * len(take))
        # No map rendering needed during pose/cloud extraction.


backend = Capture(
    1024,
    intr,
    report=lambda x: None,
    motion_noise=robot.motion_noise,
    frame_delay=robot.cameras[0].capture_delay,
    oracle=(lambda t: poses[bytime[t]]) if mode == "true" else None,
)
samples = np.array(
    [
        [
            m["local"],
            *m["camera"],
            *m["angular_velocity"],
            MOTION_STATES.index(m["state"]),
        ]
        for m in motion
    ]
)
if mode == "true":
    schedule = [(i, times[i] - times[0]) for i in range(0, len(rows), 6) if (i + 1) % 8]
else:
    schedule = [(r[0], r[1]) for r in json.loads((source / "replay-L4/frames.json").read_text())]
fed = 0
began = time.perf_counter()
for n, (index, stamp) in enumerate(schedule):
    now = times[0] + stamp
    arrived = np.searchsorted(samples[:, 0], now, side="right")
    backend.add_motion(samples[fed:arrived])
    fed = max(fed, arrived)
    rgb = np.asarray(Image.open(source / "rec" / rows[index]["name"]).convert("RGB"))
    backend.integrate(
        rgb,
        None,
        np.array(rows[index]["camera"]).reshape(4, 4),
        stamp,
        captured=times[index],
    )
    estimates.append(backend.camera_pose.copy())
    ids.append(index)
    if n % 50 == 0:
        print(
            mode,
            run,
            n,
            len(schedule),
            "seconds",
            round(time.perf_counter() - began, 1),
            flush=True,
        )
if mode == "online":
    # Ito's own cloud: fused poses and fused MASt3R scale. Align its world to truth's the
    # way train.py aligns the fused poses, so both live in the same frame.
    from truth import align

    alignment = np.linalg.inv(align(estimates[0], poses[ids[0]]))
    points = [p @ alignment[:3, :3].T.astype(np.float32) + alignment[:3, 3] for p in points]
np.savez(
    out / (mode + "-extracted.npz"),
    ids=ids,
    estimates=estimates,
    points=np.concatenate(points) if points else [],
    colors=np.concatenate(colors) if colors else [],
    point_frames=indices,
)
print("DONE", mode, run, "seconds", time.perf_counter() - began, flush=True)
