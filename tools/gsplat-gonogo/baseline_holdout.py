# ruff: noqa: E402
"""Build Ito's fixed point-map on stride-6 train frames only; export for held-out scoring."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["CUPY_CACHE_DIR"] = str(ROOT / "cache/cupy")
os.environ["CUDA_CACHE_PATH"] = str(ROOT / "cache/cuda-windows")
sys.path[:0] = [str(ROOT / "repo"), str(ROOT / "repo/e2e")]
import numpy as np
from PIL import Image

from ito.reconstruction import mast3r_runtime

mast3r_runtime.MODELS = ROOT.parent / "rf/models"
import torch

from ito.protocol import Intrinsics, RobotDescription
from ito.reconstruction.ring import MOTION_STATES
from ito.reconstruction.slam import SLAMBackend

run = sys.argv[1]
source = ROOT.parent / "rf-run/out" / run
out = ROOT / "out" / run / "baseline-heldout"
out.mkdir(exist_ok=True, parents=True)
rows = [json.loads(x) for x in (source / "rec/frames.jsonl").read_text().splitlines()]
motion = [json.loads(x) for x in (source / "rec/motion.jsonl").read_text().splitlines()]
intr = Intrinsics.model_validate_json((source / "rec/camera.json").read_text())
robot = RobotDescription.model_validate_json((source / "rec/robot.json").read_text())
data = np.load(out.parent / "data.npz")
poses = data["poses"]
times = data["times"]
lookup = {t: i for i, t in enumerate(times)}
backend = SLAMBackend(
    1048576,
    intr,
    report=lambda x: None,
    motion_noise=robot.motion_noise,
    frame_delay=robot.cameras[0].capture_delay,
    oracle=lambda t: poses[lookup[t]],
    voxel_size=0.04,
    fade_seconds=0.5,
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
fed = 0
for index in range(0, len(rows), 6):
    if (index + 1) % 8 == 0:
        continue
    now = times[index]
    arrived = np.searchsorted(samples[:, 0], now, side="right")
    backend.add_motion(samples[fed:arrived])
    fed = max(fed, arrived)
    rgb = np.asarray(Image.open(source / "rec" / rows[index]["name"]).convert("RGB"))
    backend.integrate(
        rgb,
        None,
        np.array(rows[index]["camera"]).reshape(4, 4),
        now - times[0],
        captured=now,
    )
    torch.cuda.synchronize()
    backend.expire(now - times[0], np.inf)
    if index % 300 == 0:
        print(run, index, len(rows), backend.count, flush=True)
np.savez_compressed(out / "map.npz", records=backend.records[backend.keys >= 0])
print("DONE", backend.count, flush=True)
