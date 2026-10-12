# ruff: noqa: E402
"""Reconstruct L1 at its saved processing schedule; keep exact prediction masks/targets."""

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
from slam_replay import AHEAD, render

from ito.protocol import Intrinsics, RobotDescription
from ito.reconstruction.ring import MOTION_STATES
from ito.reconstruction.slam import SLAMBackend

run = sys.argv[1]
source = ROOT.parent / "rf-run/out" / run
out = ROOT / "out" / run / "baseline"
out.mkdir(exist_ok=True, parents=True)
rows = [json.loads(x) for x in (source / "rec/frames.jsonl").read_text().splitlines()]
motion = [json.loads(x) for x in (source / "rec/motion.jsonl").read_text().splitlines()]
intr = Intrinsics.model_validate_json((source / "rec/camera.json").read_text())
robot = RobotDescription.model_validate_json((source / "rec/robot.json").read_text())
data = np.load(out.parent / "data.npz")
poses = data["poses"]
times = data["times"]
bytime = {t: i for i, t in enumerate(times)}
backend = SLAMBackend(
    1048576,
    intr,
    report=lambda x: None,
    motion_noise=robot.motion_noise,
    frame_delay=robot.cameras[0].capture_delay,
    oracle=lambda t: poses[bytime[t]],
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
schedule = json.loads((source / "replay-L1/frames.json").read_text())
pending = []
predictions = []
fed = 0
for n, item in enumerate(schedule):
    index, stamp, _, tracked, *_ = item
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
    torch.cuda.synchronize()
    if tracked:
        for due, records, live, snapshot in [p for p in pending if p[0] <= stamp]:
            pending.remove((due, records, live, snapshot))
            if stamp - due > 0.5:
                continue
            image, covered = render(records, live, poses[index], intr, (rgb.shape[1], rgb.shape[0]))
            if covered.mean() > 0.05:
                error = ((image[covered] - rgb[covered] / 255) ** 2).mean()
                psnr = float(-10 * np.log10(error))
                number = len(predictions)
                result = {
                    "number": number,
                    "snapshot_s": snapshot,
                    "target_s": stamp,
                    "frame": index,
                    "capture_s": float(times[index] - times[0]),
                    "coverage": float(covered.mean()),
                    "psnr": psnr,
                }
                predictions.append(result)
                np.savez_compressed(
                    out / f"predict-{number:03d}.npz",
                    image=(image * 255).astype(np.uint8),
                    covered=covered,
                )
        if not pending or stamp - pending[-1][0] + AHEAD > 0.5:
            pending.append((stamp + AHEAD, backend.records.copy(), backend.keys >= 0, stamp))
    backend.expire(stamp, np.inf)
    if n % 50 == 0:
        print(run, n, len(schedule), "predictions", len(predictions), flush=True)
(out / "predictions.json").write_text(json.dumps(predictions, indent=2))
print("DONE", run, "median", np.median([r["psnr"] for r in predictions]), flush=True)
