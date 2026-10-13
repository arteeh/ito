"""Actual gsplat forward/backward/Adam load for the shared-GPU trace (Linux CUDA)."""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sequence", default="tour")
    parser.add_argument("--batch", type=int, default=32768)
    parser.add_argument("--patch", type=int, default=128)
    parser.add_argument("--admit", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_CACHE_PATH"] = str(args.work / "m2-out/cuda-cache")
    import torch
    from gsplat import rasterization

    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    index = 692 if args.sequence == "sweep180" else 2283
    snapshot = np.load(
        args.work / f"m1-out/verified/repeat-1/{args.sequence}/snapshot-{index:06d}.npz"
    )
    records = snapshot["records"]
    records = records[np.linspace(0, len(records) - 1, min(args.batch, len(records)), dtype=int)]
    camera = snapshot["pose"] @ np.diag([1, -1, -1, 1])
    view = torch.tensor(np.linalg.inv(camera), device="cuda", dtype=torch.float32)[None]
    recording = args.work / f"rf-run/out/{args.sequence}/rec"
    intr = json.loads((recording / "camera.json").read_text())
    size = args.patch
    left, top = (intr["width"] - size) // 2, (intr["height"] - size) // 2
    K = torch.tensor(
        [[intr["fx"], 0, intr["cx"] - left], [0, intr["fy"], intr["cy"] - top], [0, 0, 1]],
        device="cuda",
        dtype=torch.float32,
    )[None]
    rows = [json.loads(s) for s in (recording / "frames.jsonl").read_text().splitlines()]
    image = np.asarray(Image.open(recording / rows[index]["name"]).convert("RGB")).copy()
    target = (
        torch.tensor(
            image[top : top + size, left : left + size], device="cuda", dtype=torch.float32
        )
        / 255
    )
    params = [
        torch.nn.Parameter(torch.tensor(a.copy(), device="cuda"))
        for a in (
            records[:, 0, :3],
            records[:, 2, :],
            np.log(np.maximum(records[:, 1, :3], 1e-6)),
            np.log(
                np.clip(records[:, 0, 3], 0.001, 0.999)
                / (1 - np.clip(records[:, 0, 3], 0.001, 0.999))
            ),
            records[:, 3, :3] * 0.2820947918 + 0.5,
        )
    ]
    optimizer = torch.optim.Adam(params, lr=1e-5)
    grant = [0.0]
    pipe_closed = threading.Event()

    def receive():
        for line in sys.stdin:
            grant[0] = float(line)
        pipe_closed.set()

    if args.admit:
        threading.Thread(target=receive, daemon=True).start()
    costs = []
    estimate = 0.003
    grant_overruns = 0
    start = time.perf_counter()
    try:
        while time.perf_counter() - start < 1000 and not (args.out / "stop").exists():
            deadline = float("inf")
            if args.admit and len(costs) >= 10:
                while not (args.out / "stop").exists() and not pipe_closed.is_set():
                    deadline = grant[0]
                    if time.time() + estimate < deadline < time.time() + 0.02:
                        break
                    time.sleep(0.0005)  # Wait for the cross-OS display grant, not a duty cycle.
                if (args.out / "stop").exists() or pipe_closed.is_set():
                    break
            began = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            means, quats, scales, opacity, colors = params
            rgb, _, _ = rasterization(
                means,
                quats,
                scales.exp(),
                opacity.sigmoid(),
                colors,
                view,
                K,
                size,
                size,
                packed=False,
                rasterize_mode="antialiased",
            )
            loss = (rgb[0] - target).square().mean()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - began
            costs.append(elapsed * 1000)
            estimate = max(0.0005, min(0.003, max(elapsed * 1.25, estimate * 0.95)))
            grant_overruns += int(args.admit and time.time() > deadline)
            if torch.cuda.memory_reserved() >= 2 * 1024**3:
                raise RuntimeError("Training exceeded its 2 GiB reservation allowance")
            if len(costs) == 10:
                (args.out / "ready").touch()
                while not (args.out / "start").exists():
                    if (args.out / "stop").exists() or time.perf_counter() - start > 180:
                        return
                    time.sleep(0.01)
    finally:
        (args.out / "training.json").write_text(
            json.dumps(
                dict(
                    kind="gsplat forward/backward/Adam, one local patch, no duty sleeps",
                    admission=args.admit,
                    grant_overruns=grant_overruns,
                    gaussians=len(records),
                    patch=size,
                    steps=len(costs),
                    step_ms=costs,
                    torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
