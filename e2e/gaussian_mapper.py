"""Drive the real Gaussian engine: training, correction, eviction, alignment and deltas.

Run under the Ceres GPU lock: python e2e/gaussian_mapper.py --out out/mapper-check.
Needs precompiled gsplat (the release models bundle or an installed binary wheel).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ito.protocol import Intrinsics
from ito.reconstruction.mapper import GaussianMap, Keyframe
from ito.reconstruction.mapper_online import OnlineMapper
from ito.render.scene import GaussianBuffer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("out/mapper-check"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    i = Intrinsics(width=96, height=72, fx=82, fy=82, cx=48, cy=36)
    y, x = np.mgrid[: i.height, : i.width]
    z = 2 + 0.3 * np.sin(x * 0.09) + 0.2 * np.cos(y * 0.08)
    points = (
        np.stack(((x - i.cx) / i.fx, -(y - i.cy) / i.fy, -np.ones_like(x)), -1) * z[..., None]
    ).reshape(-1, 3)
    rgb = np.stack(
        (
            100 + 90 * np.sin(x * 0.15),
            120 + 80 * np.sin(y * 0.16),
            120 + 90 * np.cos((x + y) * 0.11),
        ),
        -1,
    ).astype(np.uint8)
    frame = Keyframe(
        1,
        rgb,
        np.eye(4),
        i,
        points,
        rgb.reshape(-1, 3),
        np.ones(len(points)),
        time.perf_counter(),
        z.astype(np.float32),
    )
    model = GaussianMap(8000)
    model.add(frame)

    def score(pred):
        return float(-10 * torch.log10((pred - model.tensor(rgb) / 255).square().mean()))

    with torch.no_grad():
        before = score(model.render(frame.pose, i)[0])
    began = time.perf_counter()
    for _ in range(120):
        model.step()
    torch.cuda.synchronize()
    training_s = time.perf_counter() - began
    with torch.no_grad():
        rendered = model.render(frame.pose, i)[0]
        after = score(rendered)
        Image.fromarray((rendered.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(
            args.out / "trained.png"
        )
    assert after > before + 1, (before, after)
    records = model.records(1)
    GaussianBuffer(records)
    shifted = np.eye(4, dtype=np.float32)
    shifted[:3, 3] = [0.1, -0.2, 0.3]
    model.correct({1: shifted})
    corrected = model.records(1)
    assert len(records) == len(corrected)
    assert np.allclose(corrected[:, 0, :3], records[:, 0, :3] + shifted[:3, 3], atol=1e-5)
    model.correct({1: frame.pose})
    # Hold the map fixed and perturb a rendered view to exercise all pose derivatives.
    target = (rendered.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    initial = np.eye(4, dtype=np.float32)
    initial[0, 3] = 0.025
    began = time.perf_counter()
    aligned, sigma, inliers = model.align(target, initial, i, steps=80)
    alignment_s = time.perf_counter() - began
    assert inliers > 128 and np.isfinite(sigma).all(), (sigma, inliers)
    assert np.linalg.norm(aligned[:3, 3]) < 0.025, aligned
    assert model.count <= model.budget
    model.forget([1])
    assert model.count == 0 and model.records(1).shape == (0, 4, 4)

    online = OnlineMapper(8000, print)
    try:
        online.submit(frame)
        deadline = time.perf_counter() + 15
        while not online.count and time.perf_counter() < deadline:
            online.expire(0, 0)
            time.sleep(0.01)
        assert online.count > 0, online.failure
        online.forget([1])
        while online.count and time.perf_counter() < deadline:
            online.expire(0, 0)
            time.sleep(0.01)
        assert online.count == 0
    finally:
        online.close()
    report = dict(
        before_psnr=before,
        after_psnr=after,
        training_s=training_s,
        count=len(records),
        alignment_s=alignment_s,
        inliers=inliers,
        aligned_translation=aligned[:3, 3].tolist(),
        sigma=sigma.tolist(),
        vram_mb=torch.cuda.max_memory_allocated() / 2**20,
    )
    (args.out / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    print("PASS training, metric anchor correction, budget, alignment, asynchronous forget")


if __name__ == "__main__":
    main()
