"""Causal mapper-only replay on go/no-go's prediction targets and coverage masks.

python e2e/mapper_prefix.py RECORDING EVIDENCE [--cloud true-extracted.npz] --out OUT

EVIDENCE is a docs/gsplat-gonogo.md output directory: data.npz (true poses, times), a
MASt3R cloud (camera points per frame and the poses they were placed with) and
baseline/predictions.json + predict-NNN.npz (main's point-splat prediction and coverage
at each 1-second-ahead target). Frames reach the mapper at capture + --input-delay; a
frame becomes a keyframe (seeds Gaussians) when --keyframe-interval has passed since the
last one, otherwise it only trains. Optimisation runs --duty of the (simulated) time, the
rest is the tracker's. The front end's own GPU cost is not charged: this isolates map
quality from tracking; e2e/slam_replay.py measures the whole system.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ito.protocol import Intrinsics
from ito.reconstruction.mapper import GaussianMap, Keyframe, ssim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--cloud", default="true-extracted.npz")
    parser.add_argument("--pose", choices=("true", "cloud"), default="true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--input-delay", type=float, default=0.12)
    parser.add_argument("--keyframe-interval", type=float, default=0.3)
    parser.add_argument("--views", choices=("cloud", "all"), default="cloud")
    parser.add_argument("--events", type=int, default=0)
    parser.add_argument("--duty", type=float, default=0.25)
    parser.add_argument("--budget", type=int, default=400_000)
    parser.add_argument("--set", action="append", default=[], help="mapper constant: SEED_MAX=9000")
    args = parser.parse_args()
    from ito.reconstruction import mapper

    for override in args.set:
        name, value = override.split("=", 1)
        setattr(mapper, name, type(getattr(mapper, name))(value))
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    intr = Intrinsics.model_validate_json((args.recording / "camera.json").read_text())
    rows = [json.loads(s) for s in (args.recording / "frames.jsonl").read_text().splitlines()]
    data = np.load(args.evidence / "data.npz")
    stamps = data["times"] - data["times"][0]
    truth = data["poses"]
    cloud = np.load(args.evidence / args.cloud)
    estimates = cloud["estimates"]
    if args.cloud.startswith(("online", "fused")):
        from truth import align

        estimates = np.linalg.inv(align(estimates[0], truth[int(cloud["ids"][0])])) @ estimates
    cloud_pose = {int(n): p for n, p in zip(cloud["ids"], estimates, strict=True)}
    events = json.loads((args.evidence / "baseline/predictions.json").read_text())
    if args.events:
        events = events[: args.events]
    with_points = set(np.unique(cloud["point_frames"]).astype(int).tolist())
    ids = np.array(sorted(with_points)) if args.views == "cloud" else np.arange(len(rows))
    # Every eighth image stays held out from training and seeding (go/no-go's split).
    ids = ids[ids % 8 != 7]
    model = GaussianMap(args.budget)
    # Warm the CUDA kernels outside the clock, as a running app has long since done.
    warm = model.tensor(np.zeros((intr.height, intr.width, 3)))
    model._append(
        dict(
            means=model.tensor([[0, 0, -1.0]]),
            scales=model.tensor([[-4.0] * 3]),
            quats=model.tensor([[1.0, 0, 0, 0]]),
            opacities=model.tensor([0.0]),
            sh0=model.tensor([[[0.0] * 3]]),
            owner=torch.zeros(1, device="cuda", dtype=torch.long),
        ),
        {},
    )
    pred = model.render(np.eye(4), intr, train=True)[0]
    (pred - warm).abs().mean().backward()
    model._remove(torch.ones(1, dtype=torch.bool, device="cuda"))
    for optimizer in model.optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    elapsed, cursor, optimization_s, keyframe_at, anchor = 0.0, 0, 0.0, -np.inf, None
    records, keyframes = [], 0
    for event in events:
        deadline = event["snapshot_s"]
        while elapsed < deadline:
            began = time.perf_counter()
            while cursor < len(ids) and stamps[ids[cursor]] + args.input_delay <= elapsed:
                index = int(ids[cursor])
                cursor += 1
                rgb = np.asarray(Image.open(args.recording / rows[index]["name"]).convert("RGB"))
                pose = truth[index] if args.pose == "true" else cloud_pose.get(index)
                if pose is None:
                    continue
                key = index in with_points and stamps[index] - keyframe_at >= args.keyframe_interval
                if not key and anchor is None:
                    continue
                local = colors = None
                if key:
                    selected = cloud["point_frames"] == index
                    source = cloud_pose[index]
                    local = (cloud["points"][selected] - source[:3, 3]) @ source[:3, :3]
                    colors = cloud["colors"][selected]
                    keyframe_at, anchor = stamps[index], index
                    keyframes += 1
                model.add(
                    Keyframe(
                        index,
                        rgb,
                        pose,
                        intr,
                        local,
                        colors,
                        None if local is None else np.ones(len(local)),
                        float(stamps[index]),
                        anchor=index if key else anchor,
                    )
                )
            if model.count:
                model.step()
            torch.cuda.synchronize()
            cost = time.perf_counter() - began
            optimization_s += cost
            elapsed += max(0.02, cost / args.duty)
        index = int(event["frame"])
        target = np.asarray(Image.open(args.recording / rows[index]["name"]).convert("RGB"))
        target_gpu = model.tensor(target) / 255
        baseline = np.load(args.evidence / f"baseline/predict-{event['number']:03d}.npz")
        mask = torch.as_tensor(baseline["covered"], device="cuda")
        # The frozen prefix at the target's true view: nothing after the snapshot is in it.
        with torch.no_grad():
            pred, _, alpha, _ = model.render(truth[index], intr)
            pred = pred.clamp(0, 1)
            base = model.tensor(baseline["image"]) / 255

            def psnr(x, m=None, target=target_gpu):
                d = (x - target).square()
                return float(-10 * torch.log10(d[m].mean() if m is not None else d.mean()))

            row = dict(
                event,
                mapper_psnr=psnr(pred, mask),
                baseline_psnr=psnr(base, mask),
                mapper_full_psnr=psnr(pred),
                baseline_full_psnr=psnr(base),
                mapper_ssim=float(ssim(pred, target_gpu)),
                baseline_ssim=float(ssim(base, target_gpu)),
                mapper_coverage=float((alpha > 0.5).float().mean()),
                gaussians=model.count,
                keyframes=keyframes,
                views=len(model.views),
                steps=model.steps,
                optimization_s=optimization_s,
                vram_mb=torch.cuda.max_memory_allocated() / 2**20,
            )
        records.append(row)
        if event["number"] % 5 == 0:
            side = np.concatenate(
                (baseline["image"], (pred.cpu().numpy() * 255).astype(np.uint8), target), axis=1
            )
            Image.fromarray(side).save(args.out / f"predict-{event['number']:03d}.png")
        print(
            json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}),
            flush=True,
        )

    def median(key):
        return round(float(np.median([r[key] for r in records])), 2)

    report = dict(
        kind="mapper-only; front-end GPU cost not charged",
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        **{
            k: median(k)
            for k in (
                "mapper_psnr",
                "baseline_psnr",
                "mapper_full_psnr",
                "baseline_full_psnr",
                "mapper_ssim",
                "baseline_ssim",
            )
        },
        gaussians=model.count,
        vram_mb=round(torch.cuda.max_memory_allocated() / 2**20),
        frames=records,
    )
    (args.out / "result.json").write_text(json.dumps(report, indent=2))
    print("SUMMARY", json.dumps({k: v for k, v in report.items() if k != "frames"}))


if __name__ == "__main__":
    main()
