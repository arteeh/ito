# ruff: noqa: E501
"""Causal prefix-map score at the baseline's exact 1-second prediction targets.
Precomputed causal MASt3R clouds and poses are revealed at capture+120ms. Their GPU
extraction cost is NOT charged: this is an optimistic online optimizer experiment.
"""

import argparse
import json
import math
import time

import numpy as np
import torch
from gsplat.strategy.ops import _update_param_with_optimizer
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from train import C0, FILES, Dataset, Model, ssim


@torch.no_grad()
def append(model, p, c):
    if not len(p):
        return
    distances = cKDTree(p).query(p, k=min(4, len(p)))[0][:, 1:]
    scale = np.clip(np.sqrt((distances**2).mean(1)), 0.002, 0.08)
    n = len(p)
    values = {
        "means": p,
        "scales": np.log(scale)[:, None].repeat(3, 1),
        "quats": np.tile([1.0, 0, 0, 0], (n, 1)),
        "opacities": np.full(n, math.log(0.1 / 0.9)),
        "sh0": ((c - 0.5) / C0)[:, None],
        "shN": np.zeros((n, 15, 3)),
    }
    tensors = {k: torch.tensor(v, dtype=torch.float32, device="cuda") for k, v in values.items()}

    def param_fn(name, p):
        return torch.nn.Parameter(torch.cat([p, tensors[name]], 0))

    def optimizer_fn(key, v):
        return torch.cat([v, torch.zeros((n, *v.shape[1:]), device="cuda")])

    _update_param_with_optimizer(param_fn, optimizer_fn, model.p, model.opt)
    for k, v in model.state.items():
        if isinstance(v, torch.Tensor):
            model.state[k] = torch.cat([v, torch.zeros((n, *v.shape[1:]), device="cuda")])
    model.update_filter()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--pose", default="true")
    parser.add_argument(
        "--init", default="mast3r", choices=["mast3r", "sparse", "calibrated", "online"]
    )
    args = parser.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    rng = np.random.default_rng(42)
    data = Dataset(args.run, args.pose, args.init)
    model = Model(data, limit=0, maxpoints=2000)
    suffix = "" if args.init == "mast3r" else "-" + args.init
    out = data.out / (args.pose + "-stream" + suffix)
    out.mkdir(parents=True, exist_ok=True)
    events = json.loads((data.out / "baseline/predictions.json").read_text())
    e = np.load(data.out / FILES[args.init])
    points = e["points"]
    colors = e["colors"].astype(np.float32) / 255
    ids = e["point_frames"]
    valid = np.isin(ids, data.train)
    points = points[valid]
    colors = colors[valid]
    ids = ids[valid]
    if data.point_world(args.init) != args.pose:
        transforms = data.to_pose_world(ids)
        points = np.einsum("nij,nj->ni", transforms[:, :3, :3], points) + transforms[:, :3, 3]
    unique_ids = np.unique(ids)
    inserted = {0}
    grid = {tuple(x) for x in np.floor(model.p["means"].detach().cpu().numpy() / 0.015).astype(int)}
    # Warm kernels without parameter updates.
    r, _, _ = model.render(0)
    (r.mean() + ssim(r[0], data.image(0))).backward()
    for p in model.p.values():
        p.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    elapsed = 0.4
    step = 0
    records = []
    last_insert = -1.0
    optimization_s = 0.0
    insertion_s = 0.0
    selected = set(np.linspace(0, len(events) - 1, 5).astype(int))
    started = time.perf_counter()
    for event in events:
        deadline = event["snapshot_s"]
        if deadline < elapsed - 0.2:
            continue
        while elapsed < deadline:
            available = data.train[data.times[data.train] <= max(0, min(elapsed, deadline) - 0.12)]
            if len(available) == 0:
                available = data.train[:1]
            if elapsed - last_insert > 0.5:
                begin = time.perf_counter()
                new_ids = [
                    int(i)
                    for i in unique_ids
                    if i not in inserted and data.times[i] <= elapsed - 0.12
                ]
                if new_ids:
                    candidates = np.flatnonzero(np.isin(ids, new_ids))
                    rng.shuffle(candidates)
                    chosen = []
                    for k in candidates:
                        key = tuple(np.floor(points[k] / 0.015).astype(int))
                        if key not in grid:
                            chosen.append(k)
                            grid.add(key)
                        if len(chosen) >= 1500:
                            break
                    inserted.update(new_ids)
                    model.filter_ids = available
                    if len(chosen) >= 4:
                        append(model, points[chosen], colors[chosen])
                torch.cuda.synchronize()
                cost = time.perf_counter() - begin
                elapsed += cost
                insertion_s += cost
                last_insert = elapsed
            recent = available[data.times[available] > elapsed - 2]
            pool = recent if len(recent) and rng.random() < 0.5 else available
            i = int(rng.choice(pool))
            torch.cuda.synchronize()
            began = time.perf_counter()
            model.step(i, step)
            torch.cuda.synchronize()
            cost = time.perf_counter() - began
            elapsed += cost
            optimization_s += cost
            step += 1
        # Only scoring accesses future target image/pose. No gradients, no future filter views.
        i = event["frame"]
        baseline = np.load(data.out / "baseline" / f"predict-{event['number']:03d}.npz")
        with torch.no_grad():
            r, _a, _ = model.render(i)
            r = r[0].clamp(0, 1)
            target = data.image(i)
            covered = torch.tensor(baseline["covered"], device="cuda")
            psnr = float(-10 * torch.log10((r[covered] - target[covered]).square().mean()))
            full = float(-10 * torch.log10((r - target).square().mean()))
            quality = float(ssim(r, target))
            record = dict(
                **event,
                gs_psnr=psnr,
                gs_full_psnr=full,
                gs_ssim=quality,
                actual_snapshot_s=elapsed,
                steps=step,
                gaussians=len(model.p["means"]),
                gpu_mb=torch.cuda.max_memory_allocated() / 2**20,
            )
            records.append(record)
            if event["number"] in selected:
                arrays = [
                    data.images[i].numpy(),
                    (r.cpu().numpy() * 255).astype(np.uint8),
                    baseline["image"],
                ]
                sheet = Image.fromarray(np.concatenate(arrays, axis=1))
                ImageDraw.Draw(sheet).text(
                    (8, 8),
                    f"REAL | CAUSAL 3DGS {psnr:.2f} dB | ITO {event['psnr']:.2f} dB; frame {i + 1}; map t={deadline:.2f}s",
                    fill="white",
                    stroke_width=2,
                    stroke_fill="black",
                )
                sheet.save(out / f"predict-{event['number']:03d}-frame-{i + 1:06d}.png")
        if len(records) % 10 == 0:
            print(
                args.run,
                args.pose,
                len(records),
                len(events),
                "psnr",
                round(psnr, 2),
                "gaussians",
                len(model.p["means"]),
                flush=True,
            )
        (out / "predictions.json").write_text(json.dumps(records, indent=2))
    model.save(out)
    report = {
        "recording": args.run,
        "pose": args.pose,
        "initialization": args.init,
        "predictions": len(records),
        "prediction_psnr_median": float(np.median([r["gs_psnr"] for r in records])),
        "prediction_psnr_mean": float(np.mean([r["gs_psnr"] for r in records])),
        "ssim_mean": float(np.mean([r["gs_ssim"] for r in records])),
        "baseline_same_targets_median": float(np.median([r["psnr"] for r in records])),
        "optimization_s": optimization_s,
        "insertion_s": insertion_s,
        "wall_s": time.perf_counter() - started,
        "precomputed_depth_and_pose_cost_excluded": True,
        "render": model.benchmark(),
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print("DONE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
