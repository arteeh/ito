# ruff: noqa: E402
"""Held-out static-map baseline using replay's nearest-point 7x7 splat rule on CUDA."""

import ast
import json
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
from train import Dataset, Model, ssim


@torch.no_grad()
def render(records, pose, K, w, h):
    position = records[:, 0, :3]
    radius = records[:, 1, 0]
    color = (records[:, 3, :3] * 0.2820947918 + 0.5).clamp(0, 1)
    cam = torch.tensor(pose, dtype=torch.float32, device="cuda")
    local = (position - cam[:3, 3]) @ cam[:3, :3]
    depth = -local[:, 2]
    ahead = depth > 1e-6
    local, depth, radius, color = (
        local[ahead],
        depth[ahead],
        radius[ahead],
        color[ahead],
    )
    u = K[0, 0] * local[:, 0] / depth + K[0, 2]
    v = K[1, 2] - K[1, 1] * local[:, 1] / depth
    pixels = (radius * K[0, 0] / depth).round().clamp(0, 3).long()
    u = u.round().long()
    v = v.round().long()
    zbuffer = torch.full((w * h,), float("inf"), device="cuda")
    image = torch.zeros((w * h, 3), device="cuda")
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            x = u + dx
            y = v + dy
            inside = (pixels >= max(abs(dx), abs(dy))) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
            index = y[inside] * w + x[inside]
            z = depth[inside]
            zbuffer.scatter_reduce_(0, index, z, reduce="amin", include_self=True)
            nearest = z <= zbuffer[index]
            image[index[nearest]] = color[inside][nearest]
    return image.reshape(h, w, 3), torch.isfinite(zbuffer).reshape(h, w)


def main():
    run = sys.argv[1]
    d = Dataset(run)
    out = d.out / "baseline-heldout"
    records = torch.tensor(np.load(out / "map.npz")["records"], device="cuda")
    metrics = []
    model = Model(d)
    ckpt = torch.load(d.out / "true/model.pt", map_location="cuda", weights_only=True)
    model.p = torch.nn.ParameterDict(
        {k: torch.nn.Parameter(v, requires_grad=False) for k, v in ckpt["params"].items()}
    )
    model.filter = ckpt["filter"]
    model.degree = ckpt["degree"]
    selected = set(d.test[np.linspace(0, len(d.test) - 1, 5).astype(int)])
    # Verify this accelerated scoring path against the exact replay function.
    source = (d.out.parents[1] / "repo/e2e/slam_replay.py").read_text()
    function = next(
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "render"
    )
    namespace = {"np": np}
    exec(  # noqa: S102 - exact local, version-pinned replay render function.
        compile(ast.Module(body=[function], type_ignores=[]), "slam_replay.render", "exec"),
        namespace,
    )
    from types import SimpleNamespace

    intr = SimpleNamespace(
        fx=float(d.K[0, 0]),
        fy=float(d.K[1, 1]),
        cx=float(d.K[0, 2]),
        cy=float(d.K[1, 2]),
    )
    subset = records[:20000]
    cpu, cmask = namespace["render"](
        subset.cpu().numpy(), np.ones(len(subset), bool), d.poses[7], intr, (d.w, d.h)
    )
    gpu, gmask = render(subset, d.poses[7], d.K, d.w, d.h)
    verification = {
        "subset_count": len(subset),
        "mean_absolute_error": float(np.abs(cpu - gpu.cpu().numpy()).mean()),
        "mask_disagreement": float(np.mean(cmask != gmask.cpu().numpy())),
    }
    (out / "renderer-verification.json").write_text(json.dumps(verification, indent=2))
    with torch.no_grad():
        for n, i in enumerate(d.test):
            r, covered = render(records, d.poses[i], d.K, d.w, d.h)
            target = d.image(i)
            p = float(-10 * torch.log10((r - target).square().mean()))
            cp = float(-10 * torch.log10((r[covered] - target[covered]).square().mean()))
            q = float(ssim(r, target))
            metrics.append(
                {
                    "frame": int(i),
                    "psnr": p,
                    "covered_psnr": cp,
                    "ssim": q,
                    "coverage": float(covered.float().mean()),
                }
            )
            if i in selected:
                gs = model.render(int(i))[0][0].clamp(0, 1)
                arrays = [
                    d.images[i].numpy(),
                    (gs.cpu().numpy() * 255).astype(np.uint8),
                    (r.cpu().numpy() * 255).astype(np.uint8),
                ]
                im = Image.fromarray(np.concatenate(arrays, 1))
                ImageDraw.Draw(im).text(
                    (8, 8),
                    f"REAL | 3DGS true 180s | ITO train-only static map; held-out frame {i + 1}",
                    fill="white",
                    stroke_width=2,
                    stroke_fill="black",
                )
                im.save(out / f"heldout-{i + 1:06d}.png")
            if n % 50 == 0:
                print(run, n, len(d.test), p, flush=True)
    report = {
        "psnr_mean": float(np.mean([x["psnr"] for x in metrics])),
        "ssim_mean": float(np.mean([x["ssim"] for x in metrics])),
        "coverage_mean": float(np.mean([x["coverage"] for x in metrics])),
        "covered_psnr_median": float(np.median([x["covered_psnr"] for x in metrics])),
        "count": len(records),
        "frames": metrics,
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print("DONE", run, report["psnr_mean"], report["ssim_mean"], flush=True)


if __name__ == "__main__":
    main()
