"""Causal depth inference only. Run through depthmesh_gpu.sh for the shared GPU lock."""

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CONFIG = {
    "vits": dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384]),
    "vitl": dict(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]),
}


def load_model(root, variant):
    encoder = "vitl" if variant.endswith("large") else "vits"
    folder = root / (
        "vda"
        if variant.startswith("video")
        else "promptda"
        if variant.startswith("prompt")
        else "da2"
    )
    if variant.startswith("metric"):
        folder /= "metric_depth"
    sys.path.insert(0, str(folder))
    checkpoint = root / "weights" / f"{variant}.{'ckpt' if variant.startswith('prompt') else 'pth'}"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing weights: {checkpoint}; run depthmesh_setup.sh")
    if variant.startswith("prompt"):
        from promptda.promptda import PromptDA

        model = PromptDA(encoder=encoder, ckpt_path=str(checkpoint))
    else:
        package = (
            "video_depth_anything.video_depth_stream"
            if variant.startswith("video")
            else "depth_anything_v2.dpt"
        )
        cls = "VideoDepthAnything" if variant.startswith("video") else "DepthAnythingV2"
        kwargs = CONFIG[encoder].copy()
        if variant.startswith("metric"):
            kwargs["max_depth"] = 20
        model = getattr(importlib.import_module(package), cls)(**kwargs)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.cuda().eval(), folder, checkpoint


def predict(model, rgb, tof, variant, size):
    if variant.startswith("video"):
        return model.infer_video_depth_one(rgb, input_size=size, device="cuda", fp32=False)
    if variant.startswith("prompt"):
        from scipy.ndimage import distance_transform_edt

        valid = (tof > 0.05) & np.isfinite(tof)
        if valid.sum() < 4:
            raise ValueError("PromptDA needs at least four valid optional ToF zones")
        # Missing zones are nearest-zone extrapolations, never dense sim-depth inputs.
        indices = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        prompt = torch.from_numpy(tof[tuple(indices)].copy()).cuda()[None, None]
        if float(prompt.max() - prompt.min()) < 0.001:
            return np.full(rgb.shape[:2], float(prompt.mean()), "f4")
        h, w = rgb.shape[:2]
        sh, sw = (
            int(round(h * size / min(h, w) / 14)) * 14,
            int(round(w * size / min(h, w) / 14)) * 14,
        )
        image = torch.from_numpy(rgb.copy()).cuda().permute(2, 0, 1)[None].float() / 255
        image = F.interpolate(image, (sh, sw), mode="bilinear", align_corners=False)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            depth = model.predict(image, prompt)
        return (
            F.interpolate(depth.float(), (h, w), mode="bilinear", align_corners=False)[0, 0]
            .cpu()
            .numpy()
        )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        return model.infer_image(rgb[..., ::-1].copy(), input_size=size)


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Learned depth needs CUDA; the mesh renderer itself supports CPU/Mesa")
    torch.set_num_threads(4)
    # Torch allocations leave room for the renderer and CUDA context under the 8GB cap.
    torch.cuda.set_per_process_memory_fraction(0.58)
    model, folder, checkpoint = load_model(args.root, args.variant)
    output = args.root / "depth" / args.variant / args.sequence
    output.mkdir(parents=True, exist_ok=True)
    source = args.root / "prepared" / args.sequence
    trajectory = np.load(source / "trajectory.npz")
    report = dict(
        variant=args.variant,
        sequence=args.sequence,
        input_size=args.size,
        precision="fp16 autocast",
        device=torch.cuda.get_device_name(),
        model_git=os.popen(f'git -C "{folder}" rev-parse HEAD').read().strip(),
        checkpoint_sha256=hashlib.file_digest(open(checkpoint, "rb"), "sha256").hexdigest(),
        causal=True,
        timings=[],
        relative_gauge=None,
        tof="ideal registered 8x8 zone medians, 4m cap; absent in other variants",
    )
    first = None
    for n, frame in enumerate(trajectory["frames"]):
        sample = np.load(source / f"{frame:06d}.npz")
        rgb = sample["rgb"]
        torch.cuda.synchronize()
        start = time.perf_counter()
        raw = predict(
            model,
            rgb,
            sample["tof"] if args.variant.startswith("prompt") else None,
            args.variant,
            args.size,
        ).astype("f4")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if args.variant.startswith(("da2", "video")):
            if first is None:
                lo, hi = np.percentile(raw, [5, 95])
                if hi - lo < 1e-6:
                    raise ValueError("Relative depth is degenerate at startup")
                # Declared arbitrary metric gauge: first-frame 5/95% inverse-depth map to 4/.4m.
                a = (1 / 0.4 - 1 / 4) / (hi - lo)
                b = 1 / 4 - a * lo
                first = (float(a), float(b))
                report["relative_gauge"] = dict(
                    a=first[0],
                    b=first[1],
                    rule="Fixed startup inverse depth: p5=4m, p95=.4m; no truth fit",
                )
            depth = 1 / np.maximum(raw * first[0] + first[1], 1 / 40)
        else:
            depth = raw
        np.save(output / f"{frame:06d}.npy", depth.astype("f4"))
        report["timings"].append(
            dict(
                frame=int(frame),
                seconds=elapsed,
                median_depth_m=float(np.median(depth)),
                tof_valid=int((sample["tof"] > 0).sum())
                if args.variant.startswith("prompt")
                else None,
            )
        )
        peak = torch.cuda.max_memory_reserved() / 2**30
        if peak >= 7.5:
            raise RuntimeError(f"GPU memory limit reached: {peak:.2f}GiB Torch reservation")
        if n % 25 == 0:
            print(
                args.variant,
                args.sequence,
                n,
                "/",
                len(trajectory["frames"]),
                f"{elapsed * 1000:.1f}ms {peak:.2f}GiB",
                flush=True,
            )
    report["peak_torch_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    report["seconds_percentiles"] = dict(
        zip(
            ("p50", "p95", "p99"),
            np.percentile([r["seconds"] for r in report["timings"]], [50, 95, 99]).tolist(),
            strict=True,
        )
    )
    (output / "inference.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("../p47-out").resolve())
    p.add_argument(
        "--variant",
        choices=[
            "da2-small",
            "da2-large",
            "metric-small",
            "metric-large",
            "video-small",
            "prompt-small",
        ],
        required=True,
    )
    p.add_argument("--sequence", choices=["sweep180", "tour"], required=True)
    p.add_argument("--size", type=int, default=392)
    run(p.parse_args())
