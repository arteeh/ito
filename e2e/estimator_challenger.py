"""Bounded MapAnything check on four causal RGB views, with optional generic odometry.

bash e2e/oracle.sh e2e/estimator_challenger.py RECORDING --source MAP_ANYTHING_CLONE \
  --weights WEIGHTS --at 10 --out OUT [--motion]
This is a four-view latency/geometry diagnostic, not a live estimator or renderer result.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--at", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--motion", action="store_true")
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--window", type=float, default=1.5)
    parser.add_argument("--multiview-confidence", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.cropping import crop_resize_if_necessary
    from mapanything.utils.image import preprocess_inputs
    from PIL import Image
    from scipy.spatial.transform import Rotation

    from ito.protocol import RobotDescription
    from ito.reconstruction.motion import MotionBuffer
    from ito.reconstruction.ring import MOTION_STATES

    def rows(path):
        return [json.loads(s) for s in path.read_text().splitlines()]

    sources = rows(args.recording / "source/frames.jsonl")
    motions = rows(args.recording / "received/motion.jsonl")
    robot = RobotDescription.model_validate_json(
        (args.recording / "received/robot.json").read_text()
    )
    k = json.loads((args.recording / "source/camera.json").read_text())
    K = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1.0]], np.float32)
    origin = sources[0]["captured"]
    times = np.array([r["captured"] - origin for r in sources])
    selected = [
        sources[max(0, int(np.searchsorted(times, t, side="right") - 1))]
        for t in np.linspace(max(0, args.at - args.window), args.at, args.views)
    ]
    available = max(r["available"] for r in selected)
    motion = MotionBuffer(robot.motion_noise, seconds=120)
    motion.extend(
        np.array(
            [
                [r["local"], *r["camera"], *r["angular_velocity"], MOTION_STATES.index(r["state"])]
                for r in motions
                if r["arrival"] <= available
            ]
        )
    )
    axes = np.diag([1.0, -1.0, -1.0, 1.0])
    views = []
    for row in selected:
        data = np.load(args.recording / "source" / row["file"])
        view = dict(img=Image.fromarray(data["rgb"]), intrinsics=K)
        if args.motion:
            view.update(
                camera_poses=(axes @ motion.at(row["captured"]) @ axes).astype(np.float32),
                is_metric_scale=True,
            )
        views.append(view)
    processed = preprocess_inputs(views, resize_mode="fixed_size", size=(280, 504))
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    model = MapAnything.from_pretrained(str(args.weights)).eval().to("cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    began = time.perf_counter()
    with torch.inference_mode():
        predictions = model.infer(
            processed,
            use_amp=False,
            apply_mask=False,
            mask_edges=False,
            apply_confidence_mask=False,
            use_multiview_confidence=args.multiview_confidence,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - began
    metrics, manifest = [], []
    alignment = None
    for i, (row, prediction) in enumerate(zip(selected, predictions, strict=True)):
        data = np.load(args.recording / "source" / row["file"])
        # Truth is used only after inference, resized through exactly the same crop.
        _, truth_depth, _ = crop_resize_if_necessary(
            Image.fromarray(data["rgb"]),
            resolution=(280, 504),
            depthmap=data["depth"],
            intrinsics=K,
        )
        z = prediction["depth_z"].squeeze().cpu().numpy()
        pose = axes @ prediction["camera_poses"].squeeze().cpu().numpy() @ axes
        truth = data["camera"]
        if alignment is None:
            alignment = pose @ np.linalg.inv(truth)
        truth = alignment @ truth
        valid = (z > 0) & (truth_depth > 0) & np.isfinite(z)
        metrics.append(
            dict(
                source_frame_id=row["source_frame_id"],
                captured=row["captured"] - origin,
                depth_ratio=float(np.median(z[valid] / truth_depth[valid])),
                depth_absolute_m=float(np.median(np.abs(z[valid] - truth_depth[valid]))),
                rotation_deg=float(
                    np.degrees(Rotation.from_matrix(truth[:3, :3].T @ pose[:3, :3]).magnitude())
                ),
                translation_m=float(np.linalg.norm(pose[:3, 3] - truth[:3, 3])),
                valid_fraction=float(valid.mean()),
            )
        )
        rgb = (
            (prediction["img_no_norm"].squeeze(0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        )
        confidence = prediction["conf"].squeeze().cpu().numpy()
        weights = (
            {"depth_confidence": np.nan_to_num(confidence).clip(0, 1).astype(np.float32)}
            if args.multiview_confidence
            else {}
        )
        np.savez(
            args.out / f"{i}.npz",
            depth=z,
            pose=pose,
            rgb=rgb,
            true_depth=truth_depth,
            true_pose=data["camera"],
            model_confidence=confidence.astype(np.float32),
            **weights,
        )
        manifest.append(dict(file=f"{i}.npz", captured=row["captured"] - origin))
        if i == 0:
            calibration = processed[i]["intrinsics"].squeeze().cpu().numpy()
            (args.out / "camera.json").write_text(
                json.dumps(
                    dict(
                        width=z.shape[1],
                        height=z.shape[0],
                        fx=float(calibration[0, 0]),
                        fy=float(calibration[1, 1]),
                        cx=float(calibration[0, 2]),
                        cy=float(calibration[1, 2]),
                    ),
                    indent=2,
                )
            )
    result = dict(
        model="facebook/map-anything-apache",
        source_revision=subprocess.check_output(
            ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
        ).strip(),
        weights=str(args.weights),
        motion=args.motion,
        multiview_confidence=args.multiview_confidence,
        precision="full fp32 (no AMP)",
        resolution=[280, 504],
        window=(
            f"{args.views} views over {args.window:g}s through selected final exposure; "
            "no future views or truth inputs"
        ),
        seconds=elapsed,
        peak_cuda_reserved=torch.cuda.max_memory_reserved(),
        metrics=metrics,
    )
    (args.out / "report.json").write_text(json.dumps(result, indent=2))
    (args.out / "frames.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
