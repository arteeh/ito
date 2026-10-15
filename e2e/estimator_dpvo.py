"""Causal DPVO challenger; each current pose is frozen once, without terminate()/future BA.

bash e2e/oracle.sh e2e/estimator_dpvo.py RECORDING --weights dpvo.pth --out OUT
DPVO translation remains monocular. The rotation-only arm explicitly uses robot translation.
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--hz", type=float, default=30)
    parser.add_argument("--precision", choices=("full", "half"), default="full")
    args = parser.parse_args()
    if args.hz <= 0 or args.seconds is not None and args.seconds <= 0:
        parser.error("--hz and --seconds must be positive")
    runner_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    import cv2
    import dpvo.dpvo as implementation
    import torch
    from dpvo.config import cfg
    from dpvo.lietorch import SE3

    from ito.protocol import RobotDescription
    from ito.reconstruction.fusion import PoseGraph
    from ito.reconstruction.motion import MotionBuffer
    from ito.reconstruction.ring import MOTION_STATES

    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    config = cfg.clone()
    config.PATCHES_PER_FRAME = 96
    config.REMOVAL_WINDOW = 22
    config.OPTIMIZATION_WINDOW = 10
    config.PATCH_LIFETIME = 13
    config.KEYFRAME_THRESH = 15.0
    config.MIXED_PRECISION = args.precision == "half"
    config.LOOP_CLOSURE = config.CLASSIC_LOOP_CLOSURE = False
    # DPVO's update() otherwise enables AMP even when MIXED_PRECISION is false.
    if args.precision == "full":
        implementation.autocast = lambda **_: torch.autocast("cuda", enabled=False)
    source = args.recording / "source"
    camera = json.loads((source / "camera.json").read_text())
    frames = [json.loads(s) for s in (source / "frames.jsonl").read_text().splitlines()]
    motions = [
        json.loads(s) for s in (args.recording / "received/motion.jsonl").read_text().splitlines()
    ]
    robot = RobotDescription.model_validate_json(
        (args.recording / "received/robot.json").read_text()
    )
    motion = MotionBuffer(robot.motion_noise, seconds=180)
    graph = PoseGraph(motion)
    native_keyframes = {}
    next_keyframe = 0.0
    samples = np.array(
        [
            [r["local"], *r["camera"], *r["angular_velocity"], MOTION_STATES.index(r["state"])]
            for r in motions
        ]
    )
    arrivals = np.array([r["arrival"] for r in motions])
    stamps = np.array([r["local"] for r in motions])
    width, height = 288, 512
    intrinsics = torch.tensor(
        [
            camera["fx"] * width / camera["width"],
            camera["fy"] * height / camera["height"],
            camera["cx"] * width / camera["width"],
            camera["cy"] * height / camera["height"],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    tracker = implementation.DPVO(config, str(args.weights), ht=height, wd=width, viz=False)
    args.out.mkdir(parents=True, exist_ok=True)
    axes = np.diag([1.0, -1.0, -1.0, 1.0])
    origin = frames[0]["captured"]
    available = next_capture = 0.0
    cursor = 0
    alignment = None
    manifest = []
    errors = {
        "native_rotation_deg": [],
        "motion_rotation_deg": [],
        "motion_translation_m": [],
        "fused_rotation_deg": [],
    }
    evaluation_alignment = None
    with torch.inference_mode():
        for row in frames:
            captured = row["captured"]
            when = captured - origin
            if args.seconds is not None and when > args.seconds:
                break
            if when < next_capture or captured < stamps[0]:
                continue
            after = int(np.searchsorted(stamps, captured))
            if after >= len(stamps):
                break
            started = max(row["available"] - origin, arrivals[after] - origin, when)
            if started < available:
                continue
            next_capture = when + 0.99 / args.hz
            end = int(np.searchsorted(arrivals, origin + started, side="right"))
            motion.extend(samples[cursor:end])
            cursor = end
            prior = motion.at(captured)
            with np.load(source / row["file"]) as data:
                rgb = data["rgb"]
                truth = data["camera"]  # Evaluation only; never passed into the tracker.
            torch.cuda.synchronize()
            began = time.perf_counter()
            image = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            image = torch.from_numpy(image[..., ::-1].copy()).permute(2, 0, 1).cuda()
            ordinal = tracker.counter
            tracker(when, image, intrinsics)
            torch.cuda.synchronize()
            observed = tracker.n > 0 and tracker.pg.tstamps_[tracker.n - 1] == ordinal
            native = None
            if observed:
                native = (
                    axes @ SE3(tracker.pg.poses_[tracker.n - 1]).inv().matrix().cpu().numpy() @ axes
                )
                if alignment is None:
                    alignment = prior @ np.linalg.inv(native)
                native = alignment @ native
            pose = prior.copy()
            usable = native is not None and tracker.is_initialized and np.isfinite(native).all()
            if usable:
                pose[:3, :3] = native[:3, :3]
            # Rotation-only correction through the existing uncertainty gate. Monocular
            # translation supplies no metric factor; the generic motion path stays explicit.
            node = graph.add_frame(captured)
            reference = graph.latest_keyframe()
            if usable and reference in native_keyframes:
                relative = np.linalg.inv(native_keyframes[reference]) @ native
                sigma = np.repeat((np.radians(0.5), 1e6), 3)
                graph.add_visual(reference, node, relative, sigma)
            graph.solve_frame()
            if usable and when >= next_keyframe:
                dropped = graph.promote()
                graph.solve_window()
                native_keyframes[node] = native.copy()
                for old in dropped:
                    native_keyframes.pop(old, None)
                next_keyframe = when + 0.5
            fused = prior.copy()
            fused[:3, :3] = graph.pose(node)[:3, :3]
            elapsed = time.perf_counter() - began
            available = started + elapsed
            if evaluation_alignment is None:
                evaluation_alignment = prior @ np.linalg.inv(truth)
            aligned_truth = evaluation_alignment @ truth
            for key, estimate in (
                ("native_rotation_deg", pose),
                ("motion_rotation_deg", prior),
                ("fused_rotation_deg", fused),
            ):
                errors[key].append(
                    float(
                        np.degrees(
                            Rotation.from_matrix(
                                aligned_truth[:3, :3].T @ estimate[:3, :3]
                            ).magnitude()
                        )
                    )
                )
            errors["motion_translation_m"].append(
                float(np.linalg.norm(prior[:3, 3] - aligned_truth[:3, 3]))
            )
            np.savez(
                args.out / row["file"],
                pose=pose,
                fused_pose=fused,
                native_pose=native if native is not None else np.full((4, 4), np.nan),
                motion_pose=prior,
                true_pose=truth,
            )
            manifest.append(
                dict(
                    file=row["file"],
                    source_frame_id=row["source_frame_id"],
                    captured=when,
                    started=started,
                    available=available,
                    estimate_seconds=elapsed,
                    initialized=bool(tracker.is_initialized),
                    observed=bool(observed),
                    used_native_rotation=bool(usable),
                    motion_through=end - 1,
                    rejected_visual=graph.rejected_visual,
                )
            )
            if len(manifest) % 50 == 0:
                print(json.dumps(manifest[-1]), flush=True)
    (args.out / "frames.json").write_text(json.dumps(manifest, indent=2))
    result = dict(
        config=config.dump(),
        precision=args.precision,
        shape=[height, width],
        translation="generic motion prior, NOT metric DPVO",
        frames=len(manifest),
        tracker_hz=len(manifest) / (manifest[-1]["available"] - manifest[0]["started"]),
        native_rotations=sum(r["used_native_rotation"] for r in manifest),
        latency_ms=np.percentile(
            [r["estimate_seconds"] * 1000 for r in manifest], [50, 95, 100]
        ).tolist(),
        errors={
            key: dict(
                zip(
                    ("median", "p95", "max"),
                    np.percentile(value, [50, 95, 100]).tolist(),
                    strict=True,
                )
            )
            for key, value in errors.items()
        },
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        runner_sha256=runner_digest,
        weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        cuda_peak_bytes=torch.cuda.max_memory_allocated(),
    )
    (args.out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
