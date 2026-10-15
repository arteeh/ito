"""Freeze causal main estimates for the five oracle arms, using the Windows release.

RELEASE/runtime/python.exe -I e2e/oracle_estimate.py RECORDING --out OUT
No map alignment, retrospective interpolation or truth-assisted depth calibration.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from e2e.observation_recording import motion_scale, robot_description


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--precision", choices=("full", "half"), default="full")
    parser.add_argument("--input-resolution", type=int, default=100)
    parser.add_argument("--stamp-delay-ms", type=float, default=0)
    parser.add_argument("--frame-delay-ms", type=float, default=0)
    parser.add_argument("--seconds", type=float, help="Bound a diagnostic by source exposure time")
    parser.add_argument(
        "--observations",
        choices=("prior", "camera", "imu", "imu-tof", "prior-imu-tof"),
        default="prior",
    )
    args = parser.parse_args()
    import cv2
    import torch

    from ito.protocol import Intrinsics, SensorObservations
    from ito.reconstruction import mast3r_runtime
    from ito.reconstruction.slam import SLAMBackend

    mast3r_runtime.MODELS = args.models

    estimator_source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__).resolve().parents[1] / "ito/reconstruction").glob("*.py")
        if path.name
        in {
            "slam.py",
            "sensors.py",
            "motion.py",
            "ring.py",
            "fusion.py",
            "mast3r_frontend.py",
            "metric_scale.py",
            "depth_support.py",
            "__init__.py",
            "tracking.py",
        }
    }
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.recording / "source"
    rows = [json.loads(s) for s in (source / "frames.jsonl").read_text().splitlines()]
    motions = [
        json.loads(s) for s in (args.recording / "received/motion.jsonl").read_text().splitlines()
    ]
    robot = robot_description((args.recording / "received/robot.json").read_text())
    k = Intrinsics.model_validate_json((source / "camera.json").read_text())
    origin = rows[0]["captured"]
    motion_arrival = np.array([r["arrival"] for r in motions])
    motion_samples = np.array(
        [[r["local"], *r["camera"], *r["angular_velocity"], motion_scale(r)] for r in motions]
    )

    class Extract(SLAMBackend):
        def place(self, cloud, pose, when):
            self.cloud = cloud

    backend = Extract(
        1_000_000,
        k,
        report=print,
        motion_noise=robot.motion_noise if args.observations.startswith("prior") else None,
        # Exact sim exposure times, on the same WSL clock as robot Motion.
        frame_delay=args.frame_delay_ms / 1000,
        exposure=0,
        precision=args.precision,
        input_resolution=args.input_resolution,
    )
    sensor_path = args.recording / "received/sensors.jsonl"
    sensor_rows = (
        [json.loads(line) for line in sensor_path.read_text().splitlines()]
        if sensor_path.exists()
        else []
    )
    sensor_cursor = 0
    backend.cloud = None
    vision = None
    track = backend.frontend.track

    def record_vision(*args, **kwargs):
        nonlocal vision
        result = track(*args, **kwargs)
        vision = dict(
            node=args[1],
            reference=result.reference,
            relative=None if result.relative is None else result.relative.tolist(),
            sigma=None if result.relative is None else result.sigma.tolist(),
            keyframe=bool(result.keyframe),
            match=float(result.match),
        )
        return result

    backend.frontend.track = record_vision
    publications = []

    def published(pose, correction, stamp):
        publications.append(dict(captured=stamp - origin, seconds=time.perf_counter() - began))

    backend.pose_publisher = published
    cursor = 0
    available = 0
    next_capture = 0
    manifest = []
    for row in rows:
        captured = row["captured"]
        if args.seconds is not None and captured - origin > args.seconds:
            break
        stamped = captured + args.stamp_delay_ms / 1000
        # 10 Hz fixed RGB admission. Motion arrives on its recorded independent clock.
        prior = args.observations.startswith("prior")
        if captured < next_capture or captured < motions[0]["local"]:
            continue
        # Wait causally for a motion sample covering the exact exposure.
        after = np.searchsorted([m["local"] for m in motions], captured)
        if prior and after >= len(motions):
            break
        arrival = max(
            row["available"] - origin,
            motions[after]["arrival"] - origin if prior else 0,
            stamped - origin,
        )
        if arrival < available:
            continue
        next_capture = captured + 0.099
        started = arrival
        end = np.searchsorted(motion_arrival, origin + started, side="right")
        if prior:
            backend.add_motion(motion_samples[cursor:end])
        while (
            sensor_cursor < len(sensor_rows)
            and sensor_rows[sensor_cursor]["arrival"] <= origin + started
        ):
            observation = SensorObservations.model_validate_json(
                json.dumps(sensor_rows[sensor_cursor]["observation"])
            )
            sensor_cursor += 1
            imu = observation.imu if "imu" in args.observations else ()
            ranges = observation.ranges if "tof" in args.observations else ()
            if imu or ranges:
                backend.add_sensors(observation.model_copy(update={"imu": imu, "ranges": ranges}))
        cursor = end
        frame = np.load(source / row["file"])
        backend.cloud = None
        vision = None
        publications = []
        torch.cuda.synchronize()
        began = time.perf_counter()
        backend.integrate(frame["rgb"], None, np.eye(4), captured - origin, captured=stamped)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        available = started + elapsed
        cloud = backend.cloud
        if cloud is None:
            continue
        h, w = cloud.shape
        z = -cloud.points[:, 2].reshape(h, w).cpu().numpy()
        valid = cloud.valid.reshape(h, w).cpu().numpy()
        z[~valid] = 0
        # Crop exactly the sensor footprint; padding from the patch canvas is not depth.
        image = backend.frontend.image
        sw, sh = image.scaled
        z = z[image.top : image.top + sh, image.left : image.left + sw]
        depth = cv2.resize(z, (k.width, k.height), interpolation=cv2.INTER_NEAREST)
        extra = {}
        if getattr(cloud, "confidence", None) is not None:
            confidence = cloud.confidence.reshape(h, w).cpu().numpy()
            confidence = confidence[image.top : image.top + sh, image.left : image.left + sw]
            extra["depth_confidence"] = cv2.resize(
                confidence, (k.width, k.height), interpolation=cv2.INTER_NEAREST
            ).astype(np.float32)
        name = row["file"]
        np.savez(
            args.out / name,
            depth=depth.astype(np.float32),
            pose=backend.camera_pose,
            rgb=frame["rgb"],
            true_depth=frame["depth"],
            true_pose=frame["camera"],
            **extra,
        )
        manifest.append(
            dict(
                source_frame_id=row["source_frame_id"],
                file=name,
                captured=captured - origin,
                arrival=arrival,
                started=started,
                available=available,
                estimate_seconds=elapsed,
                motion_through=int(end) - 1,
                rejected_visual=backend.graph.rejected_visual,
                delay=backend.graph.td,
                delay_jitter=backend.graph.delay_jitter,
                scale=math.exp(backend.depth_estimator.log_scale),
                scale_observations=backend.depth_estimator.accepted,
                range_updates=backend.sensors.range_updates,
                imu_samples=len(backend.sensors.imu),
                scale_known=backend.depth_estimator.known,
                reused_depth_pixels=getattr(backend.frontend, "fused_pixels", 0),
                refined_depth_pixels=getattr(backend.frontend, "fused_pixels", 0),
                frontend_scale=getattr(backend.frontend, "output_scale", 1),
                unconstrained_bundle_scale_step=getattr(backend.frontend, "bundle_scale_step", 1),
                graph_pose=backend.camera_pose.tolist(),
                visual_factor=vision,
                motion_pose=backend.motion.at(stamped - backend.graph.td).tolist(),
                pose_publications=publications,
            )
        )
        publications = []
        if len(manifest) % 30 == 0:
            print(json.dumps(manifest[-1]), flush=True)
    (args.out / "frames.json").write_text(json.dumps(manifest, indent=2))
    (args.out / "settings.json").write_text(
        json.dumps(
            dict(
                observation_arm=args.observations,
                precision=args.precision,
                input_resolution=args.input_resolution,
                max_splats=1_000_000,
                frame_admission_hz=10,
                estimator="independent motion/vision estimator, no map alignment",
                exposure=0,
                frame_delay=args.frame_delay_ms / 1000,
                stamp_delay=args.stamp_delay_ms / 1000,
                source_seconds=args.seconds,
                common_start=motions[0]["local"]-origin,
                depth_confidence=getattr(
                    backend.frontend, "depth_confidence_kind", "network weight"
                ),
                estimator_source_sha256=estimator_source_hashes,
                source_sha256=hashlib.sha256((source / "frames.jsonl").read_bytes()).hexdigest(),
                origin=origin,
                estimated=len(manifest),
            ),
            indent=2,
        )
    )
    print(f"Estimated {len(manifest)} frames", flush=True)


if __name__ == "__main__":
    main()
