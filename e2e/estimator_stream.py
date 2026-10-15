"""Drive the real reconstruction process while motion publishes on its own clock.

RELEASE/runtime/python.exe -I e2e/estimator_stream.py RECORDING --out OUT
Run under ceres.lock. Full / Input 100% / Max 1M, no render readback or GPU cap.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e2e.observation_recording import motion_scale, robot_description
from ito import clock
from ito.protocol import Intrinsics
from ito.reconstruction import Reconstruction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--allow-stale", action="store_true")
    parser.add_argument("--stamp-delay-ms", type=float, default=0)
    parser.add_argument("--frame-delay-ms", type=float, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    def rows(path):
        return [json.loads(s) for s in path.read_text().splitlines()]

    frames = rows(args.recording / "source/frames.jsonl")
    motions = rows(args.recording / "received/motion.jsonl")
    robot = robot_description((args.recording / "received/robot.json").read_text())
    k = Intrinsics.model_validate_json((args.recording / "source/camera.json").read_text())
    origin = frames[0]["captured"]
    published, delays, regressions, work, motion_lag = [], [], [], [], []
    admitted = 0
    with Reconstruction(
        k,
        backend="slam",
        motion_noise=robot.motion_noise,
        exposure=0,
        frame_delay=args.frame_delay_ms / 1000,
    ) as worker:
        deadline = time.perf_counter() + 90
        while not (worker.status() or "").startswith("MASt3R ready"):
            worker.poll()
            if time.perf_counter() > deadline:
                raise RuntimeError("Estimator did not become ready")
            time.sleep(0.02)
        start = clock.now()
        shift = start - origin
        frame_index = motion_index = 0
        latest = 0
        while clock.now() - start < args.seconds:
            now = clock.now()
            due = []
            while motion_index < len(motions) and motions[motion_index]["arrival"] + shift <= now:
                sample = motions[motion_index]
                due.append(
                    (
                        sample["local"] + shift,
                        np.array(sample["camera"]).reshape(4, 4),
                        sample["angular_velocity"],
                        motion_scale(sample),
                    )
                )
                motion_index += 1
            if due:
                began = clock.now()
                worker.motion(due)
                result = worker.pose()
                if result is not None:
                    stamp = result[3]
                    if stamp < latest:
                        regressions.append([latest, stamp])
                    latest = max(latest, stamp)
                    published.append(stamp)
                    delays.append((clock.now() - began) * 1000)
                    motion_lag.append(max(0, due[-1][0] - stamp) * 1000)
            newest = None
            while (
                frame_index < len(frames)
                and max(
                    frames[frame_index]["available"],
                    frames[frame_index]["captured"] + args.stamp_delay_ms / 1000,
                )
                + shift
                <= now
            ):
                newest = frames[frame_index]
                frame_index += 1
            if newest:
                frame = np.load(args.recording / "source" / newest["file"])
                admitted += worker.submit(
                    frame["rgb"],
                    captured_at=newest["captured"] + shift + args.stamp_delay_ms / 1000,
                )
            packet = worker.poll()
            if packet is not None:
                worker.ring.acknowledge(packet.revision)
            work.append(worker.work_seconds.value * 1000)
            time.sleep(0.001)
        integrated = worker.integrated.value
    source = Path(sys.modules[Reconstruction.__module__].__file__).parent
    result = dict(
        settings=dict(precision="full", input_resolution=100, max_splats=1000000),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        estimator_source_sha256={
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in (
                "__init__.py",
                "slam.py",
                "fusion.py",
                "mast3r_frontend.py",
                "metric_scale.py",
            )
            if (source / name).is_file()
        },
        seconds=args.seconds,
        stamp_delay_ms=args.stamp_delay_ms,
        frame_delay_ms=args.frame_delay_ms,
        submitted=admitted,
        integrated=integrated,
        pose_publications=len(published),
        unique_pose_stamps=len(set(published)),
        pose_hz=len(set(published)) / args.seconds,
        pose_hz_active=(len(set(published)) - 1) / (max(published) - min(published)),
        motion_to_shared_pose_ms=dict(
            zip(("p50", "p95", "max"), np.percentile(delays, [50, 95, 100]).tolist(), strict=True)
        ),
        latest_motion_to_pose_stamp_ms=dict(
            zip(
                ("p50", "p95", "max"),
                np.percentile(motion_lag, [50, 95, 100]).tolist(),
                strict=True,
            )
        ),
        max_surface_work_ms=max(work),
        stamp_regressions=regressions,
    )
    (args.out / "stream.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not args.allow_stale and (regressions or len(set(published)) < 0.95 * len(published)):
        raise RuntimeError("Motion publication regressed or repeated a stale pose")


if __name__ == "__main__":
    main()
