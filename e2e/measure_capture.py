"""Causal, fixed-input SLAM replay. Run under ceres.lock with the SLAM environment.

python e2e/measure_capture.py RECORDING --schedule PLAN.json --models MODELS --out OUT
The plan fixes frame admissions; measured completion gates publication. Snapshot I/O is
outside the replay clock. A slow candidate never publishes a result before it finishes.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from e2e.observation_recording import robot_description


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def product_digest():
    root = Path(__file__).resolve().parents[1]
    value = hashlib.sha256()
    for path in sorted(p for p in (root / "ito").rglob("*") if p.suffix in (".py", ".glsl")):
        value.update(path.relative_to(root).as_posix().encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=1_048_576)
    args = parser.parse_args()
    from ito.protocol import Intrinsics
    from ito.reconstruction import mast3r_runtime

    mast3r_runtime.MODELS = args.models
    from ito.reconstruction.slam import SLAMBackend

    rows = [json.loads(s) for s in (args.recording / "frames.jsonl").read_text().splitlines()]
    motion = [json.loads(s) for s in (args.recording / "motion.jsonl").read_text().splitlines()]
    from slam_replay import motion_rows

    samples = motion_rows(motion)
    # Old files recorded sample time but not arrival: disclose the assumption in the plan.
    arrivals = np.array([m.get("arrival", m["local"]) for m in motion])
    robot = robot_description((args.recording / "robot.json").read_text())
    intrinsics = Intrinsics.model_validate_json((args.recording / "camera.json").read_text())
    plan = json.loads(args.schedule.read_text())
    if digest(args.recording / "frames.jsonl") != plan["frames_sha256"]:
        raise ValueError("Schedule belongs to another recording")
    args.out.mkdir(parents=True, exist_ok=True)
    backend = SLAMBackend(
        args.budget,
        intrinsics,
        report=lambda s: None,
        motion_noise=robot.motion_noise,
        frame_delay=robot.cameras[0].capture_delay,
        exposure=robot.cameras[0].exposure,
        voxel_size=0.04,
        fade_seconds=0.5,
    )
    backend.grow(args.budget)  # Every slot up front: snapshots compare whole arrays.
    import torch

    start = rows[0]["captured"]
    snapshots = iter(sorted(plan["snapshots"], key=lambda s: s["deadline"]))
    target = next(snapshots, None)
    available = 0.0
    first = None
    admitted, publications, deadline_poses = [], [], []
    fed = 0
    refined_poses = {}
    refined_through = -1
    latest_pose = None
    latest_frame = -1
    max_arrival = None
    observed_frame = np.full(args.budget, -1, np.int64)
    source_times = np.array([r["captured"] - start for r in rows])

    def save(until):
        nonlocal target
        while target is not None and target["deadline"] < until:
            deadline = target["deadline"]
            if admitted and max(admitted) > target["frame"]:
                raise AssertionError("Future observation in prefix")
            if latest_pose is not None:
                live = backend.keys >= 0
                records = backend.records[live].copy()
                # Preserve the production retirement fade at this deadline.
                retiring = records[:, 3, 3] == 1
                records[retiring, 0, 3] *= np.clip(
                    (records[retiring, 1, 3] - deadline) / backend.fade, 0, 1
                )
                np.savez_compressed(
                    args.out / f"snapshot-{target['frame']:06d}.npz",
                    records=records,
                    pose=latest_pose,
                    first_pose=first[1],
                    first_index=first[0],
                    index=latest_frame,
                    target_index=target["frame"],
                    deadline=deadline,
                    published=available,
                    observation_indices=np.array(admitted),
                    max_observation_arrival=max_arrival,
                    coordinate_revision=0,
                    last_surface_update=backend.seen[live],
                    observation_frame_indices=observed_frame[live],
                    last_observed_capture=source_times[np.maximum(observed_frame[live], 0)],
                )
            target = next(snapshots, None)

    for event in plan["admissions"]:
        index, arrival = event["frame"], event["arrival"]
        began_at = max(arrival, available)
        # Deadlines before this job's completion can only see the old state. Save before
        # mutating the backend, using a copy for the subsequently completed publication.
        row = rows[index]
        rgb = np.asarray(Image.open(args.recording / row["name"]).convert("RGB"))
        old_records, old_keys, old_seen = (
            backend.records.copy(),
            backend.keys.copy(),
            backend.seen.copy(),
        )
        end = int(np.searchsorted(arrivals, start + began_at, side="right"))
        backend.add_motion(samples[fed:end])
        fed = end
        before_matched, before_tracked = backend.matched, backend.tracked
        before_refreshed = backend.refreshed
        began = time.perf_counter()
        backend.integrate(
            rgb,
            None,
            np.array(row["camera"], np.float32).reshape(4, 4),
            began_at,
            captured=row["captured"],
        )
        torch.cuda.synchronize()
        if torch.cuda.memory_reserved() >= 8 * 1024**3:
            raise RuntimeError("Replay exceeded the 8 GiB CUDA reservation limit")
        cost = time.perf_counter() - began
        done = began_at + cost
        new_records, new_keys, new_seen = backend.records, backend.keys, backend.seen
        backend.records, backend.keys, backend.seen = old_records, old_keys, old_seen
        save(done)
        backend.records, backend.keys, backend.seen = new_records, new_keys, new_seen
        observed_frame[(new_seen != old_seen) & (new_keys >= 0)] = index
        available = done
        admitted.append(index)
        max_arrival = arrival
        latest_frame = index
        for j in range(refined_through + 1, index + 1):
            if backend.motion.covers(rows[j]["captured"] - backend.graph.td):
                refined_poses[j] = backend.pose_at(rows[j]["captured"]).ravel().tolist()
        refined_through = index
        if backend.tracked > before_tracked:
            latest_pose = backend.camera_pose.copy()
            if first is None:
                first = (index, latest_pose.copy())
        publications.append(
            dict(
                frame=index,
                source_frame_id=row.get("source_frame_id"),
                exposure=row.get("exposure_time"),
                arrival=arrival,
                started=began_at,
                max_motion_arrival=float(arrivals[end - 1] - start) if end else -1.0,
                publication=done,
                seconds=cost,
                coordinate_revision=0,
                visual_factors=backend.matched - before_matched,
                propagated_poses=backend.tracked - before_tracked,
                surface_updates=int(backend.refreshed > before_refreshed),
                pose=None if latest_pose is None else latest_pose.ravel().tolist(),
            )
        )
        backend.expire(done, np.inf)
        if index % 90 == 0:
            print(f"frame {index + 1}, publication {done:.3f}s, {backend.count} splats", flush=True)
    save(float("inf"))
    published_times = [p["publication"] for p in publications]
    for index, source in enumerate(rows):
        deadline = source["captured"] - start
        k = int(np.searchsorted(published_times, deadline, side="right")) - 1
        refined = refined_poses.get(index)
        deadline_poses.append(
            dict(
                frame=index,
                deadline=deadline,
                available_pose=publications[k]["pose"] if k >= 0 else None,
                published=publications[k]["publication"] if k >= 0 else None,
                retrospective_pose=refined,
            )
        )
    for name, values in (("publications", publications), ("poses", deadline_poses)):
        (args.out / f"{name}.jsonl").write_text("".join(json.dumps(v) + "\n" for v in values))
    report = dict(
        schedule_sha256=digest(args.schedule),
        frames_sha256=plan["frames_sha256"],
        seconds=available,
        visual_factors_s=backend.matched / available,
        propagated_poses_s=backend.tracked / available,
        surface_updates_s=sum(p["surface_updates"] for p in publications) / available,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        capture_script_sha256=digest(Path(__file__)),
        product_sha256=product_digest(),
        budget=args.budget,
        baseline="316a556",
        timing="fixed admissions, measured serial completion; no concurrent display",
        retrospective="first subsequent integration, separate from deadline-available pose",
    )
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
