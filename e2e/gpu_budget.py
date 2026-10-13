"""Shared-load stereo trace, using frozen step-1 maps and recorded camera motion.

Run via e2e/gpu_budget.sh; output stays outside the checkout.
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from measure_gl import View


def load_worker(args, ready, start, stop):
    from PIL import Image
    from slam_replay import motion_rows

    from ito.protocol import Intrinsics, RobotDescription
    from ito.reconstruction import mast3r_runtime
    from ito.reconstruction.slam import SLAMBackend

    mast3r_runtime.MODELS = args.work / "pp/models"
    recording = args.work / f"rf-run/out/{args.sequence}/rec"
    intrinsics = Intrinsics.model_validate_json((recording / "camera.json").read_text())
    robot = RobotDescription.model_validate_json((recording / "robot.json").read_text())
    rows = [json.loads(s) for s in (recording / "frames.jsonl").read_text().splitlines()]
    motion = [json.loads(s) for s in (recording / "motion.jsonl").read_text().splitlines()]
    samples = motion_rows(motion)
    arrivals = np.array([m["local"] for m in motion])
    backend = SLAMBackend(
        524288,
        intrinsics,
        report=lambda s: None,
        motion_noise=robot.motion_noise,
        frame_delay=robot.cameras[0].capture_delay,
        exposure=robot.cameras[0].exposure,
        voxel_size=0.04,
        fade_seconds=0.5,
    )
    import torch

    costs = []
    cursor = 0
    origin = rows[0]["captured"]
    for index in range(0, len(rows), 6):
        if stop.is_set():
            break
        row = rows[index]
        rgb = np.asarray(Image.open(recording / row["name"]).convert("RGB"))
        end = int(np.searchsorted(arrivals, row["captured"], side="right"))
        backend.add_motion(samples[cursor:end])
        cursor = end
        began = time.perf_counter()
        backend.integrate(
            rgb,
            None,
            np.array(row["camera"], np.float32).reshape(4, 4),
            row["captured"] - origin,
            captured=row["captured"],
        )
        torch.cuda.synchronize()
        costs.append((time.perf_counter() - began) * 1000)
        if torch.cuda.memory_reserved() >= 7 * 1024**3:
            raise RuntimeError("Tracking reservation exceeds 7 GiB; stopping before 8 GiB")
        if index == 0:
            ready.set()
            if not start.wait(90):
                raise RuntimeError("Display did not start")
    (args.out / "load.json").write_text(
        json.dumps(
            dict(
                kind="production SLAMBackend, every sixth recorded frame, no duty sleeps",
                integrated=len(costs),
                integration_ms=costs,
                torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            ),
            indent=2,
        )
    )
    ready.set()


def main():
    from truth import align

    from ito.protocol import Intrinsics

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sequence", choices=["sweep180", "tour"], default="tour")
    parser.add_argument("--load", choices=["none", "slam"], default="none")
    parser.add_argument("--frames", type=int, default=1800)
    parser.add_argument("--capacity", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    index = 692 if args.sequence == "sweep180" else 2283
    snap = np.load(args.work / f"m1-out/verified/repeat-1/{args.sequence}/snapshot-{index:06d}.npz")
    intrinsics = Intrinsics.model_validate_json(
        (args.work / f"rf-run/out/{args.sequence}/rec/camera.json").read_text()
    )
    truths = np.load(args.work / f"gs/out/{args.sequence}/data.npz")["poses"]
    world = align(snap["first_pose"], truths[int(snap["first_index"])])
    camera = np.asarray(world @ truths[index], np.float32)
    records = snap["records"]
    occupied = len(records)
    if args.capacity:
        if args.capacity < occupied:
            records = records[np.linspace(0, occupied - 1, args.capacity, dtype=int)]
            occupied = len(records)
        else:
            records = np.pad(records, ((0, args.capacity - occupied), (0, 0), (0, 0)))
    context = mp.get_context("spawn")
    ready, start, stop = (context.Event() for _ in range(3))
    worker = None
    if args.load == "slam":
        worker = context.Process(target=load_worker, args=(args, ready, start, stop))
        worker.start()
        until = time.perf_counter() + 180
        while not ready.wait(0.1):
            if not worker.is_alive() or time.perf_counter() > until:
                stop.set()
                worker.join(2)
                if worker.is_alive():
                    worker.kill()
                raise RuntimeError("Tracking failed to initialize; see load log")
    view = None
    try:
        view = View(intrinsics)
        view.upload(records)
        durations, lateness = [], []
        with view.context:
            right = view.context.simple_framebuffer(view.size, components=4, dtype="f4")
            start.set()
            epoch = time.perf_counter()
            # Replay recorded orientations at 90 Hz, camera position held at the frozen map.
            for n in range(args.frames + 90):
                due = epoch + n / 90
                remaining = due - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                began = time.perf_counter()
                head = np.eye(4, dtype=np.float32)
                trajectory_index = int(n / 3) % min(len(truths), 600)
                head[:3, :3] = truths[0, :3, :3].T @ truths[trajectory_index, :3, :3]
                view.renderer.sort(camera, head)
                for side, target in ((-1, view.target), (1, right)):
                    eye = head.copy()
                    eye[:3, 3] += head[:3, 0] * side * 0.032
                    view.renderer.draw(
                        camera, eye, view.projection, target, clear=(0, 0, 0, 0), sort=False
                    )
                view.context.finish()
                done = time.perf_counter()
                if n >= 90:
                    durations.append((done - began) * 1000)
                    lateness.append((done - due) * 1000)
            right.release()
        streak = longest = 0
        for duration in lateness:
            streak = streak + 1 if duration > 1000 / 90 else 0
            longest = max(streak, longest)
        report = dict(
            scope="90 Hz offscreen stereo GL completion; no compositor or scanout",
            sequence=args.sequence,
            load=args.load,
            renderer=view.device,
            occupied=occupied,
            allocated=len(records),
            sort_capacity=view.renderer.capacity,
            upload_ms=view.upload_ms,
            frames=args.frames,
            stereo_ms={f"p{p:g}": float(np.percentile(durations, p)) for p in (50, 95, 99, 99.9)},
            deadline_ms={f"p{p:g}": float(np.percentile(lateness, p)) for p in (50, 95, 99, 99.9)},
            longest_stall_ms=max(durations),
            longest_deadline_ms=max(lateness),
            missed_fraction=float(np.mean(np.array(lateness) > 1000 / 90)),
            longest_miss_run=longest,
        )
        np.savez(args.out / "frames.npz", duration_ms=durations, completion_ms=lateness)
        (args.out / "display.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        stop.set()
        if worker:
            worker.join(10)
            if worker.is_alive():
                worker.kill()
                worker.join()
            if worker.exitcode:
                raise RuntimeError(f"Tracking exited {worker.exitcode}")
        if view:
            view.close()


if __name__ == "__main__":
    main()
