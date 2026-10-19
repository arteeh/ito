"""Replay a driver recording (tools/research/data.py) through Ito's own reconstruction backend
(SLAMBackend, robot-first tracking), offline and frame-exact: every K-th frame, in order, with the
robot's motion and head sensors as they had arrived by then. Training runs on its own thread as live
(ITO_FIT_THREAD), and the replay is paced in real time (each frame no earlier than its stream time),
so the trainer gets the share of time it gets live. (--inline-fit N: N inline training steps per frame
instead, unpaced; that path grows dark pillars of splats by the table on close: do not use for maps.)
Poses are deterministic up to GPU nondeterminism; maps vary a little with the trainer's timing.

Run with a Windows build's Python (it has MASt3R, gsplat and CuPy), the worktree first on the path:

    cd builds\\BUILD\\ito && runtime\\python.exe -I REPLAY.py REC OUT [--every 10] [--fit 15]
        [--start S] [--end E] [--truth poses|poses,depth|depth] [--poses FILE.npz]
        [--precision full|half]

--poses: map by an outside estimator's poses (seq + Ito-world poses, export.py import), through the
same path exact poses take (--truth poses): Ito's mapper and depth, someone else's poses. Frames
between the given ones get poses interpolated in time; with --truth depth, the exact depth too.

Writes OUT/poses.npz (seq, poses: the published map-frame camera pose of every mapped frame),
OUT/map-*.npz (the live map every 30 s and at the end, as ITO_DUMP_MAP), OUT/frames.jsonl
(per frame: work ms, splats, VRAM). Environment switches (ITO_FUSE, ITO_SLAM, ...) apply as live."""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# --code ROOT (before anything imports ito): replay another checkout's reconstruction (a detached
# worktree of another branch), not this one.
CODE = sys.argv[sys.argv.index("--code") + 1] if "--code" in sys.argv else str(HERE.parents[1])
sys.path[:0] = [str(HERE), CODE]

import numpy as np  # noqa: E402
from data import Recording  # noqa: E402

LOOKAHEAD = 0.05  # s: motion and sensor samples that had arrived when the worker took a frame


def matrix(pose):
    from data import pose_matrix

    return pose_matrix(pose["position"], pose["orientation"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rec")
    p.add_argument("out")
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--fit", type=int, default=0, help="ignored (old inline steps)")
    p.add_argument("--inline-fit", type=int, default=0)
    p.add_argument("--speed", type=float, default=1.0, help="stream seconds per second; 0: unpaced")
    p.add_argument("--start", type=float)
    p.add_argument("--end", type=float)
    p.add_argument("--truth", default="")
    p.add_argument("--precision", default="full")
    p.add_argument("--poses")
    p.add_argument("--dump-every", type=float, default=30.0)
    p.add_argument("--code", help="checkout root whose ito package to replay")
    p.add_argument("--max-splats", type=int, default=1_000_000)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["ITO_FIT_THREAD"] = "0" if a.inline_fit else "1"

    from ito.protocol import Intrinsics, MotionNoise
    from ito.reconstruction.ring import MOTION_STATES
    from ito.reconstruction.slam import SLAMBackend

    rec = Recording(a.rec)
    meta = rec.meta
    i = meta["intrinsics"]
    intrinsics = Intrinsics(**i)
    noise = MotionNoise(**meta["motion_noise"])
    messages = []

    def report(text):
        messages.append(text)

    backend = SLAMBackend(
        a.max_splats,
        intrinsics,
        report=report,
        motion_noise=noise,
        frame_delay=meta["frame_delay"],
        exposure=meta["exposure"],
        precision=a.precision,
        input_resolution=100,
        voxel_size=0.04,
        fade_seconds=0.5,
        device="cuda",
    )
    times, poses, gyro, states = rec.motion
    motion_rows = np.zeros((len(times), 21))
    motion_rows[:, 0] = times
    motion_rows[:, 1:17] = poses.reshape(-1, 16)
    motion_rows[:, 17:20] = gyro
    motion_rows[:, 20] = [MOTION_STATES.index(s) for s in states]
    sensors = []
    for message in rec.sensors:
        imu = [
            dict(
                sensor=u["sensor"],
                mount=matrix(u["mount"]),
                gyro_sigma=u["gyro_sigma"],
                accel_sigma=u["accel_sigma"],
                samples=[
                    (s["time"], s["angular_velocity"], s["specific_force"], s["orientation"])
                    for s in u["samples"]
                ],
            )
            for u in message["imu"]
        ]
        ranges = [
            dict(
                sensor=g["sensor"],
                time=g["time"],
                mount=matrix(g["mount"]),
                shape=(g["rows"], g["cols"]),
                field_of_view=g["field_of_view"],
                distances=[math.nan if d is None else d for d in g["distances"]],
                sigma=g["sigma"],
            )
            for g in message["ranges"]
        ]
        last = max(
            [s[0] for u in imu for s in u["samples"]] + [g["time"] for g in ranges] or [0]
        )
        sensors.append((last, imu, ranges))
    sensors.sort(key=lambda item: item[0])
    frames = [f for f in rec.frames if f.get("file")]
    t0 = frames[0]["received"]
    frames = [
        f
        for f in frames
        if (a.start is None or f["received"] - t0 >= a.start)
        and (a.end is None or f["received"] - t0 <= a.end)
    ]
    truth_modes = set(filter(None, a.truth.split(",")))
    if "depth" in truth_modes:
        # Exact depth exists for every 3rd frame: the same mapping rate from those.
        frames = [f for f in frames if f.get("depth")][:: max(1, round(a.every / 3))]
    else:
        frames = frames[:: a.every]
    outside = None
    if a.poses:
        given = np.load(a.poses)
        by_seq = {f["seq"]: f["received"] for f in rec.frames}
        order = np.argsort([by_seq[int(s)] for s in given["seq"]])
        times_given = np.array([by_seq[int(given["seq"][i])] for i in order])
        poses_given = np.asarray(given["poses"])[order]
        from scipy.spatial.transform import Rotation, Slerp

        slerp = Slerp(times_given, Rotation.from_matrix(poses_given[:, :3, :3]))

        def outside(t):
            if t < times_given[0] or t > times_given[-1]:
                return None
            m = np.eye(4)
            m[:3, :3] = slerp([t]).as_matrix()[0]
            m[:3, 3] = [np.interp(t, times_given, poses_given[:, k, 3]) for k in range(3)]
            return m
    motion_cursor = sensor_cursor = 0
    published = []
    log = (out / "frames.jsonl").open("w")
    next_dump = a.dump_every
    import torch

    began_all = time.perf_counter()
    trainer = None
    if not a.inline_fit:
        stopped = [False]
        trainer = backend.start_training(lambda: True, lambda: stopped[0], lambda exc: None)
    stream_start = frames[0]["received"]
    for f in frames:
        if a.speed and not a.inline_fit:
            due = began_all + (f["received"] - stream_start) / a.speed
            wait = due - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        captured = f["received"]
        while motion_cursor < len(times) and times[motion_cursor] <= captured + LOOKAHEAD:
            motion_cursor += 1
        backend.add_motion(motion_rows[: motion_cursor][-512:])
        while sensor_cursor < len(sensors) and sensors[sensor_cursor][0] <= captured + LOOKAHEAD:
            backend.add_sensors(*sensors[sensor_cursor][1:])
            sensor_cursor += 1
        rgb = rec.rgb(f)
        truth = None
        if outside is not None:
            pose = outside(f["received"])
            if pose is None:
                continue
            depth = rec.depth(f) if "depth" in truth_modes else None
            if "depth" in truth_modes and depth is None:
                continue
            truth = (pose, f["exposed"] if "exposed" in f else captured, depth, None)
        elif truth_modes and f.get("pose") is not None:
            depth = rec.depth(f) if "depth" in truth_modes else None
            if "depth" in truth_modes and depth is None:
                continue
            truth = (
                f["pose"] if "poses" in truth_modes else None,
                f["exposed"],
                depth,
                None,
            )
        now = captured - t0
        tracked = backend.tracked
        backend.source_sequence = f["seq"] + 1
        backend.source_captured = captured
        began = time.perf_counter()
        # As live: everything the worker does to the map holds the map's lock; the trainer thread
        # takes it only between its steps.
        with backend.lock:
            backend.integrate(
                rgb, None, np.eye(4, dtype=np.float32), now, captured=captured, truth=truth
            )
            backend.expire(now, 1 << 60)
        work = time.perf_counter() - began
        fit_began = time.perf_counter()
        for _ in range(a.inline_fit):
            if not backend.fit():
                break
        fit = time.perf_counter() - fit_began
        if backend.tracked != tracked:
            published.append(
                (
                    f["seq"],
                    backend.camera_pose.astype(np.float64).copy(),
                    backend.correction.astype(np.float64).copy(),
                    now,
                )
            )
        log.write(
            json.dumps(
                dict(
                    seq=f["seq"],
                    t=round(now, 3),
                    work_ms=round(work * 1000, 1),
                    fit_ms=round(fit * 1000, 1),
                    splats=int(backend.count),
                    tracked=int(backend.tracked),
                    matched=int(backend.matched),
                    keyframes=len(backend.graph.keyframes),
                    fit_steps=int(backend.fit_steps),
                    vram_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2),
                )
            )
            + "\n"
        )
        if now >= next_dump:
            next_dump += a.dump_every
            with backend.lock:
                dump(backend, out, now)
    if trainer is not None:
        stopped[0] = True
        trainer.join(timeout=10)
    with backend.lock:
        dump(backend, out, frames[-1]["received"] - t0)
    np.savez(
        out / "poses.npz",
        seq=np.array([row[0] for row in published]),
        poses=np.array([row[1] for row in published]),
        corrections=np.array([row[2] for row in published]),
        t=np.array([row[3] for row in published]),
    )
    (out / "summary.json").write_text(
        json.dumps(
            dict(
                frames=len(frames),
                published=len(published),
                seconds=time.perf_counter() - began_all,
                splats=int(backend.count),
                vram_peak_gib=torch.cuda.max_memory_reserved() / 2**30,
                last_report=messages[-1] if messages else None,
            ),
            indent=1,
        )
    )
    print("done", out)


def dump(backend, out, now):
    live = np.flatnonzero((backend.keys >= 0) & ~backend.retiring)
    np.savez(
        out / f"map-{now:07.1f}.npz",
        records=backend.records[live],
        seconds=now,
        captures=backend.captures[live],
        sequences=backend.sequences[live],
        support=backend.support[live],
        owners=backend.owners[live],
    )


if __name__ == "__main__":
    main()
