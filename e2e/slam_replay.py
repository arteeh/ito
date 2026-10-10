"""Replay a recorded camera stream through MASt3R-SLAM at its live pace; measure tracking.

uv run --extra slam python e2e/slam_replay.py RECORDING [--out DIR] [--gaze-prior]

RECORDING comes from e2e/slam_drive.py --record. The backend sees what the live worker
would: the newest frame whenever it finishes the previous one, with the robot's pose
prior. Reports SLAM rate, lost frames, local map restarts by cause (unmatched, heading,
failure) in total and per 100 s, how long the pilot would see the flat feed, how far the
SLAM heading strays from the robot's measured gaze, and how well the scene seen so far
predicts the camera image one second later. --gaze-prior builds the prior from the
recorded telemetry gaze, for recordings of drivers that sent none.

To compare with another revision's backend, run this script with that checkout first on
the path; a backend that does not count restarts itself has them inferred per frame:

git worktree add ../ito-main main
ln -s "$PWD/models" ../ito-main/models
PYTHONPATH=../ito-main uv run --extra slam python e2e/slam_replay.py RECORDING
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ito.app.pilot import TRACKING_LOST
from ito.protocol import Intrinsics
from ito.reconstruction import slam as backend_module
from ito.reconstruction.slam import SLAMBackend, heading
from ito.render import pose
from ito.render.pose import quaternion

AHEAD = 1.0  # Seconds between a scene snapshot and the frame it has to predict.


def render(records, live, camera, intrinsics, size):
    """Nearest-first point splats of the fused scene, seen from a world-from-camera pose."""
    position = records[live, 0, :3]
    radius = records[live, 1, 0]
    color = np.clip(records[live, 3, :3] * 0.2820947918 + 0.5, 0, 1)
    local = (position - camera[:3, 3]) @ camera[:3, :3]
    depth = -local[:, 2]
    ahead = depth > 1e-6
    local, depth, radius, color = local[ahead], depth[ahead], radius[ahead], color[ahead]
    u = intrinsics.fx * local[:, 0] / depth + intrinsics.cx
    v = intrinsics.cy - intrinsics.fy * local[:, 1] / depth
    pixels = np.clip(np.round(radius * intrinsics.fx / depth), 0, 3).astype(int)
    width, height = size
    zbuffer = np.full(height * width, np.inf)
    image = np.zeros((height * width, 3))
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            reach = max(abs(dx), abs(dy))
            x = np.round(u).astype(int) + dx
            y = np.round(v).astype(int) + dy
            inside = (pixels >= reach) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
            index = y[inside] * width + x[inside]
            np.minimum.at(zbuffer, index, depth[inside])
            nearest = depth[inside] <= zbuffer[index]
            image[index[nearest]] = color[inside][nearest]
    covered = np.isfinite(zbuffer)
    return image.reshape(height, width, 3), covered.reshape(height, width)


def rotation(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def prior_at(priors, times, when):
    """The recorded prior at `when`, interpolated between the frames around it."""
    after = int(np.clip(np.searchsorted(times, when), 1, len(times) - 1))
    a, b = priors[after - 1], priors[after]
    f = float(np.clip((when - times[after - 1]) / max(times[after] - times[after - 1], 1e-9), 0, 1))
    qa, qb = np.array(quaternion(a)), np.array(quaternion(b))
    if qa @ qb < 0:
        qb = -qb
    q = qa * (1 - f) + qb * f
    result = a.copy()
    result[:3, :3] = rotation(q / np.linalg.norm(q))
    result[:3, 3] = a[:3, 3] * (1 - f) + b[:3, 3] * f
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--budget", type=int, default=1_048_576)
    parser.add_argument("--limit", type=float, default=0, help="stop after this many seconds")
    parser.add_argument("--gaze-prior", action="store_true")
    parser.add_argument("--unmeasured", action="store_true", help="hide the robot's heading")
    parser.add_argument("--measured", action="store_true", help="trust the recorded prior")
    parser.add_argument(
        "--prior-delay",
        type=float,
        default=0,
        help="seconds by which the recorded prior led its frame: use the prior recorded then",
    )
    parser.add_argument(
        "--set", action="append", default=[], help="override MASt3R-SLAM config: tracking.Q_conf=2"
    )
    args = parser.parse_args()
    out = args.out or Path("e2e/out/slam-replay") / args.recording.name
    out.mkdir(parents=True, exist_ok=True)
    intrinsics = Intrinsics.model_validate_json((args.recording / "camera.json").read_text())
    rows = [json.loads(line) for line in (args.recording / "frames.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row["gaze"] is not None]
    times = np.array([row["captured"] for row in rows])
    gazes = np.unwrap([row["gaze"] for row in rows])
    priors = [
        None if row["camera"] is None else np.array(row["camera"]).reshape(4, 4) for row in rows
    ]
    messages = []
    backend = SLAMBackend(
        args.budget,
        intrinsics,
        report=messages.append,
        voxel_size=0.04,
        fade_seconds=0.5,
    )
    import torch
    from mast3r_slam.config import config

    for override in args.set:
        name, value = override.split("=", 1)
        section, key = name.split(".")
        config[section][key] = type(config[section][key])(value)
    start = times[0]
    now = start
    index = -1
    processed = []  # (frame, time, seconds, tracked, keyframes, message, heading, gaze)
    pending = []  # (due time, records, live) snapshots waiting for their future frame
    predictions = []
    splats, drops = [0], []  # Splats shown after each frame; frames that cut them by a tenth
    previews = 0
    inferred = dict(unmatched=0, heading=0, failure=0)  # For backends that do not count.
    origin = None
    while True:
        newer = np.flatnonzero(times <= now)
        if not len(newer) or newer[-1] <= index:
            if index + 1 >= len(rows):
                break
            now = max(now, times[index + 1])
            continue
        index = int(newer[-1])
        if args.limit and now - start > args.limit:
            break
        row = dict(rows[index])
        if args.prior_delay:
            row["gaze"] = float(np.interp(times[index] - args.prior_delay, times, gazes))
        rgb = np.asarray(Image.open(args.recording / row["name"]).convert("RGB"))
        if args.gaze_prior:
            prior, measured = pose(yaw=row["gaze"]), True
        else:
            prior = (
                prior_at(priors, times, times[index] - args.prior_delay)
                if args.prior_delay
                else priors[index]
            ).astype(np.float32)
            measured = row.get("measured", False) or args.measured
        tracked_before = backend.tracked
        realigned_before = getattr(backend, "realigned", 0)
        failures_before = backend.failures
        retained = len(backend.frames)
        before = len(messages)
        began = time.perf_counter()
        backend.integrate(rgb, None, prior, now - start, measured=measured and not args.unmeasured)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - began
        news = messages[before:]
        tracked = backend.tracked > tracked_before
        if getattr(backend, "realigned", 0) > realigned_before:
            # Such a backend realigns on its third unmatched frame or on a heading stray.
            misses = getattr(backend_module, "MEASURED_MISSES", 2)
            inferred["unmatched" if failures_before >= misses else "heading"] += 1
        if any("restarted" in m for m in news):
            # Every retained view failed, or the local solve diverged.
            gave_up = failures_before >= max(retained, 1) + 2
            inferred["unmatched" if gave_up else "failure"] += 1
        slam = heading(backend.camera_pose[:3, :3])
        if origin is None:
            origin = (slam, row["gaze"])
        processed.append(
            (
                index,
                now - start,
                seconds,
                tracked,
                len(backend.frames),
                news[-1] if news else "",
                math.remainder(slam - origin[0], math.tau),
                math.remainder(row["gaze"] - origin[1], math.tau),
            )
        )
        if tracked:
            camera = backend.camera_pose.astype(np.float64)
            for due, records, live in [p for p in pending if p[0] <= now]:
                pending.remove((due, records, live))
                if now - due > 0.5:
                    continue
                image, covered = render(
                    records, live, camera, intrinsics, (rgb.shape[1], rgb.shape[0])
                )
                if covered.mean() > 0.05:
                    error = ((image[covered] - rgb[covered] / 255) ** 2).mean()
                    predictions.append((float(covered.mean()), float(-10 * np.log10(error))))
                    if previews < 12 and len(predictions) % 10 == 1:
                        side = np.concatenate((rgb / 255, image), axis=1)
                        Image.fromarray((side * 255).astype(np.uint8)).save(
                            out / f"predict-{previews:02d}.png"
                        )
                        previews += 1
            if not pending or now - pending[-1][0] + AHEAD > 0.5:
                live = backend.keys >= 0
                pending.append((now + AHEAD, backend.records.copy(), live.copy()))
        # Splats the scene shows: a drop of a tenth in one frame is a view carving the room.
        shown = int(np.count_nonzero((backend.keys >= 0) & ~backend.retiring))
        if shown < 0.9 * splats[-1]:
            drops.append((round(now - start, 2), splats[-1], shown))
        splats.append(shown)
        backend.expire(now - start, np.inf)
        # The live worker also publishes and sleeps between frames.
        now += seconds + 0.01
    durations = np.array([p[2] for p in processed])
    ok = np.array([p[3] for p in processed])
    stamps = np.array([p[1] for p in processed])
    # The pilot's view: tracked within TRACKING_LOST, else flat.
    last_ok, flat = None, 0.0
    for stamp, good, after in zip(stamps, ok, np.append(stamps[1:], stamps[-1]), strict=True):
        if good:
            last_ok = stamp
        if last_ok is None or after - last_ok >= TRACKING_LOST:
            flat += after - stamp
    restarts = dict(getattr(backend, "restarts", inferred))
    per_100s = 100 / float(stamps[-1])
    stray = np.degrees(np.abs([math.remainder(p[6] - p[7], math.tau) for p in processed if p[3]]))
    report = dict(
        recording=str(args.recording),
        measured=not args.unmeasured,
        prior_delay=args.prior_delay,
        overrides=args.set,
        seconds=round(float(stamps[-1]), 1),
        frames_available=len(rows),
        frames_processed=len(processed),
        slam_hz=round(len(processed) / float(stamps[-1]), 2),
        frame_ms_median=round(float(np.median(durations)) * 1000, 1),
        frame_ms_p95=round(float(np.percentile(durations, 95)) * 1000, 1),
        lost_frames=int((~ok).sum()),
        restarts=restarts,
        restarts_total=sum(restarts.values()),
        restarts_per_100s={k: round(v * per_100s, 1) for k, v in restarts.items()},
        restarts_inferred=not hasattr(backend, "restarts"),
        heading_corrections=getattr(backend, "corrections", None),
        recalls=getattr(backend, "recalls", None),
        provisional_frames=getattr(backend, "provisional", None),
        splats_peak=max(splats),
        splats_final=splats[-1],
        splat_drops_over_10pct=len(drops),
        splat_drops=drops[:20],
        flat_fraction=round(flat / float(stamps[-1]), 3),
        gaze_range_deg=round(math.degrees(np.ptp([p[7] for p in processed])), 1),
        heading_error_deg_median=round(float(np.median(stray)), 1),
        heading_error_deg_p95=round(float(np.percentile(stray, 95)), 1),
        heading_error_deg_max=round(float(stray.max()), 1),
        prediction_psnr_median=round(float(np.median([p[1] for p in predictions])), 2)
        if predictions
        else None,
        prediction_coverage_median=round(float(np.median([p[0] for p in predictions])), 3)
        if predictions
        else None,
        messages=sorted({m.split(" | ")[0] for m in messages}),
    )
    (out / "frames.json").write_text(json.dumps(processed))
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(
        f"restarts per 100 s: {sum(restarts.values()) * per_100s:.1f} ("
        + ", ".join(f"{k} {v}" for k, v in report["restarts_per_100s"].items())
        + ")"
        + (" inferred" if report["restarts_inferred"] else "")
        + f"; recalls {report['recalls']}, provisional frames {report['provisional_frames']}, "
        f"splat drops over 10% {report['splat_drops_over_10pct']}"
    )


if __name__ == "__main__":
    main()
