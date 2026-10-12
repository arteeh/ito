"""Replay a recorded camera stream through SLAM at its live pace; measure tracking.

uv run --extra slam python e2e/slam_replay.py RECORDING [--pose RUNG] [--truth TRUTH.jsonl]
uv run --with scipy==1.15.3 python e2e/slam_replay.py RECORDING --pose L3 --truth T --poses-only

RECORDING comes from e2e/slam_drive.py --record. The backend sees what the live worker
would: the newest frame whenever it finishes the previous one, with the robot's pose
prior and (robot-first) every Motion sample that had arrived by then. --pose picks the
camera pose of the tracking plan's oracle ladder (#36); MASt3R supplies depth on every rung:

  L1 truth     the simulator's true pose (needs --truth)
  L2 receive   the robot's prior as frames carry it: receive-stamped, 20 Hz
  L3 source    the robot's Motion: source-stamped, 50 Hz, at capture time minus the delay
  L4 fused     robot-first tracking (ito/reconstruction/slam.py)
  L5 vision    vision-only MASt3R-SLAM, today's main (the default)

Reports SLAM rate, lost frames, map restarts by cause (vision only; robot-first has none),
how long the pilot would see the flat feed, how far the heading strays from the robot's
measured gaze, how well the scene seen so far predicts the camera image one second later,
frame-to-pose latency and, for L4, how fast fusion's corrections move the eye. With
--truth (the virtual Microduck's microduck-sim --truth file of the same drive) it scores
every pose: rotation p50/p95/max, translation after a fixed first-frame alignment, and the
fraction of usable poses (within 2 degrees and 5 cm), and as "every_frame" every recorded
frame's pose, tracked or not, once the next tracked frame is in: the backend's pose_at where
it has one (the robot's motion between fused poses), else the published poses interpolated.
--poses-only skips MASt3R, for L2, L3 and L4 (fusion without vision) on a CPU.
--gaze-prior builds the prior from the recorded telemetry gaze, for recordings of drivers
that sent none.

To compare with another revision's backend, run this script with that checkout first on
the path; a backend that does not count restarts itself has them inferred per frame:

git worktree add ../ito-main main
ln -s "$PWD/models" ../ito-main/models
PYTHONPATH=../ito-main uv run --extra slam python e2e/slam_replay.py RECORDING
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ito.app.pilot import TRACKING_LOST
from ito.protocol import Intrinsics, RobotDescription
from ito.reconstruction.slam_vision import heading
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


RUNGS = {"L1": "truth", "L2": "receive", "L3": "source", "L4": "fused", "L5": "vision"}


def motion_rows(motion):
    """Recorded Motion samples as the worker's ring hands them over, on the pilot clock."""
    from ito.reconstruction.ring import MOTION_STATES

    return np.array(
        [
            [m["local"], *m["camera"], *m["angular_velocity"], MOTION_STATES.index(m["state"])]
            for m in motion
        ]
    )


def eye_corrections(corrections, stamps):
    """How fast fusion's world-from-odometry correction moves: degrees/s and m/s at the eye."""
    turns, slides = [], []
    for (c0, t0, eye), (c1, t1, _) in zip(corrections, corrections[1:], strict=False):
        dt = t1 - t0
        if dt <= 0:
            continue
        change = c1 @ np.linalg.inv(c0)
        cosine = np.clip((np.trace(change[:3, :3]) - 1) / 2, -1, 1)
        turns.append(math.degrees(math.acos(cosine)) / dt)
        slides.append(float(np.linalg.norm(change[:3, :3] @ eye + change[:3, 3] - eye)) / dt)
    if not turns:
        return None
    return dict(
        turn_deg_s_p99=round(float(np.percentile(turns, 99)), 2),
        slide_cm_s_p99=round(float(np.percentile(slides, 99)) * 100, 2),
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("recording", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--pose", default="L5", help="L1-L5 or its name: " + ", ".join(RUNGS.values())
    )
    parser.add_argument("--truth", type=Path, help="the drive's microduck-sim --truth file")
    parser.add_argument("--poses-only", action="store_true", help="no MASt3R: L2, L3 or L4")
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
    parser.add_argument(
        "--describe",
        action="append",
        default=[],
        help="robot description override: cameras.exposure=0, motion_noise.time_s=0.006",
    )
    args = parser.parse_args()
    source = RUNGS.get(args.pose, args.pose)
    if source not in RUNGS.values():
        parser.error(f"--pose must be one of {', '.join([*RUNGS, *RUNGS.values()])}")
    rung = next(k for k, v in RUNGS.items() if v == source)
    if source == "truth" and not args.truth:
        parser.error("L1 needs --truth")
    if args.poses_only and source not in ("receive", "source", "fused"):
        parser.error("--poses-only replays L2, L3 or L4")
    out = args.out or Path("e2e/out/slam-replay") / f"{args.recording.name}-{rung}"
    out.mkdir(parents=True, exist_ok=True)
    intrinsics = Intrinsics.model_validate_json((args.recording / "camera.json").read_text())
    rows = [json.loads(line) for line in (args.recording / "frames.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row["gaze"] is not None]
    times = np.array([row["captured"] for row in rows])
    gazes = np.unwrap([row["gaze"] for row in rows])
    priors = [
        None if row["camera"] is None else np.array(row["camera"]).reshape(4, 4) for row in rows
    ]
    motion_path = args.recording / "motion.jsonl"
    motion = (
        [json.loads(line) for line in motion_path.read_text().splitlines() if line.strip()]
        if motion_path.exists()
        else []
    )
    robot = (
        RobotDescription.model_validate_json((args.recording / "robot.json").read_text())
        if (args.recording / "robot.json").exists()
        else None
    )
    for override in args.describe:
        # Recordings from before a field existed replay as a driver sending it would.
        name, value = override.split("=", 1)
        part, key = name.split(".")
        data = robot.model_dump()
        for item in data[part] if part == "cameras" else [data[part]]:
            item[key] = None if value == "none" else float(value)
        robot = RobotDescription.model_validate(data)
    if source in ("source", "fused", "truth", "receive") and not args.poses_only and not motion:
        parser.error(f"{rung} needs a recording with the robot's Motion (motion.jsonl)")
    if source in ("source", "fused") and (not motion or robot is None or not robot.motion_noise):
        parser.error(f"{rung} needs the robot's Motion and its motion noise (robot.json)")
    samples = motion_rows(motion) if motion else np.empty((0, 21))
    delay = robot.cameras[0].capture_delay if robot else 0.0
    truth = None
    if args.truth:
        sys.path.insert(0, str(Path(__file__).parent))
        from truth import Truth

        truth = Truth(args.truth, motion)
    images = {}

    def image(index):
        if index not in images:
            images.clear()
            images[index] = np.asarray(
                Image.open(args.recording / rows[index]["name"]).convert("RGB")
            )
        return images[index]

    matched = {}

    def true_pose(index):
        if index not in matched:
            found = truth.match(image(index), times[index]) if truth else None
            matched[index] = None if found is None else found[1]
        return matched[index]

    buffer = None
    if motion and robot is not None and robot.motion_noise is not None:
        from ito.reconstruction.motion import MotionBuffer

        buffer = MotionBuffer(robot.motion_noise, seconds=1e9)
        buffer.extend(samples)
    by_time = {float(t): i for i, t in enumerate(times)}

    def oracle(captured):
        index = by_time[float(captured)]
        if source == "truth":
            return true_pose(index)
        if source == "receive":
            return priors[index]
        return buffer.at(captured - delay)

    if args.poses_only:
        return poses_only(args, out, rung, source, rows, times, oracle, buffer, delay, true_pose)
    messages = []
    if source == "vision":
        from ito.reconstruction import slam_vision as backend_module
        from ito.reconstruction.slam_vision import VisionSLAMBackend

        backend = VisionSLAMBackend(
            args.budget, intrinsics, report=messages.append, voxel_size=0.04, fade_seconds=0.5
        )
    else:
        from ito.reconstruction.slam import SLAMBackend

        backend = SLAMBackend(
            args.budget,
            intrinsics,
            report=messages.append,
            motion_noise=robot.motion_noise if robot else None,
            frame_delay=delay,
            # Only when known: older backends (another revision first on the path) lack it.
            **({"exposure": exposure} if (exposure := camera_exposure(robot)) is not None else {}),
            oracle=None if source == "fused" else oracle,
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
    fed = 0  # motion samples handed to the backend
    processed = []  # (frame, time, seconds, tracked, keyframes, message, heading, gaze)
    pending = []  # (due time, records, live) snapshots waiting for their future frame
    predictions = []
    estimates, truths, latencies, corrections = [], [], [], []
    every_estimates, every_truths = [], []
    scored, last_published = -1, None  # every recorded frame up to `scored` has been scored
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
        rgb = image(index)
        if args.gaze_prior:
            prior, measured = pose(yaw=row["gaze"]), True
        else:
            prior = (
                prior_at(priors, times, times[index] - args.prior_delay)
                if args.prior_delay
                else priors[index]
            ).astype(np.float32)
            measured = row.get("measured", False) or args.measured
        if source != "vision":
            # The worker has every sample that arrived before it took this frame.
            arrived = int(np.searchsorted(samples[:, 0], now, side="right"))
            backend.add_motion(samples[fed:arrived])
            fed = max(fed, arrived)
            if source == "truth" and true_pose(index) is None:
                now += 0.01
                continue
        tracked_before = backend.tracked
        realigned_before = getattr(backend, "realigned", 0)
        failures_before = getattr(backend, "failures", 0)
        retained = len(getattr(backend, "frames", ()))
        before = len(messages)
        began = time.perf_counter()
        if source == "vision":
            backend.integrate(
                rgb, None, prior, now - start, measured=measured and not args.unmeasured
            )
        else:
            backend.integrate(rgb, None, prior, now - start, captured=times[index])
        torch.cuda.synchronize()
        seconds = time.perf_counter() - began
        news = messages[before:]
        tracked = backend.tracked > tracked_before
        if getattr(backend, "realigned", 0) > realigned_before:
            # Such a backend realigns on its third unmatched frame or on a heading stray.
            misses = getattr(backend_module, "MEASURED_MISSES", 2)
            inferred["unmatched" if failures_before >= misses else "heading"] += 1
        if source == "vision" and any("restarted" in m for m in news):
            # Every retained view failed, or the local solve diverged.
            gave_up = failures_before >= max(retained, 1) + 2
            inferred["unmatched" if gave_up else "failure"] += 1
        estimate = (
            np.asarray(oracle(times[index]), dtype=np.float64)
            if source in ("truth", "receive", "source")
            else backend.camera_pose.astype(np.float64)
        )
        slam = heading(estimate[:3, :3])
        if origin is None:
            origin = (slam, row["gaze"])
        processed.append(
            (
                index,
                now - start,
                seconds,
                tracked,
                len(backend.graph.keyframes) if hasattr(backend, "graph") else len(backend.frames),
                news[-1] if news else "",
                math.remainder(slam - origin[0], math.tau),
                math.remainder(row["gaze"] - origin[1], math.tau),
            )
        )
        if tracked:
            latencies.append(now + seconds - times[index])
            if truth is not None and (true := true_pose(index)) is not None:
                estimates.append(estimate)
                truths.append(true)
            if truth is not None:
                for j in range(scored + 1, index + 1):
                    if j == index:
                        between = estimate
                    elif source != "fused":
                        between = oracle(times[j])
                    elif hasattr(backend, "pose_at"):
                        between = backend.pose_at(times[j])
                    elif last_published is not None:
                        between = interpolate(*last_published, times[index], estimate, times[j])
                    else:
                        continue
                    if between is not None and (true := true_pose(j)) is not None:
                        every_estimates.append(np.asarray(between, dtype=np.float64))
                        every_truths.append(true)
                scored, last_published = index, (times[index], estimate)
            if source == "fused":
                corrections.append(
                    (backend.correction.astype(np.float64), times[index], estimate[:3, 3])
                )
            camera = estimate
            for due, records, live in [p for p in pending if p[0] <= now]:
                pending.remove((due, records, live))
                if now - due > 0.5:
                    continue
                rendered, covered = render(
                    records, live, camera, intrinsics, (rgb.shape[1], rgb.shape[0])
                )
                if covered.mean() > 0.05:
                    error = ((rendered[covered] - rgb[covered] / 255) ** 2).mean()
                    predictions.append((float(covered.mean()), float(-10 * np.log10(error))))
                    if previews < 12 and len(predictions) % 10 == 1:
                        side = np.concatenate((rgb / 255, rendered), axis=1)
                        Image.fromarray((side * 255).astype(np.uint8)).save(
                            out / f"predict-{previews:02d}.png"
                        )
                        previews += 1
            if not pending or now - pending[-1][0] + AHEAD > 0.5:
                live = backend.keys >= 0
                pending.append((now + AHEAD, backend.records.copy(), live.copy()))
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
        rung=rung,
        pose=source,
        measured=not args.unmeasured,
        prior_delay=args.prior_delay,
        overrides=args.set + args.describe,
        seconds=round(float(stamps[-1]), 1),
        frames_available=len(rows),
        frames_processed=len(processed),
        slam_hz=round(len(processed) / float(stamps[-1]), 2),
        frame_ms_median=round(float(np.median(durations)) * 1000, 1),
        frame_ms_p95=round(float(np.percentile(durations, 95)) * 1000, 1),
        frame_to_pose_ms_median=round(float(np.median(latencies)) * 1000, 1) if latencies else None,
        frame_to_pose_ms_p95=round(float(np.percentile(latencies, 95)) * 1000, 1)
        if latencies
        else None,
        lost_frames=int((~ok).sum()),
        matched_frames=getattr(backend, "matched", None),
        frame_delay_ms=round(backend.graph.td * 1000, 1) if hasattr(backend, "graph") else None,
        restarts=restarts,
        restarts_total=sum(restarts.values()),
        restarts_per_100s={k: round(v * per_100s, 1) for k, v in restarts.items()},
        restarts_inferred=not hasattr(backend, "restarts"),
        heading_corrections=getattr(backend, "corrections", None),
        eye_corrections=eye_corrections(corrections, stamps) if corrections else None,
        flat_fraction=round(flat / float(stamps[-1]), 3),
        gaze_range_deg=round(math.degrees(np.ptp([p[7] for p in processed])), 1),
        heading_error_deg_median=round(float(np.median(stray)), 1) if len(stray) else None,
        heading_error_deg_p95=round(float(np.percentile(stray, 95)), 1) if len(stray) else None,
        heading_error_deg_max=round(float(stray.max()), 1) if len(stray) else None,
        truth=score_truth(estimates, truths) if truth else None,
        every_frame=score_truth(every_estimates, every_truths) if truth else None,
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
    )


def camera_exposure(robot):
    return getattr(robot.cameras[0], "exposure", None) if robot and robot.cameras else None


def interpolate(t0, pose0, t1, pose1, t):
    """Between two world-from-camera poses by time: slerp and lerp."""
    from scipy.spatial.transform import Rotation

    f = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
    ends = Rotation.from_matrix(np.array([pose0[:3, :3], pose1[:3, :3]]))
    result = np.eye(4)
    turn = (ends[0].inv() * ends[1]).as_rotvec()
    result[:3, :3] = (ends[0] * Rotation.from_rotvec(turn * f)).as_matrix()
    result[:3, 3] = pose0[:3, 3] * (1 - f) + pose1[:3, 3] * f
    return result


def score_truth(estimates, truths):
    from truth import score

    return score(estimates, truths)


def poses_only(args, out, rung, source, rows, times, oracle, buffer, delay, true_pose):
    """Every frame's pose from the robot alone, or fusion without vision; no MASt3R."""
    from ito.reconstruction.fusion import PoseGraph

    graph = PoseGraph(buffer, td=delay) if source == "fused" else None
    estimates, truths, corrections = [], [], []
    began = time.perf_counter()
    for index, captured in enumerate(times):
        if args.limit and captured - times[0] > args.limit:
            break
        if graph is not None:
            if not buffer.covers(captured - graph.td):
                continue
            node = graph.add_frame(captured)
            graph.solve_frame()
            if index % 6 == 0:
                graph.promote()
                graph.solve_window()
            estimate = graph.pose(node)
            corrections.append((graph.correction(node), captured, estimate[:3, 3]))
        else:
            estimate = oracle(captured)
            if estimate is None:
                continue
        true = true_pose(index) if args.truth else None
        if true is not None:
            estimates.append(np.asarray(estimate, dtype=np.float64))
            truths.append(true)
    report = dict(
        recording=str(args.recording),
        rung=rung,
        pose=source,
        poses_only=True,
        frames=len(times),
        cpu_ms_per_frame=round((time.perf_counter() - began) * 1000 / max(len(times), 1), 2),
        eye_corrections=eye_corrections(corrections, None) if corrections else None,
        restarts_total=0,
        truth=score_truth(estimates, truths) if args.truth else None,
    )
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
