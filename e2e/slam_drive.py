"""Pilot a live robot through a scripted walk with MASt3R-SLAM; measure the 3D view.

uv run --extra slam python e2e/slam_drive.py HOST:PORT [--code CODE] [--record DIR]
    [--route short|tour] [--laps N]

The short route walks 3 s forward, looks left and right, and walks 3 s back. The tour
looks left, right and behind, walks, strafes, reverses and turns while walking. Then it
reports how long the pilot saw the flat feed and why, the SLAM rate, capture-to-scene
latency and how well the SLAM camera's heading follows the robot's measured gaze (body
heading plus head pan from telemetry), with a screenshot per leg. --record saves every
frame handed to reconstruction, with the robot's pose prior and gaze, for e2e/slam_replay.py.
"""

import argparse
import json
import math
import os
import queue
import threading
from collections import Counter
from pathlib import Path

import numpy as np
import pygame
from PIL import Image

from ito import clock
from ito.app.__main__ import main as pilot_main
from ito.app.pilot import TRACKING_LOST
from ito.reconstruction import Reconstruction
from ito.app import frames as _frames
from ito.link import media as _media

# Frame-flow counters, dumped to flow.jsonl each second: how many of the robot's video frames
# reach the pilot's track, pair with their metadata (by kind) and get offered to SLAM.
FLOW = Counter()
JOINS = []  # the live FrameJoin, for its per-kind join counts
_put = _media.LatestTrack._put


def _counting_put(self, frame):
    if self.kind == "video" and frame is not None:
        FLOW["track_in"] += 1
        FLOW["track_dropped"] += self._frames.full()
    _put(self, frame)


_media.LatestTrack._put = _counting_put
_decoded, _described = _frames.FrameJoin.decoded, _frames.FrameJoin.described


def _counting_decoded(self, frame):
    if not JOINS or JOINS[-1] is not self:
        JOINS.append(self)
    FLOW["decoded"] += 1
    pairs = _decoded(self, frame)
    FLOW["joined"] += len(pairs)
    return pairs


def _counting_described(self, metadata):
    FLOW["metadata"] += 1
    pairs = _described(self, metadata)
    FLOW["joined"] += len(pairs)
    return pairs


_frames.FrameJoin.decoded = _counting_decoded
_frames.FrameJoin.described = _counting_described

# (gaze yaw degrees, gaze pitch degrees, held keys, seconds to hold once there).
# Positive yaw looks left. The body follows the gaze once the head runs out of pan.
ROUTES = {}
ROUTES["short"] = [
    (0, 0, "", 2),
    (0, 0, "w", 3),
    (0, 0, "", 1),
    (90, 0, "", 2),
    (-90, 0, "", 2),
    (0, 0, "", 1),
    (0, 0, "s", 3),
    (0, 0, "", 2),
]
ROUTES["tour"] = [
    (0, 0, "", 4),
    (90, 0, "", 4),
    (180, 0, "", 4),
    (90, 0, "", 3),
    (0, 0, "", 3),
    (-90, 0, "", 4),
    (0, 0, "", 3),
    (0, 0, "w", 6),
    (0, 20, "", 2),
    (0, -25, "", 2),
    (0, 0, "", 2),
    (60, 0, "w", 5),
    (60, 0, "d", 3),
    (60, 0, "a", 3),
    (60, 0, "s", 4),
    (-60, 0, "w", 6),
    (-150, 0, "", 4),
    (-150, 0, "w", 5),
    (0, 0, "", 4),
]
ROUTE = ROUTES["tour"]
KEYS = {"w": pygame.K_w, "a": pygame.K_a, "s": pygame.K_s, "d": pygame.K_d}
TURN_RATE = math.radians(45)  # A brisk but ordinary head turn.
# Closed loop (#25): the gait walks backward slower than forward and looks past the head's pan
# limit turn the body, so an open-loop route creeps forward and skews lap by lap into walls.
HOME_MARGIN = 0.02  # m short of the start, along the start heading, that counts as back
FACE_TOLERANCE = math.radians(4)  # body yaw off the start heading that a lap re-faces
FACE_OVERSHOOT = 1.40 + math.radians(5)  # gaze past the pan limit (GAZE_PAN) turns the body
STALL_WINDOW, STALL_DISTANCE = 1.0, 0.02  # walking yet moved under 2 cm in 1 s: a wall


def key(code, down=None):
    if down is None:
        key(code, True)
        key(code, False)
    else:
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN if down else pygame.KEYUP, key=code))


def heading(matrix):
    """Yaw of a world-from-camera pose's view direction (-Z), positive to the left."""
    forward = -np.asarray(matrix)[:3, 2]
    return math.atan2(-forward[0], -forward[2])


def gaze(telemetry):
    """The robot camera's heading: its head FK when it reports one, else body yaw plus pan."""
    if "camera_yaw" in telemetry:
        return telemetry["camera_yaw"]
    if "base_yaw" not in telemetry:
        return None
    return math.remainder(telemetry["base_yaw"] + telemetry.get("head_yaw", 0.0), 2 * math.pi)


def reason(app, now):
    """Why the pilot sees the flat feed on this display frame."""
    if app.failure:
        return "worker failed"
    if not app.tracked_frames:
        return "starting"
    if not app.tracking:
        return "tracking lost"
    if now - app.last_tracking >= TRACKING_LOST:
        return "no pose update"
    return "settling"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("address")
    parser.add_argument("--code")
    parser.add_argument("--out", type=Path, default=Path("e2e/out/slam-drive"))
    parser.add_argument("--record", type=Path, help="save the frames reconstruction received")
    parser.add_argument("--route", choices=sorted(ROUTES), default="short")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument(
        "--open-loop",
        action="store_true",
        help="walk on timers only: no walking back to the start, re-facing or stall aborts",
    )
    parser.add_argument("--startup-timeout", type=float, default=300)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("capture-*.png"):
        old.unlink()
    # A private settings folder; the robot's one pilot credential may be copied in first.
    os.environ["XDG_CONFIG_HOME"] = os.environ["APPDATA"] = str(out / "config")
    route = ROUTES[args.route] * args.laps
    # One line per frame the reconstruction worker took, and per frame offered to it.
    (out / "timeline.jsonl").unlink(missing_ok=True)
    os.environ["ITO_TIMELINE"] = str(out / "timeline.jsonl")
    offered = (out / "offered.jsonl").open("w", buffering=1)
    legs = (out / "legs.jsonl").open("w", buffering=1)
    flow_file = (out / "flow.jsonl").open("w", buffering=1)

    def flow_dump():
        while True:
            joins = JOINS[-1].joins if JOINS else {}
            flow_file.write(json.dumps(dict(at=round(clock.now(), 3), **FLOW, joins=joins)) + "\n")
            threading.Event().wait(1.0)

    threading.Thread(target=flow_dump, daemon=True).start()
    lap = len(ROUTES[args.route])
    home = {}  # base x, y, yaw when the drive began
    walked = []  # (time, x, y) during the current walking leg
    facing = None  # when this lap's re-facing turn began
    closed = Counter()  # stalls, re-facings, back legs ended at the start
    pilot = None
    recorded = queue.Queue(maxsize=512)
    dropped = 0

    offer = Reconstruction.submit

    def offering_submit(self, rgb, depth=None, camera=None, captured_at=None, **flags):
        accepted = offer(self, rgb, depth, camera, captured_at, **flags)
        offered.write(
            json.dumps(
                dict(
                    at=round(clock.now(), 4),
                    captured=None if captured_at is None else round(captured_at, 4),
                    accepted=accepted,
                    seq=self.sequence.value,
                    gaze=None if pilot is None else gaze(pilot.telemetry),
                )
            )
            + "\n"
        )
        return accepted

    Reconstruction.submit = offering_submit

    if args.record:
        args.record.mkdir(parents=True, exist_ok=True)
        for old in args.record.glob("*.jpg"):
            old.unlink()
        submit = Reconstruction.submit

        def recording_submit(self, rgb, depth=None, camera=None, captured_at=None, **flags):
            nonlocal dropped
            accepted = submit(self, rgb, depth, camera, captured_at, **flags)
            if pilot is not None:
                telemetry = pilot.telemetry
                row = dict(
                    captured=captured_at,
                    accepted=accepted,
                    measured=flags.get("measured", False),
                    camera=None if camera is None else np.asarray(camera).ravel().tolist(),
                    gaze=gaze(telemetry),
                    base=[telemetry.get(k) for k in ("base_x", "base_y", "base_yaw")],
                    head=[telemetry.get(k) for k in ("head_yaw", "head_pitch", "neck_pitch")],
                )
                try:
                    recorded.put_nowait((rgb.copy(), row))
                except queue.Full:
                    dropped += 1
            return accepted

        Reconstruction.submit = recording_submit

        def writer():
            (args.record / "camera.json").unlink(missing_ok=True)
            with (args.record / "frames.jsonl").open("w") as index:
                count = 0
                while (item := recorded.get()) is not None:
                    rgb, row = item
                    count += 1
                    row["name"] = f"{count:06d}.jpg"
                    Image.fromarray(rgb).save(args.record / row["name"], quality=95)
                    index.write(json.dumps(row) + "\n")

        saver = threading.Thread(target=writer, daemon=True)
        saver.start()

    leg = -1
    leg_started = began = previous = clock.now()
    arrived = None
    ready_at = None
    held = ""
    samples = []  # (dt, flat, reason) per display frame after the first 3D view
    statuses = Counter()
    transitions = []
    last_status = None
    poses = []  # (seconds, tracked frame count, SLAM heading, robot gaze) on each update
    workers = set()
    error = None
    origin = {}
    rearmed = 0.0

    def drive(app, window, value):
        nonlocal pilot, leg, leg_started, arrived, ready_at, held, previous, last_status, error
        nonlocal rearmed, facing
        pilot = app
        now = clock.now()
        dt, previous = now - previous, now
        if app.worker:
            workers.add(app.worker.process.pid)
            if args.record and not (args.record / "camera.json").exists():
                (args.record / "camera.json").write_text(app.worker.intrinsics.model_dump_json())
        status = app.reconstruction_status
        if status != last_status:
            # Frame counters change every update; keep the kind of message only.
            kind = status.split(" | ")[0]
            if not transitions or transitions[-1][1] != kind:
                transitions.append((round(now - began, 2), kind))
            statuses[kind] += 1
            last_status = status
        if app.refusal:
            error = f"The robot refused this pilot: {app.refusal}"
        if ready_at is None:
            if error or now - began > args.startup_timeout:
                error = error or f"SLAM never showed the 3D view: {status!r} {app.failure!r}"
                pygame.event.post(pygame.event.Event(pygame.QUIT))
                return
            if app.tracked_frames and not app.state.flat_video and gaze(app.telemetry) is not None:
                ready_at = leg_started = now
                window.input.yaw = window.input.pitch = 0.0
                key(pygame.K_r)
                key(pygame.K_F12)
            return
        if leg == len(route):
            return
        if not app.armed and now - rearmed > 1:
            # A display hitch (a screenshot being saved) disarms the robot, as it should.
            key(pygame.K_r)
            rearmed = now
        if app.connections > 1 or app.state.status.link != "CONNECTED":
            error = f"Robot link dropped during the drive: {app.state.status.detail}"
            leg = len(route)
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            return
        flat = app.state.flat_video
        samples.append((dt, flat, reason(app, now) if flat else None))
        looked = gaze(app.telemetry)
        if app.tracked_frames != (poses[-1][1] if poses else None) and looked is not None:
            if not origin:
                origin.update(slam=heading(app.camera_pose), gaze=looked)
            poses.append(
                (
                    now - ready_at,
                    app.tracked_frames,
                    math.remainder(heading(app.camera_pose) - origin["slam"], 2 * math.pi),
                    math.remainder(looked - origin["gaze"], 2 * math.pi),
                    app.tracking,
                )
            )
        base = [app.telemetry.get(k) for k in ("base_x", "base_y", "base_yaw")]
        base = None if None in base else base
        if base and not home:
            home.update(x=base[0], y=base[1], yaw=base[2])
        if base and held:
            walked.append((now, base[0], base[1]))
        if leg >= 0:
            target_yaw, target_pitch, _, hold = route[leg]
            if facing is not None:
                skew = math.remainder(home["yaw"] - base[2], 2 * math.pi) if base else 0.0
                if abs(skew) < FACE_TOLERANCE / 2 or now - facing > 8:
                    legs.write(json.dumps(dict(leg=leg, at=round(now, 4), faced=round(math.degrees(skew), 1))) + "\n")
                    facing = None
                else:
                    target_yaw = math.degrees(home["yaw"] + math.copysign(FACE_OVERSHOOT, skew))
            yaw_error = math.remainder(math.radians(target_yaw) - window.input.yaw, 2 * math.pi)
            pitch_error = math.radians(target_pitch) - window.input.pitch
            step = TURN_RATE * min(dt, 0.05)
            window.input.yaw += float(np.clip(yaw_error, -step, step))
            window.input.pitch += float(np.clip(pitch_error, -step, step))
            if max(abs(yaw_error), abs(pitch_error)) > step or facing is not None:
                arrived = None
            elif arrived is None:
                arrived = now
        done = leg < 0
        if not done and arrived is not None:
            if "s" in held and not args.open_loop and home and base:
                # Walk back to where the drive began, not for a fixed time (3x as a cap).
                ahead = (base[0] - home["x"]) * math.cos(home["yaw"]) + (base[1] - home["y"]) * math.sin(home["yaw"])
                back = ahead <= HOME_MARGIN
                closed["home"] += back
                done = back or now - arrived >= 3 * route[leg][3]
            else:
                done = now - arrived >= route[leg][3]
        if held and not args.open_loop and not done and now - leg_started > 1.5 and walked:
            recent = [w for w in walked if now - w[0] <= STALL_WINDOW]
            if recent and recent[0][0] <= now - 0.9 * STALL_WINDOW:
                moved = math.hypot(recent[-1][1] - recent[0][1], recent[-1][2] - recent[0][2])
                if moved < STALL_DISTANCE:
                    closed["stalls"] += 1
                    legs.write(json.dumps(dict(leg=leg, at=round(now, 4), stalled=round(moved, 3))) + "\n")
                    done = True
        if done:
            for name in held:
                key(KEYS[name], False)
            leg += 1
            arrived = None
            leg_started = now
            key(pygame.K_F12)
            if leg == len(route):
                pygame.event.post(pygame.event.Event(pygame.QUIT))
                return
            held = route[leg][2]
            walked.clear()
            legs.write(json.dumps(dict(leg=leg, at=round(now, 4), step=route[leg])) + "\n")
            if leg % lap == 0 and leg and not args.open_loop and home and base:
                # Each lap starts facing the way the first did.
                if abs(math.remainder(home["yaw"] - base[2], 2 * math.pi)) > FACE_TOLERANCE:
                    facing = now
                    closed["faced"] += 1
            for name in held:
                key(KEYS[name], True)

    result = pilot_main(
        [
            args.address,
            *(["--code", args.code] if args.code else []),
            "--size",
            "1280",
            "720",
            "--fps",
            "60",
            "--reconstruction",
            "slam",
            "--audio-source",
            "none",
            "--audio-sink",
            "none",
            "--capture-dir",
            str(out),
            "--metrics",
            str(out / "metrics.jsonl"),
        ],
        on_frame=drive,
    )
    if args.record:
        recorded.put(None)
        saver.join()
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    latencies = [
        r["capture_to_splat_visible_ms"]
        for r in rows
        if not r["flat_video"] and r["capture_to_splat_visible_ms"] is not None
    ]
    total = sum(dt for dt, _, _ in samples)
    flat_time = Counter()
    for dt, flat, why in samples:
        if flat:
            flat_time[why] += dt
    episodes = sum(1 for a, b in zip(samples, samples[1:], strict=False) if b[1] and not a[1])
    tracked = poses[-1][1] - poses[0][1] if len(poses) > 1 else 0
    # Heading error once the SLAM pose has had a moment to follow the gaze.
    heading_error = [
        abs(math.degrees(math.remainder(slam - looked, 2 * math.pi)))
        for _, _, slam, looked, ok in poses
        if ok
    ]
    base_end = pilot and [pilot.telemetry.get(k) for k in ("base_x", "base_y")]
    base_end = None if not base_end or None in base_end else base_end
    report = dict(
        result=result,
        error=error,
        startup_s=None if ready_at is None else round(ready_at - began, 1),
        driven_s=round(total, 1),
        flat_fraction=round(sum(flat_time.values()) / total, 3) if total else None,
        flat_seconds_by_reason={k: round(v, 1) for k, v in flat_time.most_common()},
        flat_episodes=episodes,
        status_counts=dict(statuses.most_common()),
        worker_processes=len(workers),
        recorded_dropped=dropped,
        capture_to_scene_ms_median=float(np.median(latencies)) if latencies else None,
        capture_to_scene_ms_p95=float(np.percentile(latencies, 95)) if latencies else None,
        tracked_frames=tracked,
        tracking_hz=round(tracked / (poses[-1][0] - poses[0][0]), 2) if tracked else None,
        heading_error_deg_median=round(float(np.median(heading_error)), 1)
        if heading_error
        else None,
        heading_error_deg_p95=round(float(np.percentile(heading_error, 95)), 1)
        if heading_error
        else None,
        closed_loop=None if args.open_loop else dict(closed),
        drift_m=round(math.hypot(base_end[0] - home["x"], base_end[1] - home["y"]), 3)
        if home and base_end
        else None,
        gaze_range_deg=round(math.degrees(np.ptp([p[3] for p in poses])), 1) if poses else None,
        status_timeline=transitions[:300],
    )
    (out / "poses.json").write_text(json.dumps(poses))
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "status_timeline"}, indent=2))
    if error:
        raise SystemExit(error)


if __name__ == "__main__":
    main()
