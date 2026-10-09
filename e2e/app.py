"""Real pilot CLI + MuJoCo, SDL input, arming, driver kill/restart and latency measurements.

DISPLAY=:97 LIBGL_ALWAYS_SOFTWARE=1 uv run python e2e/app.py
"""

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import pygame
from pygame._sdl2 import Window

from ito import clock
from ito.app.__main__ import main as pilot_main
from ito.driver import pairing
from ito.link.pairing import display

OUT = Path("e2e/out/app")


def key(code):
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=code))
    pygame.event.post(pygame.event.Event(pygame.KEYUP, key=code))


def click_scene():
    for kind in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
        fields = {"rel": (0, 0), "buttons": (0, 0, 0)} if kind == pygame.MOUSEMOTION else {}
        pygame.event.post(pygame.event.Event(kind, pos=(600, 330), button=1, **fields))


def focus(gained):
    pygame.event.post(
        pygame.event.Event(pygame.WINDOWFOCUSGAINED if gained else pygame.WINDOWFOCUSLOST)
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for path in OUT.glob("capture-*.png"):
        path.unlink()
    os.environ["XDG_CONFIG_HOME"] = os.environ["APPDATA"] = str(OUT / "config")
    gl = {}
    if sys.platform == "linux":
        gl = {"LD_LIBRARY_PATH": "/opt/data/lib/osmesa", "MUJOCO_GL": "osmesa"}
    with socket.socket() as port:
        port.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{port.getsockname()[1]}"
    log = (OUT / "driver.log").open("w")
    code_file = OUT / "pairing-code"
    code = pairing.rotate(code_file)

    def driver():
        return subprocess.Popen(
            [sys.executable, "-m", "drivers.mujoco.cli", "--port", address.split(":")[1]]
            + ["--pairing-file", str(code_file)],
            env=os.environ | gl,
            stdout=log,
            stderr=log,
        )

    robot = driver()
    began = clock.now()
    stage = 0
    changed = began
    previous = began
    samples = []
    statuses = []
    positions = []
    counts = []
    killed = restarted = None
    first_position = None
    reached_stop = False
    stall = False
    last_revisions = 0
    worker = None
    first_camera = None
    new_surfaces = 0
    stalled_input_ages = []
    release_latency_ms = None
    estop_latency_ms = None
    tested_input_stall = False
    steady_start = steady_end = None
    worker_restart_s = resume = None

    def drive(app, window, value):
        nonlocal stage, changed, previous, robot, killed, restarted, first_position
        nonlocal reached_stop, stall, last_revisions, worker, resume
        nonlocal first_camera, new_surfaces
        nonlocal release_latency_ms, estop_latency_ms, tested_input_stall
        nonlocal steady_start, steady_end, worker_restart_s
        now = clock.now()
        assert now - began < 80, (stage, app.state, app.telemetry)
        samples.append((now, now - previous, app.state.status.link))
        previous = now
        statuses.append((now, app.state.status.link, app.state.status.e_stop))
        t = app.telemetry
        if t:
            positions.append(t.copy())
        counts.append(window.renderer.count)

        def stopped():
            held = app.state.status.detail.startswith(("stopped", "e-stopped"))
            return held and t["left_command"] == t["right_command"] == 0

        def moving():
            return app.state.status.armed and t["left_command"] > 0

        def settled(condition, limit=4):
            """Driver status lags input by a few hundred ms on a loaded software GPU."""
            assert now - changed < limit, (stage, app.state.status, t)
            return condition()

        if stage == 0 and app.matched_frames > 12 and window.renderer.count > 1000:
            steady_start = now
            stage, changed = "steady", now
        elif stage == "steady" and now - changed > 6:
            steady_end = now
            Window.from_display_module().focus()
            first_position = (t["base_x"], t["base_y"])
            first_camera = app.camera_pose.copy()
            key(pygame.K_F12)
            stage, changed = "captured", now
        elif stage == "captured" and now - changed > 0.5:
            assert not app.state.status.armed and t["left_command"] == t["right_command"] == 0
            key(pygame.K_e)
            stage, changed = "estop", now
        elif stage == "estop" and settled(lambda: app.state.status.e_stop and stopped()):
            # Focus loss and a click back in must never release an e-stop.
            focus(False)
            stage, changed = "estop_refocus", now
        elif stage == "estop_refocus" and now - changed > 0.3:
            focus(True)
            click_scene()
            stage, changed = "estop_drive", now
        elif stage == "estop_drive" and now - changed > 0.3:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w))
            stage, changed = "estop_held", now
        elif stage == "estop_held" and now - changed > 1.5:
            assert value.movement[2] > 0 and window.input.captured
            assert app.state.status.e_stop and not app.state.status.armed, app.state.status
            assert stopped(), app.state.status
            key(pygame.K_r)
            stage, changed = "resumed", now
        elif stage == "resumed" and settled(moving):
            key(pygame.K_SPACE)
            stage, changed = "space", now
        elif stage == "space" and settled(stopped):
            assert not app.state.status.armed and not app.state.status.focus_hold
            focus(False)
            stage, changed = "space_refocus", now
        elif stage == "space_refocus" and now - changed > 0.3:
            # A deliberate Space stop is never undone by clicking back in.
            assert not app.state.status.focus_hold, app.state.status
            focus(True)
            click_scene()
            stage, changed = "space_drive", now
        elif stage == "space_drive" and now - changed > 0.3:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w))
            stage, changed = "space_held", now
        elif stage == "space_held" and now - changed > 1.5:
            assert value.movement[2] > 0 and window.input.captured
            assert not app.state.status.armed and stopped(), app.state.status
            key(pygame.K_r)
            stage, changed = "driving", now
        elif stage == "driving" and settled(moving):
            focus(False)
            stage, changed = "focus_lost", now
        elif stage == "focus_lost" and settled(lambda: stopped() and app.state.status.focus_hold):
            assert not app.state.status.armed and not app.state.status.e_stop
            focus(True)
            stage, changed = "refocused", now
        elif stage == "refocused" and now - changed > 0.3:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w))
            stage, changed = "ignored", now
        elif stage == "ignored" and now - changed > 1:
            # Refocused but not clicked: WASD is ignored and the prompt says why.
            assert value.movement[2] > 0 and not app.state.status.armed and stopped()
            key(pygame.K_F12)
            stage, changed = "click", now
        elif stage == "click" and now - changed > 0.3:
            click_scene()
            stage, changed = "rearmed", now
        elif stage == "rearmed" and settled(moving):
            assert not app.state.status.focus_hold
            # Driving with W held and the mouse captured, as stage 1 expects.
            stage, changed = 1, now
        elif stage == 1 and not tested_input_stall and t.get("active"):
            # Block the display for longer than both watchdogs. Held input must
            # stay live, and releasing W must reach the robot before we draw again.
            until = clock.now() + 0.65
            while clock.now() < until:
                age = clock.now() - app.latest_input.timestamp
                stalled_input_ages.append(age * 1000)
                assert age < 0.2, age
                assert app.telemetry["active"], app.state.status
                time.sleep(0.01)
            released = clock.now()
            pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame.K_w))
            while app.telemetry["left_command"] or app.telemetry["right_command"]:
                assert clock.now() - released < 0.5, app.telemetry
                time.sleep(0.01)
            assert app.telemetry["active"], app.state.status
            release_latency_ms = (clock.now() - released) * 1000
            stopped = clock.now()
            key(pygame.K_e)
            while not app.state.status.e_stop:
                assert clock.now() - stopped < 0.5, app.state.status
                time.sleep(0.01)
            estop_latency_ms = (clock.now() - stopped) * 1000
            assert app.telemetry["left_command"] == app.telemetry["right_command"] == 0
            key(pygame.K_r)
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w))
            tested_input_stall = True
        elif stage == 1 and tested_input_stall and now - changed > 2:
            # A grab's first pointer report is its baseline; the turn follows it.
            for rel in ((0, 0), (-260, -25)):
                pygame.event.post(
                    pygame.event.Event(
                        pygame.MOUSEMOTION, pos=(600, 400), rel=rel, buttons=(0, 0, 0)
                    )
                )
            stage, changed = 2, now
        elif stage == 2 and now - changed > 2:
            assert np.linalg.norm(np.array((t["base_x"], t["base_y"])) - first_position) > 0.4, (
                t,
                value,
                window.input.keys,
                window.input.active,
            )
            assert t["head_pan"] + t["base_yaw"] > 0.4, t
            records = np.frombuffer(window.renderer.scene_buffer.read(), np.float32)
            records = records.reshape(-1, 4, 4)
            points = records[records[:, 0, 3] > 0, 0, :3]
            local = (points - first_camera[:3, 3]) @ first_camera[:3, :3]
            new_surfaces = int(np.count_nonzero(np.abs(local[:, 0]) > -local[:, 2] * 0.78))
            assert new_surfaces > 100, "Turning did not reconstruct beyond the first camera view"
            key(pygame.K_F12)
            key(pygame.K_e)
            stage, changed = 3, now
        elif stage == 3 and now - changed > 1:
            assert app.state.status.e_stop and t["left_command"] == t["right_command"] == 0
            assert abs(t["left_velocity"]) < 0.3 and abs(t["right_velocity"]) < 0.3, t
            reached_stop = True
            key(pygame.K_F12)
            # Freeze the real reconstruction process while input and display continue.
            worker = psutil.Process(app.worker.process.pid)
            worker.suspend()
            resume = threading.Timer(2, worker.resume)
            resume.start()
            last_revisions = app.matched_frames
            stall = True
            stage, changed = 4, now
        elif stage == 4 and now - changed > 1.5:
            resume.cancel()
            worker.resume()
            stall = False
            assert app.matched_frames > last_revisions
            # A crashed RGB-D worker leaves the link up, shows flat video and starts again.
            worker.kill()
            stage, changed = "worker killed", now
        elif stage == "worker killed" and "restarting" in app.reconstruction_status:
            assert app.state.flat_video, "a failed worker left the 3D view up"
            assert app.state.status.link == "CONNECTED", app.state.status
            assert app.connections == 1, app.connections
            stage, changed = "worker restarting", now
        elif (
            stage == "worker restarting"
            and app.worker
            and app.worker.process.pid != worker.pid
            and not app.state.flat_video
            and window.renderer.epoch == app.worker.epoch
            and window.renderer.count > 1000
        ):
            assert app.state.status.link == "CONNECTED", app.state.status
            assert app.connections == 1, app.connections
            worker_restart_s = now - changed
            pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame.K_w))
            robot.kill()
            robot.wait()
            killed = now
            stage, changed = 5, now
        elif (
            stage == 5
            and now - changed > 4
            # Collect the same >30 offline frames even on a slow software GPU.
            # The current sample is excluded by the final t < restarted filter.
            and sum(stamp > killed + 1 for stamp, _, _ in samples) > 31
        ):
            assert app.state.status.link == "RECONNECTING", app.state
            key(pygame.K_F12)
            robot = driver()
            restarted = now
            stage, changed = 6, now
        elif (
            stage == 6
            and app.connections >= 2
            and app.telemetry
            and window.renderer.epoch == app.worker.epoch
            and now - changed > 3
        ):
            assert app.telemetry["left_command"] == app.telemetry["right_command"] == 0
            key(pygame.K_r)
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w))
            stage, changed = 7, now
        elif stage == 7 and now - changed > 2:
            assert app.telemetry["left_command"] > 0
            assert window.renderer.count > 1000
            key(pygame.K_F12)
            key(pygame.K_SPACE)
            stage, changed = 8, now
        elif stage == 8 and now - changed > 0.6:
            assert app.telemetry["left_command"] == app.telemetry["right_command"] == 0
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            stage = 9

    try:
        result = pilot_main(
            [
                address,
                "--code",
                code,
                "--size",
                "800",
                "600",
                "--fps",
                "60",
                "--max-splats",
                "16384",
                "--fov",
                "75",
                "--sensitivity",
                "0.0025",
                "--metrics",
                str(OUT / "metrics.jsonl"),
                "--capture-dir",
                str(OUT),
            ],
            on_frame=drive,
        )
        assert result == 0 and stage == 9 and reached_stop and not stall
        reloaded = False

        def reload_settings(app, window, value):
            nonlocal reloaded
            if app.settings_revision and app.connections:
                assert app.settings.fov == 75 and np.isclose(window.fov, np.deg2rad(75))
                reloaded = True
                pygame.event.post(pygame.event.Event(pygame.QUIT))

        assert (
            pilot_main(
                [f"http://{address}/", "--size", "320", "240", "--frames", "600"],
                on_frame=reload_settings,
            )
            == 0
        )
        assert reloaded, "Per-robot comfort settings or pairing did not survive restart"
        # Exercise the installed console entry point as well as SDL injection above.
        with (OUT / "cli.log").open("w") as cli_log:
            subprocess.run(
                [
                    str(Path(sys.executable).with_name("ito")),
                    address,
                    "--size",
                    "480",
                    "360",
                    "--frames",
                    "300",
                    # Empty frames are cheap: leave a real startup observation
                    # window before this installed-entry-point smoke check exits.
                    "--fps",
                    "10",
                    "--metrics",
                    str(OUT / "cli-metrics.jsonl"),
                ],
                stdout=cli_log,
                stderr=cli_log,
                check=True,
                timeout=120,
            )
        cli_rows = [
            json.loads(line) for line in (OUT / "cli-metrics.jsonl").read_text().splitlines()
        ]
        assert any(row["link"] == "CONNECTED" and row["gaussians"] > 1000 for row in cli_rows)
    finally:
        if robot.poll() is None:
            robot.terminate()
            robot.wait(timeout=8)
        log.close()
    # The restarted driver knows the pilot and keeps its spent code to itself.
    driver_log = (OUT / "driver.log").read_text()
    assert driver_log.count(f"Pairing code: {display(code)}") == 1, driver_log
    assert driver_log.count("Pairing code used by a paired pilot") == 1, driver_log
    rows = [json.loads(line) for line in (OUT / "metrics.jsonl").read_text().splitlines()]
    latency = [
        r["pilot_input_to_robot_ms"] for r in rows if r["pilot_input_to_robot_ms"] is not None
    ]
    # A paused/reconnecting scene repeats its last visibility measurement on
    # every display frame. Measure each exposure once, not once per redraw.
    exposures = {}
    steady_exposures = {}
    for row in rows:
        if row["capture_to_splat_visible_ms"] is not None and row["link"] == "CONNECTED":
            exposures.setdefault(row["scene_capture_time"], row["capture_to_splat_visible_ms"])
            # Measure normal pipeline latency before deliberately freezing it
            # or blocking on screenshot readback. Retain fault-phase metrics too.
            if steady_start <= row["time"] < steady_end:
                steady_exposures.setdefault(
                    row["scene_capture_time"], row["capture_to_splat_visible_ms"]
                )
    visible = list(steady_exposures.values())
    connected = [dt for t, dt, link in samples if link == "CONNECTED" and dt < 0.3]
    offline = [dt for t, dt, link in samples if killed + 1 < t < restarted]
    assert len(offline) > 30
    assert np.percentile(offline, 95) < max(0.15, np.percentile(connected, 95) * 2)
    assert len(latency) > 5 and len(visible) > 5
    assert np.median(latency) < 150 and np.median(visible) < 1000
    captures = sorted(OUT.glob("capture-*.png"))
    assert len(captures) == 6
    prompt = captures.pop(1)  # Taken while focus loss held the robot and W was pressed.
    banner = np.transpose(pygame.surfarray.array3d(pygame.image.load(prompt)), (1, 0, 2))
    banner = banner[520:580, 250:550].astype(float)
    assert (banner[..., 0] - banner[..., 2]).mean() > 50, "Stopped-robot prompt not visible"
    images = [
        pygame.surfarray.array3d(pygame.image.load(p))[:, 280:].astype(float) for p in captures
    ]
    assert images[0].std() > 15, "Empty reconstructed scene"
    assert np.abs(images[0] - images[1]).mean() > 10, "Scene did not follow the moving robot"
    assert images[-1].std() > 15, "Scene did not recover"
    saved = list((OUT / "config/ito/robots").glob("*.json"))
    assert any(json.loads(path.read_text())["fov"] == 75 for path in saved)
    report = {
        "pilot_input_to_robot_ms_median": float(np.median(latency)),
        "camera_capture_to_splat_visible_ms_median": float(np.median(visible)),
        "all_exposures_including_faults_ms_median": float(np.median(list(exposures.values()))),
        "steady_exposures": len(visible),
        "connected_frame_ms_p95": float(np.percentile(connected, 95) * 1000),
        "driver_dead_frame_ms_p95": float(np.percentile(offline, 95) * 1000),
        "rgbd_worker_restart_s": round(worker_restart_s, 2),
        "max_splats": max(counts),
        "splats_beyond_initial_view": new_surfaces,
        "display_frames": len(samples),
        "stalled_display_input_age_ms_max": max(stalled_input_ages),
        "stalled_display_key_release_ms": release_latency_ms,
        "stalled_display_estop_ms": estop_latency_ms,
        "captures": list(map(str, captures)),
        "stopped_prompt": str(prompt),
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(
        "PASS: live room, physical drive/head motion, e-stop, worker stall, reconnect, settings, "
        "stopped prompt, focus-loss resume, no auto-resume after stop/e-stop"
    )


if __name__ == "__main__":
    main()
