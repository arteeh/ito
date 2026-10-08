"""Real OpenXR app + MuJoCo. Start the active XR runtime, then: uv run python e2e/openxr.py.

Windows: start SteamVR (or Virtual Desktop with VDXR active), wear the headset, run
`uv run python e2e/openxr.py`. Captures, metrics and logs go to e2e/out/xr.
Linux: DISPLAY=:97 LIBGL_ALWAYS_SOFTWARE=1 uv run python e2e/openxr.py
Use --reference-space standing to exercise room-scale reference space.
"""

import argparse
import contextlib
import json
import multiprocessing as mp
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from itertools import groupby
from pathlib import Path

import psutil

from ito import clock

OUT = Path("e2e/out/xr")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-space", choices=("seated", "standing"), default="seated")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.child:
        # Also bound runtimes that never enter READY or wait forever for a worn headset.
        with (OUT / "pilot.log").open("w") as log:
            child = subprocess.Popen(
                [sys.executable, __file__, "--child", "--reference-space", args.reference_space],
                stdout=log,
                stderr=log,
                start_new_session=sys.platform != "win32",
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
            )
            try:
                result = child.wait(timeout=150)
            except BaseException:
                # A runtime waiting for a headset must not leave the simulator running.
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], check=False)
                else:
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                raise
        if result:
            raise SystemExit((OUT / "pilot.log").read_text())
        print((OUT / "summary.json").read_text())
        print("PASS: XR stereo, live robot, safety, recenter, splat budget, scene stall, reconnect")
        return
    import numpy as np
    import pygame
    from pygame._sdl2 import Window

    from ito.app.__main__ import main as pilot_main
    from ito.driver import pairing
    from ito.render import pose

    for path in OUT.glob("capture-*.png"):
        path.unlink()
    os.environ["XDG_CONFIG_HOME"] = str(OUT / "config")
    with socket.socket() as port:
        port.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{port.getsockname()[1]}"
    driver_log = (OUT / "driver.log").open("w")
    code_file = OUT / "pairing-code"
    code = pairing.rotate(code_file)

    def start_driver():
        env = dict(os.environ)
        gl = "glfw" if sys.platform == "win32" else "osmesa"
        if gl == "osmesa" and Path("/opt/data/lib/osmesa").exists():
            env["LD_LIBRARY_PATH"] = "/opt/data/lib/osmesa:" + env.get("LD_LIBRARY_PATH", "")
        return subprocess.Popen(
            [
                sys.executable,
                "-m",
                "drivers.mujoco.cli",
                "--port",
                address.split(":")[1],
                "--gl",
                gl,
                "--pairing-file",
                str(code_file),
            ],
            env=env,
            stdout=driver_log,
            stderr=driver_log,
        )

    def key(code, down=None):
        for pressed in (True, False) if down is None else (down,):
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN if pressed else pygame.KEYUP, key=code)
            )

    robot = start_driver()
    began = changed = clock.now()
    stage = 0
    position = None
    recentered = 0
    frozen = None
    timer = None
    stall_frames = 0
    observed_hands = set()
    ui_steps = []
    input_stalls = {}
    capture_input_ages = []
    controller = {}  # A simulated right controller: its aim pose and trigger value.
    panel_checks = {}

    def with_controller(actions):
        poll = actions.poll

        def polled(at):
            value = poll(at)
            if controller:
                value.axes["right_trigger"] = controller["trigger"]
                actions.aims["right"] = controller["aim"]
            return value

        actions.poll = polled

    def aim_at(x, y):
        """A right aim, 1.2 m in front of the panel, hitting panel pixel (x, y)."""
        return pose(((x / 768 - 0.5) * 1.05, -0.15 + (0.5 - y / 440) * 1.05 * 440 / 768, 0))

    def drive(app, window, value):
        nonlocal stage, changed, robot, position, recentered, frozen, timer
        nonlocal stall_frames
        now = clock.now()
        assert now - began < 110, (stage, app.state, app.telemetry)
        observed_hands.update(value.hands)
        telemetry = app.telemetry
        # SDL injection follows normal keyboard focus rules, even with a focused HMD.
        if not pygame.key.get_focused():
            Window.from_display_module().focus()
            return
        if ui_steps:
            if now - changed > 0.15:
                x, y, down = ui_steps.pop(0)
                pygame.event.post(
                    pygame.event.Event(
                        pygame.MOUSEMOTION, pos=(x, y), rel=(0, 0), buttons=(0, 0, 0)
                    )
                )
                pygame.event.post(
                    pygame.event.Event(
                        pygame.MOUSEBUTTONDOWN if down else pygame.MOUSEBUTTONUP,
                        pos=(x, y),
                        button=1,
                    )
                )
                changed = now
            return
        if stage == 0 and app.matched_frames > 8 and window.renderer.count > 1000 and value.active:
            capture = window._capture_target

            def checked_capture(target, name):
                result = capture(target, name)
                age = (clock.now() - app.latest_input.timestamp) * 1000
                capture_input_ages.append(age)
                assert age < 200, (name, age)
                return result

            window._capture_target = checked_capture
            position = np.array((telemetry["base_x"], telemetry["base_y"]))
            key(pygame.K_F12)
            stage, changed = "captured", now
        elif stage == "captured" and now - changed > 0.6:
            # Begin driving after the capture has completed.
            key(pygame.K_r)
            key(pygame.K_w, True)
            stage, changed = 1, now
        elif stage == 1 and not input_stalls and telemetry.get("active"):
            ages = []
            until = clock.now() + 0.65
            while clock.now() < until:
                ages.append((clock.now() - app.latest_input.timestamp) * 1000)
                assert ages[-1] < 200, ages[-1]
                assert app.telemetry["active"], app.telemetry
                time.sleep(0.01)
            released = clock.now()
            key(pygame.K_w, False)
            while app.telemetry["left_command"] or app.telemetry["right_command"]:
                assert clock.now() - released < 0.5, app.telemetry
                time.sleep(0.01)
            assert app.telemetry["active"], app.telemetry
            input_stalls["render_stall_input_age_ms_max"] = max(ages)
            input_stalls["render_stall_key_release_ms"] = (clock.now() - released) * 1000
            stopped = clock.now()
            key(pygame.K_e)
            while not app.state.status.e_stop:
                assert clock.now() - stopped < 0.5, app.telemetry
                time.sleep(0.01)
            input_stalls["render_stall_estop_ms"] = (clock.now() - stopped) * 1000
            key(pygame.K_r)
            key(pygame.K_w, True)
            stage, changed = "resume_estop", clock.now()
        elif stage in ("resume_estop", "resume_sampler"):
            assert now - changed < 3, (stage, app.state.status, telemetry)
            if telemetry.get("active") and not app.state.status.e_stop:
                input_stalls[stage + "_ms"] = (now - changed) * 1000
                stage, changed = ("sampler_stall" if stage == "resume_estop" else 1), now
        elif stage == "sampler_stall":
            # Re-arm must survive busy stereo rendering, not just a stale active
            # status from before the e-stop or the first fresh input packet.
            assert telemetry.get("active") and not app.state.status.e_stop, app.state.status
            if now - changed < 0.65:
                return
            poll = window.actions.poll
            entered, release = threading.Event(), threading.Event()

            def blocked(at):
                assert threading.current_thread() is threading.main_thread()
                entered.set()
                assert release.wait(2), "Sampler stall was not released"
                return poll(at)

            window.actions.poll = blocked
            try:
                assert entered.wait(1), "XR sampler waited for rendering"
                stale = app.latest_input.timestamp
                released = None
                while app.telemetry["active"] or clock.now() - stale < 0.3:
                    assert clock.now() - stale < 0.5, app.telemetry
                    if not app.telemetry["active"] and released is None:
                        released = clock.now()
                    time.sleep(0.01)
                assert app.telemetry["left_command"] == app.telemetry["right_command"] == 0
                input_stalls["sampler_stall_deadman_release_ms"] = (
                    (released or clock.now()) - stale
                ) * 1000
            finally:
                window.actions.poll = poll
                release.set()
            # Fresh tracking must not silently re-arm a deadman that timed out.
            time.sleep(0.3)
            assert not app.telemetry["active"], app.telemetry
            key(pygame.K_r)
            stage, changed = "resume_sampler", clock.now()
        elif stage == 1 and now - changed > 2:
            assert (
                np.linalg.norm(np.array((telemetry["base_x"], telemetry["base_y"])) - position)
                > 0.2
            ), (position, telemetry, value.movement, value.active, app.state.status.detail)
            key(pygame.K_e)
            key(pygame.K_w, False)
            stage, changed = 2, now
        elif stage == 2 and app.state.status.e_stop and now - changed > 0.5:
            assert telemetry["left_command"] == telemetry["right_command"] == 0
            assert window.actions.haptic_pulses > 0
            key(pygame.K_F12)
            recentered = window.recenters
            ui_steps.extend([(210, 157, True), (210, 157, False)])
            stage, changed = 3, now
        elif stage == 3 and window.recenters > recentered and now - changed > 0.5:
            # Decrement the actual ImGui budget eight times and apply it.
            ui_steps.extend([(147, 134, down) for _ in range(8) for down in (True, False)])
            ui_steps.extend([(285, 134, True), (285, 134, False)])
            ui_steps.extend([(145, 157, True), (145, 157, False)])
            stage, changed = 4, now
        elif stage == 4 and now - changed > 2:
            assert not app.state.status.e_stop
            assert app.max_splats == 8192
            assert not window.panel_visible, "the panel covers the view with nothing to show"
            # Aimed roughly ahead, at the quad but no control: the trigger is the robot's.
            with_controller(window.actions)
            controller.update(aim=aim_at(700, 400), trigger=0.5)
            stage, changed = "quad aimed", now
        elif stage == "quad aimed" and now - changed > 0.6:
            assert not window.panel_hovered and not window.panel_visible
            assert app.latest_input.axes["right_trigger"] == 0.5, app.latest_input.axes
            panel_checks["trigger_reaches_robot_aimed_ahead"] = True
            # Aim at the panel's title bar, then half-press: it points, not the robot's input.
            controller.update(aim=aim_at(60, 20), trigger=0.0)
            stage, changed = "control aimed", now
        elif stage == "control aimed" and now - changed > 0.3:
            assert window.panel_hovered and window.panel_visible, "pointing did not show it"
            controller["trigger"] = 0.5
            stage, changed = "panel pressed", now
        elif stage == "panel pressed" and now - changed > 0.6:
            assert app.latest_input.axes["right_trigger"] == 0, app.latest_input.axes
            controller["aim"] = pose((0, -0.15, 0), yaw=1.2)
            stage, changed = "panel left", now
        elif stage == "panel left" and now - changed > 0.6:
            # The press that began on the panel stays the panel's until it is released.
            assert app.latest_input.axes["right_trigger"] == 0, app.latest_input.axes
            assert not window.panel_visible
            controller["trigger"] = 0.0
            stage, changed = "trigger released", now
        elif stage == "trigger released" and now - changed > 0.3:
            controller["trigger"] = 0.5
            stage, changed = "trigger away", now
        elif stage == "trigger away" and now - changed > 0.6:
            assert app.latest_input.axes["right_trigger"] == 0.5, app.latest_input.axes
            panel_checks["trigger_reaches_robot_off_panel"] = True
            controller.clear()
            key(pygame.K_F12)
            stage, changed = "captured_scene", now
        elif stage == "captured_scene" and window.capture_number >= 3 and now - changed > 0.6:
            frozen = psutil.Process(app.worker.process.pid)
            frozen.suspend()
            timer = threading.Timer(10, frozen.resume)
            timer.start()
            stage, changed = 5, now
        elif stage == 5:
            if now - changed > 0.7:
                stall_frames += 1
            assert now - changed < 8, "Display waited for reconstruction"
            if now - changed > 2 and stall_frames >= 5:
                if frozen:
                    frozen.resume()
                    frozen = None
                    timer.cancel()
                    robot.kill()
                    robot.wait()
                stage, changed = 6, now
        elif stage == 6 and app.state.status.link == "RECONNECTING" and now - changed > 3:
            assert window.actions.haptic_pulses >= 2
            key(pygame.K_F12)
            robot = start_driver()
            stage, changed = 7, now
        elif stage == 7 and app.connections >= 2 and telemetry and now - changed > 3:
            assert telemetry["left_command"] == telemetry["right_command"] == 0
            assert window.renderer.count > 1000
            key(pygame.K_r)
            stage, changed = 8, now
        elif stage == 8 and now - changed > 0.5:
            key(pygame.K_SPACE)
            key(pygame.K_F12)
            stage, changed = 9, now
        elif stage == 9 and now - changed > 0.5:
            assert telemetry["left_command"] == telemetry["right_command"] == 0
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            stage = 10

    try:
        result = pilot_main(
            [
                address,
                "--code",
                code,
                "--mode",
                "xr",
                "--reference-space",
                args.reference_space,
                "--size",
                "768",
                "440",
                "--max-splats",
                "16384",
                "--capture-dir",
                str(OUT),
                "--metrics",
                str(OUT / "metrics.jsonl"),
            ],
            on_frame=drive,
        )
        assert result == 0 and stage == 10, (result, stage)
    finally:
        if frozen:
            with contextlib.suppress(psutil.NoSuchProcess):
                frozen.resume()
        if timer:
            timer.cancel()
            timer.join()
        if robot.poll() is None:
            robot.terminate()
            robot.wait(timeout=10)
        driver_log.close()
    orphans = mp.active_children()
    for process in orphans:
        process.terminate()
        process.join(timeout=5)
    assert not orphans, "Pilot left a reconstruction process running"
    rows = [json.loads(line) for line in (OUT / "metrics.jsonl").read_text().splitlines()]
    rendered = [r for r in rows if r["rendered"]]
    assert len(rendered) > 30
    frozen_frames = max(
        len(list(group))
        for _, group in groupby(
            (r for r in rendered if r["gaussians"] > 1000), key=lambda r: r["revision"]
        )
    )
    assert frozen_frames >= 5, "No display frames observed with an unchanged scene"
    assert any(r["link"] == "RECONNECTING" and r["rendered"] for r in rows)
    # The panel shows only when it has something the pilot needs.
    assert any(
        not r["panel_visible"] for r in rendered if r["link"] == "CONNECTED" and not r["e_stop"]
    )
    assert all(r["panel_visible"] for r in rendered if r["link"] != "CONNECTED" or r["e_stop"])
    # It also says why the robot won't move: stopped, neutral or faulted.
    assert all(r["panel_visible"] for r in rendered if r["robot_state"] != "active")
    assert any(r["robot_state"] == "stopped" for r in rendered)
    # Both eyes reuse one splat order: never more than one GPU sort per display frame.
    sorts = [b["sorts"] - a["sorts"] for a, b in zip(rows, rows[1:], strict=False)]
    assert max(sorts) == 1 and sum(sorts) > 30, sorts
    for row in rendered:
        assert len(row["eyes"]) == 2
        separation = np.linalg.norm(
            np.array(row["eyes"][0])[:3, 3] - np.array(row["eyes"][1])[:3, 3]
        )
        assert 0.02 < separation < 0.15, separation
    captures = sorted(OUT.glob("capture-*-left.png"))
    assert len(captures) == 5, captures
    first = pygame.surfarray.array3d(pygame.image.load(captures[0])).astype(float)
    right = pygame.surfarray.array3d(
        pygame.image.load(str(captures[0]).replace("left", "right"))
    ).astype(float)
    assert first.std() > 10, "Empty left eye"
    assert right.std() > 10, "Empty right eye"
    assert np.abs(first - right).mean() > 0.2, "Eyes rendered the same view"
    report = {
        **input_stalls,
        "capture_input_age_ms_max": max(capture_input_ages),
        "display_frames": len(rows),
        "stalled_scene_display_frames": stall_frames,
        "unchanged_scene_display_frames": frozen_frames,
        "frame_ms_median": float(np.median([r["frame_ms"] for r in rendered])),
        "sorts_per_frame_max": max(sorts),
        **panel_checks,
        "tracked_hands": sorted(observed_hands),
        "reference_space": args.reference_space,
        "recenters": rows[-1]["recenters"],
        "haptic_pulses": rows[-1]["haptic_pulses"],
        "captures": [str(p) for p in sorted(OUT.glob("capture-*.png"))],
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
