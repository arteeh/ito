"""The bundled simulated robot, piloted the way 'Try simulated robot' starts it.

It has no microphone or speaker, and mouse-look capture and release must leave every
ImGui control clickable. The Linux viewer check needs xdotool.

DISPLAY=:97 LIBGL_ALWAYS_SOFTWARE=1 uv run --with pillow python e2e/simulated.py
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import pygame

from ito.app.__main__ import main as pilot_main

OUT = Path("e2e/out/simulated")
SCENE = (560, 420)  # Open floor in the 800x600 window, away from the overlay panel.


def click(position):
    """A pilot's click holds the button for several frames, past mouse-look capture."""
    pygame.event.post(
        pygame.event.Event(pygame.MOUSEMOTION, pos=position, rel=(0, 0), buttons=(0, 0, 0))
    )
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=position, button=1))
    release = pygame.event.Event(pygame.MOUSEBUTTONUP, pos=position, button=1)
    threading.Timer(0.12, pygame.event.post, (release,)).start()


def key(code):
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=code, mod=0, unicode=""))
    pygame.event.post(pygame.event.Event(pygame.KEYUP, key=code, mod=0))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CONFIG_HOME"] = os.environ["APPDATA"] = str(OUT / "config")
    if sys.platform == "linux":
        os.environ.setdefault("MUJOCO_GL", "osmesa")
    stage, changed, began = "connecting", time.monotonic(), time.monotonic()
    report = {}

    def go(name):
        nonlocal stage, changed
        stage, changed = name, time.monotonic()

    def drive(app, window, value):
        now = time.monotonic()
        waited = now - changed
        assert now - began < 120, (stage, app.state.status)
        status = app.state.status
        layout = window.overlay.layout
        if stage == "connecting" and status.link == "CONNECTED" and window.renderer.count > 1000:
            # On the pilot's own PC the simulated robot has no microphone or speaker:
            # the pilot is told so, gets no mute toggles, and no audio device opens.
            assert not (status.robot_microphone or status.robot_speaker), status
            assert not {"mute_mic", "mute_speaker"} & set(layout), layout
            assert app.audio.input_status == app.audio.output_status == "off", app.audio.status
            assert not app.audio.streams and "audio" not in app.telemetry, app.telemetry
            report["yaw_before_look"] = window.input.yaw
            click(SCENE)
            go("captured")
        elif stage == "captured" and waited > 0.5:
            assert window.input.captured, "Clicking the scene did not start mouse-look"
            pygame.event.post(
                pygame.event.Event(pygame.MOUSEMOTION, pos=SCENE, rel=(-120, 0), buttons=(0, 0, 0))
            )
            go("looked")
        elif stage == "looked" and waited > 0.5:
            assert window.input.yaw > report["yaw_before_look"] + 0.2, window.input.yaw
            key(pygame.K_ESCAPE)
            go("released")
        elif stage == "released" and waited > 0.5:
            assert not window.input.captured, "Escape did not release mouse-look"
            click(layout["e_stop"])
            go("e-stop button")
        elif stage == "e-stop button" and (status.e_stop or waited > 3):
            assert status.e_stop, "E-stop button did nothing after mouse-look"
            assert not window.input.captured, "Clicking a button started mouse-look"
            report["e_stop_after_mouse_look_s"] = waited
            click(layout["resume"])
            go("resume button")
        elif stage == "resume button" and (not status.e_stop or waited > 3):
            assert not status.e_stop and not window.input.captured, status
            click(layout["show_simulation"])
            go("viewer opening")
        elif stage == "viewer opening":
            sim = window.overlay.simulation
            health = sim.viewer_state.with_suffix(".viewer")
            assert not sim.viewer_error, sim.viewer_error
            if health.exists() and json.loads(health.read_text())["frames"] > 5:
                report["viewer_started"] = json.loads(health.read_text())
                result = subprocess.run(
                    ["xdotool", "search", "--name", "MuJoCo"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                report["viewer_window"] = result.stdout.splitlines()[-1]
                subprocess.run(
                    [
                        "xdotool",
                        "windowsize",
                        report["viewer_window"],
                        "780",
                        "600",
                        "windowmove",
                        report["viewer_window"],
                        "810",
                        "0",
                    ],
                    check=True,
                )
                from pygame._sdl2 import Window

                Window.from_display_module().position = (0, 0)
                # Resume after focus moves to the new window and drive the actual robot.
                pygame.event.post(pygame.event.Event(pygame.WINDOWFOCUSGAINED))
                key(pygame.K_r)
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w))
                report["base_before"] = app.telemetry["base_x"]
                go("viewer tracking")
        elif stage == "viewer tracking" and waited > 2:
            sim = window.overlay.simulation
            current = json.loads(sim.viewer_state.with_suffix(".viewer").read_text())
            assert current["frames"] > report["viewer_started"]["frames"] + 10, current
            assert current["simulation_time"] > report["viewer_started"]["simulation_time"] + 1
            assert abs(app.telemetry["simulation_time"] - current["simulation_time"]) < 0.5
            assert abs(app.telemetry["base_x"] - report["base_before"]) > 0.1, app.telemetry
            report["viewer_tracking"] = current
            from PIL import ImageGrab

            ImageGrab.grab().save(OUT / "viewer.png")
            pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame.K_w))
            subprocess.run(["xdotool", "windowclose", report["viewer_window"]], check=True)
            report["frames_before_viewer_close"] = app.matched_frames
            go("viewer closed")
        elif stage == "viewer closed" and waited > 1:
            assert not window.overlay.simulation.visible
            assert status.link == "CONNECTED"
            assert app.matched_frames > report["frames_before_viewer_close"]
            # Reopening and toggling off must also leave the robot connected.
            click(layout["show_simulation"])
            go("viewer reopened")
        elif stage == "viewer reopened" and waited > 2:
            assert window.overlay.simulation.visible
            click(layout["show_simulation"])
            go("viewer hidden")
        elif stage == "viewer hidden" and waited > 1:
            assert not window.overlay.simulation.visible
            assert status.link == "CONNECTED"
            click(layout["show_simulation"])
            go("close with viewer")
        elif stage == "close with viewer" and waited > 2:
            sim = window.overlay.simulation
            assert sim.visible
            log = sim.log.read_text()
            assert "Pairing code:" not in log and sim.code not in log
            report["viewer_pid"] = sim.viewer.pid
            report["driver_pid"] = sim.process.pid
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            go("done")

    assert pilot_main(["--sim", "--size", "800", "600", "--fps", "30"], on_frame=drive) == 0
    assert stage == "done", stage
    assert not any(psutil.pid_exists(report[key]) for key in ("viewer_pid", "driver_pid"))
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(
        "PASS: simulated robot without audio; buttons work after mouse-look; "
        "separate viewer tracks and closes"
    )


if __name__ == "__main__":
    main()
