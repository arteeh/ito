"""The bundled simulated robot, piloted the way 'Try simulated robot' starts it.

Mouse-look capture and release must leave every ImGui control clickable.

DISPLAY=:97 LIBGL_ALWAYS_SOFTWARE=1 uv run python e2e/simulated.py
"""

import json
import os
import sys
import threading
import time
from pathlib import Path

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
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            go("done")

    assert pilot_main(["--sim", "--size", "800", "600", "--fps", "30"], on_frame=drive) == 0
    assert stage == "done", stage
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("PASS: simulated robot; buttons work after mouse-look")


if __name__ == "__main__":
    main()
