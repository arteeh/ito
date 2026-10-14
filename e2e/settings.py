"""Click the real settings menu, restart Ito, then connect and check the active choices.

ITO_E2E_OUT=/path/to/output xvfb-run uv run python e2e/settings.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pygame
from OpenGL import GL

from ito import clock, diagnostics
from ito.app import connect
from ito.app.__main__ import main as pilot_main
from ito.desktop.settings import load, settings_path

OUT = Path(os.environ.get("ITO_E2E_OUT", "e2e/out/settings"))


def click(position):
    pygame.mouse.set_pos(tuple(map(round, position)))
    for kind in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
        values = (
            {"rel": (0, 0), "buttons": (0, 0, 0)} if kind == pygame.MOUSEMOTION else {"button": 1}
        )
        pygame.event.post(pygame.event.Event(kind, pos=position, **values))


def capture(name):
    width, height = pygame.display.get_window_size()
    GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, 0)
    GL.glReadBuffer(GL.GL_BACK)
    pixels = GL.glReadPixels(0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
    surface = pygame.image.frombytes(pixels, (width, height), "RGB")
    pygame.image.save(pygame.transform.flip(surface, False, True), OUT / f"{name}.png")


def phase(name):
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CONFIG_HOME"] = os.environ["APPDATA"] = str((OUT / "config").resolve())
    os.environ.pop("ITO_DEBUG", None)
    if name == "edit":
        settings_path().unlink(missing_ok=True)
    elif name == "failure":
        settings_path().unlink(missing_ok=True)
        settings_path().mkdir()
    began = clock.now()
    changed = None
    stage = 0
    flip = pygame.display.flip
    actions = [
        ("settings", None),
        ("half", "defaults"),
        ("resolution minus", None),
        ("resolution minus", None),
        ("resolution minus", None),
        ("splats minus", None),
        ("splats plus", None),
        ("diagnostics", None),
        ("back", "lowest"),
    ]

    def frame():
        nonlocal stage, changed
        if clock.now() - began > 50:
            capture("timeout")
            raise AssertionError((name, stage, connect.layout))
        if changed is None:
            changed = clock.now()
        if clock.now() - changed < 0.5:
            flip()
            return
        if name == "edit" and stage < len(actions):
            control, shot = actions[stage]
            if control in connect.layout:
                if shot:
                    capture(shot)
                click(connect.layout[control])
                stage += 1
                changed = clock.now()
        elif name == "edit" and "settings" in connect.layout:
            assert load().model_dump() == dict(
                precision="half", input_resolution=25, max_splats=1_000_000, diagnostic_logging=True
            )
            pygame.event.post(pygame.event.Event(pygame.QUIT))
        elif name == "failure":
            if stage == 0 and "settings" in connect.layout:
                click(connect.layout["settings"])
                stage, changed = 1, clock.now()
            elif stage == 1 and "half" in connect.layout:
                click(connect.layout["half"])
                stage, changed = 2, clock.now()
            elif stage == 2 and "settings error" in connect.layout:
                capture("save-failed")
                assert settings_path().is_dir(), "A failed save changed the settings file"
                settings_path().rmdir()
                click(connect.layout["half"])
                stage, changed = 3, clock.now()
            elif stage == 3 and settings_path().is_file():
                assert load().precision == "half"
                assert "settings error" not in connect.layout
                capture("save-recovered")
                click(connect.layout["back"])
                stage, changed = 4, clock.now()
            elif stage == 4 and "settings" in connect.layout:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
        elif name == "restart":
            if stage < 2:
                assert load().precision == "half" and load().input_resolution == 25
                assert diagnostics.current().enabled
            if stage == 0 and "settings" in connect.layout:
                click(connect.layout["settings"])
                stage, changed = 1, clock.now()
            elif stage == 1 and "reset" in connect.layout:
                capture("persisted")
                click(connect.layout["reset"])
                stage, changed = 2, clock.now()
            elif stage == 2 and "back" in connect.layout:
                assert load().precision == "full" and load().input_resolution == 100
                assert load().max_splats == 1_000_000 and not diagnostics.current().enabled
                capture("reset")
                click(connect.layout["back"])
                stage, changed = 3, clock.now()
            elif stage == 3 and "simulated" in connect.layout:
                click(connect.layout["simulated"])
                stage, changed = 4, clock.now()
        flip()

    connected = False

    def drive(pilot, window, value):
        nonlocal connected
        if pilot.state.status.link == "CONNECTED" and window.renderer.count > 1000:
            assert pilot.settings.max_splats == 1_000_000
            assert "diagnostics" not in window.overlay.layout
            capture("runtime")
            connected = True
            pygame.event.post(pygame.event.Event(pygame.QUIT))

    pygame.display.flip = frame
    try:
        assert pilot_main(["--size", "1000", "800", "--fps", "30"], on_frame=drive) == 0
    finally:
        pygame.display.flip = flip
    assert stage == (len(actions) if name == "edit" else 4)
    assert name != "restart" or connected
    print(name, json.loads(settings_path().read_text()))


def main():
    if len(sys.argv) > 1:
        phase(sys.argv[1])
        return
    for name in ("edit", "restart", "failure"):
        subprocess.run([sys.executable, "-m", "e2e.settings", name], check=True, timeout=90)
    print("PASS: settings buttons, disk persistence, restart, reset and real simulation connection")


if __name__ == "__main__":
    main()
