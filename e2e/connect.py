"""Start ito with no address, click "Try simulated robot", disconnect, type a real address.

DISPLAY=:97 LIBGL_ALWAYS_SOFTWARE=1 uv run python e2e/connect.py
"""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pygame
from OpenGL import GL

from ito.app.__main__ import main as pilot_main

OUT = Path("e2e/out/connect")
# The 1280x720 connect panel with one recent robot, and the pilot overlay's button row.
ADDRESS_FIELD, SIMULATED_ROBOT, DISCONNECT = (600, 340), (640, 414), (218, 173)


def click(position):
    for kind in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
        options = {"rel": (0, 0), "buttons": (0, 0, 0)} if kind == pygame.MOUSEMOTION else {}
        options |= {} if kind == pygame.MOUSEMOTION else {"button": 1}
        pygame.event.post(pygame.event.Event(kind, pos=position, **options))


def key(code):
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=code, mod=0, unicode=""))
    pygame.event.post(pygame.event.Event(pygame.KEYUP, key=code, mod=0))


def screenshot(name):
    width, height = pygame.display.get_window_size()
    GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, 0)
    GL.glReadBuffer(GL.GL_BACK)
    GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
    pixels = GL.glReadPixels(0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
    image = pygame.transform.flip(pygame.image.frombytes(pixels, (width, height), "RGB"), 0, 1)
    pygame.image.save(image, OUT / f"{name}.png")
    return pygame.surfarray.array3d(image)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = OUT / "config"
    os.environ["XDG_CONFIG_HOME"] = os.environ["APPDATA"] = str(config)
    recent = config / "ito/recent.json"
    recent.parent.mkdir(parents=True, exist_ok=True)
    recent.write_text(json.dumps([{"address": "127.0.0.1:9", "name": "Unreachable"}]))
    gl = {"LD_LIBRARY_PATH": "/opt/data/lib/osmesa"} if sys.platform == "linux" else {}
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{probe.getsockname()[1]}"
    log = (OUT / "driver.log").open("w")
    robot = subprocess.Popen(
        [sys.executable, "-m", "drivers.mujoco.cli", "--port", address.split(":")[1]],
        env=os.environ | gl,
        stdout=log,
        stderr=log,
    )
    stage, changed, began = 0, time.monotonic(), time.monotonic()
    live = {}
    sims = []
    flip = pygame.display.flip

    def frame():
        nonlocal stage, changed
        now = time.monotonic()
        assert now - began < 90, f"stuck at stage {stage}"
        pilot, window = live.get("pilot"), live.get("window")
        connected = pilot and pilot.state.status.link == "CONNECTED"
        if stage == 0 and now - changed > 1:
            assert screenshot("connect").std() > 5, "Blank connect screen"
            click(SIMULATED_ROBOT)
            stage, changed = 1, now
        elif stage == 1 and connected and window.renderer.count > 1000 and now - changed > 2:
            assert pilot.address != "127.0.0.1:9"
            sims.extend(
                p
                for p in psutil.Process().children()
                if p.pid != robot.pid and "drivers.mujoco" in str(p.cmdline())
            )
            assert sims, "No simulated robot process"
            assert screenshot("simulated").std() > 15, "Simulated room not reconstructed"
            click(DISCONNECT)
            live.clear()
            stage, changed = 2, now
        elif stage == 2 and now - changed > 1.5:
            assert live.get("pilot") is None, "Disconnect did not return to the connect screen"
            assert not any(p.is_running() and p.status() != psutil.STATUS_ZOMBIE for p in sims)
            screenshot("back")
            click(ADDRESS_FIELD)
            stage, changed = 3, now
        elif stage == 3 and now - changed > 0.3:
            key(pygame.K_END)
            for _ in range(20):
                key(pygame.K_BACKSPACE)
            pygame.event.post(pygame.event.Event(pygame.TEXTINPUT, text=address))
            stage, changed = 4, now
        elif stage == 4 and now - changed > 0.3:
            key(pygame.K_RETURN)
            stage, changed = 5, now
        elif stage == 5 and connected and window.renderer.count > 1000:
            assert pilot.address == address, pilot.address
            screenshot("typed")
            key(pygame.K_ESCAPE)
            stage = 6
        flip()

    def on_frame(pilot, window, value):
        if stage in (1, 5):
            live.update(pilot=pilot, window=window)

    pygame.display.flip = frame
    try:
        assert pilot_main(["--metrics", str(OUT / "metrics.jsonl")], on_frame=on_frame) == 0
    finally:
        pygame.display.flip = flip
        robot.terminate()
        robot.wait(timeout=8)
        log.close()
    assert stage == 6, stage
    saved = json.loads(recent.read_text())
    assert saved[0]["address"] == address and saved[0]["name"], saved
    assert [entry["address"] for entry in saved] == [address, "127.0.0.1:9"], saved
    print("PASS: connect screen, simulated robot, disconnect, typed address, recent robots")


if __name__ == "__main__":
    main()
