"""Start ito with no address, try the simulated robot, then pair with a robot on the network.

The robot is ito-driver-mujoco listening on 0.0.0.0, reached through this machine's network
address. Pairing runs unpaired, wrong-code, correct-code, remembered-code and rotated-code
connections through the connect screen, the way a pilot would.

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

from ito.app import connect
from ito.app.__main__ import main as pilot_main
from ito.driver import pairing
from ito.link.pairing import display

OUT = Path("e2e/out/connect")
DISCONNECT = (218, 173)  # The pilot overlay's button row.


def network_address():
    """This machine's address on its network: what a pilot elsewhere would type."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("192.0.2.1", 9))  # Picks the outgoing interface; sends nothing.
        return probe.getsockname()[0]


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


def type_text(text):
    key(pygame.K_END)
    for _ in range(24):
        key(pygame.K_BACKSPACE)
    pygame.event.post(pygame.event.Event(pygame.TEXTINPUT, text=text))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = OUT / "config"
    os.environ["XDG_CONFIG_HOME"] = os.environ["APPDATA"] = str(config)
    recent = config / "ito/recent.json"
    recent.parent.mkdir(parents=True, exist_ok=True)
    recent.write_text(json.dumps([{"address": "127.0.0.1:9", "name": "Unreachable"}]))
    code_file = OUT / "pairing-code"
    code_file.unlink(missing_ok=True)
    gl = {"LD_LIBRARY_PATH": "/opt/data/lib/osmesa"} if sys.platform == "linux" else {}
    with socket.socket() as probe:
        probe.bind(("0.0.0.0", 0))
        port = probe.getsockname()[1]
    address = f"{network_address()}:{port}"
    driver = [sys.executable, "-m", "drivers.mujoco.cli", "--pairing-file", str(code_file)]
    log = (OUT / "driver.log").open("w")
    robot = subprocess.Popen(
        driver + ["--port", str(port)], env=os.environ | gl, stdout=log, stderr=log
    )
    stage, changed, began = "simulated", time.monotonic(), time.monotonic()
    live = {}
    seen = []  # Every Pilot the app made, to read why the robot refused it.
    sims = []
    codes = {}
    pair = None
    flip = pygame.display.flip

    def at(name):
        return connect.layout.get(name)

    def go(name):
        nonlocal stage, changed
        stage, changed = name, time.monotonic()
        connect.layout.clear()  # Only a redrawn connect screen counts as being back there.

    def frame():
        nonlocal pair
        now = time.monotonic()
        waited = now - changed
        assert now - began < 240, f"stuck at stage {stage}"
        pilot, window = live.get("pilot"), live.get("window")
        connected = pilot and pilot.state.status.link == "CONNECTED"
        scene = connected and window.renderer.count > 1000
        if stage == "simulated" and waited > 1 and at("simulated"):
            assert screenshot("connect").std() > 5, "Blank connect screen"
            click(at("simulated"))
            go("simulated piloting")
        elif stage == "simulated piloting" and scene and waited > 2:
            assert pilot.address != "127.0.0.1:9" and pilot.refusal is None
            sims.extend(
                p
                for p in psutil.Process().children()
                if p.pid != robot.pid and "drivers.mujoco" in str(p.cmdline())
            )
            assert sims, "No simulated robot process"
            assert screenshot("simulated").std() > 15, "Simulated room not reconstructed"
            click(DISCONNECT)
            live.clear()
            go("back")
        elif stage == "back" and waited > 1.5 and at("address"):
            assert not any(p.is_running() and p.status() != psutil.STATUS_ZOMBIE for p in sims)
            listening = (OUT / "driver.log").read_text()
            assert f"listening at http://0.0.0.0:{port}" in listening, listening
            codes["first"] = pairing.read(code_file)
            shown = subprocess.run(driver + ["--show-code"], check=True, capture_output=True)
            assert display(codes["first"]) in shown.stdout.decode()
            if os.name != "nt":
                assert code_file.stat().st_mode & 0o077 == 0
            assert f"Pairing code: {display(codes['first'])}" in listening, listening
            screenshot("back")
            click(at("address"))
            go("typing address")
        elif stage == "typing address" and waited > 0.3:
            type_text(address)
            go("unpaired")
        elif stage == "unpaired" and waited > 0.3:
            key(pygame.K_RETURN)
            go("unpaired refused")
        elif stage == "unpaired refused" and at("code") and waited > 0.5:
            assert seen[-1].code is None and "needs its pairing code" in seen[-1].refusal
            screenshot("unpaired")
            type_text("000 000" if codes["first"] != "000000" else "111 111")
            go("wrong code")
        elif stage == "wrong code" and waited > 0.3:
            key(pygame.K_RETURN)
            go("wrong code refused")
        elif stage == "wrong code refused" and len(seen) == 3 and at("code"):
            pair = at("pair")
            assert seen[-1].refusal == "Wrong pairing code", seen[-1].refusal
            assert not any(e.get("code") for e in json.loads(recent.read_text()))
            screenshot("wrong-code")
            type_text(codes["first"][:3] + "-" + codes["first"][3:])
            go("correct code")
        elif stage == "correct code" and waited > 0.3:
            click(pair)
            go("paired")
        elif stage == "paired" and scene:
            assert pilot.address == address and pilot.code == codes["first"], pilot.address
            saved = json.loads(recent.read_text())
            if os.name != "nt":
                assert recent.stat().st_mode & 0o077 == 0
            assert saved[0] == {"address": address, "name": saved[0]["name"]} | {
                "code": codes["first"]
            }, saved
            screenshot("paired")
            click(DISCONNECT)
            live.clear()
            go("remembered")
        elif stage == "remembered" and waited > 1.5 and at("recent 0"):
            assert not at("code")
            click(at("recent 0"))
            go("remembered piloting")
        elif stage == "remembered piloting" and scene:
            assert pilot.code == codes["first"] and len(seen) == 5, "remembered code not used"
            click(DISCONNECT)
            live.clear()
            subprocess.run(
                driver + ["--rotate-code"], env=os.environ | gl, check=True, capture_output=True
            )
            codes["rotated"] = pairing.read(code_file)
            assert codes["rotated"] != codes["first"]
            go("rotated")
        elif stage == "rotated" and waited > 1.5 and at("recent 0"):
            click(at("recent 0"))
            go("rotated refused")
        elif stage == "rotated refused" and waited > 0.5 and len(seen) == 6 and at("code"):
            assert seen[-1].code == codes["first"] and seen[-1].refusal == "Wrong pairing code"
            assert json.loads(recent.read_text())[0]["code"] is None, "refused code kept"
            screenshot("rotated")
            type_text(codes["rotated"])
            go("rotated code")
        elif stage == "rotated code" and waited > 0.3:
            key(pygame.K_RETURN)
            go("rotated piloting")
        elif stage == "rotated piloting" and scene:
            assert pilot.code == codes["rotated"]
            screenshot("rotated-paired")
            key(pygame.K_ESCAPE)
            go("escape piloting")
        elif stage == "escape piloting" and waited > 0.5:
            assert connected
            click(DISCONNECT)
            live.clear()
            go("escape connect")
        elif stage == "escape connect" and waited > 0.5 and at("address"):
            key(pygame.K_ESCAPE)
            go("escape idle")
        elif stage == "escape idle" and waited > 0.5:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            go("done")
        flip()

    def on_frame(pilot, window, value):
        if not seen or seen[-1] is not pilot:
            seen.append(pilot)
        if stage.endswith("piloting") or stage == "paired":
            live.update(pilot=pilot, window=window)

    pygame.display.flip = frame
    try:
        assert pilot_main(["--metrics", str(OUT / "metrics.jsonl")], on_frame=on_frame) == 0
    finally:
        pygame.display.flip = flip
        robot.terminate()
        robot.wait(timeout=8)
        log.close()
    assert stage == "done", stage
    saved = json.loads(recent.read_text())
    assert saved[0]["address"] == address and saved[0]["code"] == codes["rotated"], saved
    assert [entry["address"] for entry in saved] == [address, "127.0.0.1:9"], saved
    assert "Wrong pairing code" in (OUT / "driver.log").read_text()
    print(
        "PASS: connect screen, simulated robot, disconnect, and unpaired, wrong-code, "
        "correct-code, remembered-code and rotated-code pairing over the network"
    )


if __name__ == "__main__":
    main()
