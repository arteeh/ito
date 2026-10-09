"""Tab grabs and releases mouse-look; toggling must never move the pilot's view.

uv run python e2e/tab_look.py   (needs Xvfb, xdotool and Xvnc from tigervnc-standalone-server)

Real X pointers, not posted SDL events, so SDL's own relative mode is what gets tested:
- Xvfb with xdotool, a relative mouse: between toggles the pointer moves while captured (the
  view turns exactly by the delta: no acceleration, nothing dropped or doubled) and while
  released (the view stays). Pitch clamps short of straight up/down; the view never rolls.
- Xvnc, whose pointer is absolute like any remote desktop: SDL reports the first motion after
  a grab from a stale origin, which turned the view ~88 degrees with the pointer at x=614.
"""

import argparse
import json
import math
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import numpy as np
from sample_scene import write_scene

OUTPUT = Path("e2e/out/tab-look")


def xdotool(*args):
    return subprocess.run(
        ["xdotool", *map(str, args)], capture_output=True, text=True, check=True
    ).stdout


class VncPointer:
    """The RFB client side of a remote desktop: absolute pointer positions only."""

    def __init__(self, port):
        self.socket = connection = socket.create_connection(("127.0.0.1", port), timeout=5)
        connection.recv(12)
        connection.sendall(b"RFB 003.008\n")
        types = connection.recv(connection.recv(1)[0])
        assert 1 in types, f"Xvnc must run with -SecurityTypes None: {types}"
        connection.sendall(b"\x01")
        assert struct.unpack(">I", connection.recv(4))[0] == 0, "VNC security failed"
        connection.sendall(b"\x01")  # Shared session.
        server = connection.recv(24)
        connection.recv(struct.unpack(">I", server[20:24])[0])

    def move(self, x, y):
        self.socket.sendall(struct.pack(">BBHH", 5, 0, x, y))


def phase(pointer: str) -> dict:
    from ito.desktop import DesktopWindow
    from ito.desktop.__main__ import FileScene

    scene = OUTPUT / "scene.ply"
    write_scene(scene)
    samples = []
    report = {"pointer": pointer, "steps": []}
    failure = []
    done = threading.Event()

    def sample(pilot):
        samples.append(
            {
                "yaw": window.input.yaw,
                "pitch": window.input.pitch,
                "captured": window.input.captured,
                # Roll-free: the view's right axis stays horizontal.
                "roll": float(pilot.head[1, 0]),
            }
        )

    def settle():
        time.sleep(0.35)  # Dozens of 90 Hz input samples after X delivered everything.
        return window.input.yaw, window.input.pitch

    def step(name, expect=None):
        view = settle()
        report["steps"].append({"step": name, "yaw": view[0], "pitch": view[1]})
        if expect is not None:
            turned = math.degrees(max(abs(view[0] - expect[0]), abs(view[1] - expect[1])))
            assert np.allclose(view, expect, atol=1e-9), f"{name}: view moved {turned:.1f} deg"
        return view

    def toggle(name, view, captured):
        xdotool("key", "Tab")
        view = step(name, view)
        assert window.input.captured == captured, f"{name}: Tab did not toggle mouse-look"
        return view

    def relative_mouse(window_id):
        sensitivity = window.input.sensitivity
        # Start far from the window center so a grab-time warp would show.
        xdotool("mousemove", "--window", window_id, 40, 40)
        view = step("start", (0.0, 0.0))
        for cycle in range(4):
            view = toggle(f"grab {cycle}", view, True)
            dx, dy = (60, -25) if cycle % 2 == 0 else (-35, 15)
            # The first report after a grab is the pointer's baseline, never a turn.
            xdotool("mousemove_relative", "--", 1, 1)
            view = step(f"baseline {cycle}", view)
            for _ in range(3):
                xdotool("mousemove_relative", "--", dx, dy)
                # One-to-one: no pointer acceleration, no dropped or doubled motion.
                view = step(
                    f"look {cycle}", (view[0] - dx * sensitivity, view[1] - dy * sensitivity)
                )
            view = toggle(f"release {cycle}", view, False)
            # A free pointer moves anywhere without turning the view.
            xdotool("mousemove", "--window", window_id, 700 - 150 * cycle, 90 + 120 * cycle)
            view = step(f"free move {cycle}", view)
        # Look hard up and down: pitch clamps and never flips over; yaw is untouched.
        view = toggle("grab for pitch", view, True)
        for _ in range(8):
            xdotool("mousemove_relative", "--", 0, -400)
        up = step("look up")
        assert math.isclose(up[1], math.pi * 0.49, abs_tol=1e-9) and up[0] == view[0], up
        for _ in range(16):
            xdotool("mousemove_relative", "--", 0, 400)
        down = step("look down")
        assert math.isclose(down[1], -math.pi * 0.49, abs_tol=1e-9) and down[0] == view[0]
        down = toggle("release after pitch", down, False)
        xdotool("key", "Escape")
        step("escape", down)

    def absolute_pointer(port):
        vnc = VncPointer(port)
        vnc.move(40, 40)
        view = step("start", (0.0, 0.0))
        for cycle, (x, y) in enumerate(((614, 300), (180, 520), (840, 120), (450, 650))):
            view = toggle(f"grab {cycle}", view, True)
            vnc.move(x, y)
            view = step(f"first motion {cycle} at {x},{y}", view)
            view = toggle(f"release {cycle}", view, False)
            vnc.move(x // 2, y // 2)
            view = step(f"free move {cycle}", view)

    def drive():
        try:
            time.sleep(1.0)
            window_id = xdotool("search", "--sync", "--pid", os.getpid()).split()[-1]
            xdotool("windowmove", window_id, 0, 0)
            xdotool("windowfocus", "--sync", window_id)
            if pointer == "relative":
                relative_mouse(window_id)
            else:
                absolute_pointer(int(pointer.split(":")[1]))
        except BaseException as exc:
            failure.append(exc)
        finally:
            done.set()

    def stop(pilot):
        if done.is_set():
            import pygame

            pygame.event.post(pygame.event.Event(pygame.QUIT))

    with DesktopWindow((960, 720), fps=60) as window:
        threading.Thread(target=drive, daemon=True).start()
        window.run(FileScene(scene), on_input=stop, on_sample=sample)

    if failure:
        raise failure[0]
    assert any(row["captured"] for row in samples), "mouse-look never captured"
    assert max(abs(row["roll"]) for row in samples) < 1e-6, "view rolled"
    # No spike: one 90 Hz sample turns at most one xdotool step (60 px of yaw).
    turns = [
        abs(math.remainder(b["yaw"] - a["yaw"], math.tau))
        for a, b in zip(samples, samples[1:], strict=False)
    ]
    report["largest_yaw_step_deg"] = math.degrees(max(turns))
    assert max(turns) <= 60 * window.input.sensitivity + 1e-9, report["largest_yaw_step_deg"]
    return report


def free_display() -> int:
    return next(n for n in range(90, 200) if not Path(f"/tmp/.X11-unix/X{n}").exists())


def run_phase(server: list[str], display: int, pointer: str) -> dict:
    process = subprocess.Popen(server, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        socket_path = Path(f"/tmp/.X11-unix/X{display}")
        deadline = time.monotonic() + 10
        while not socket_path.exists():
            assert process.poll() is None, f"{server[0]} exited with {process.returncode}"
            assert time.monotonic() < deadline, f"{server[0]} did not start"
            time.sleep(0.1)
        env = dict(
            os.environ, DISPLAY=f":{display}", LIBGL_ALWAYS_SOFTWARE="1", SDL_VIDEODRIVER="x11"
        )
        result = subprocess.run(
            [sys.executable, __file__, "--phase", pointer],
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if result.returncode:
            raise SystemExit(f"{pointer} pointer phase failed:\n{result.stderr[-3000:]}")
        return json.loads(result.stdout.splitlines()[-1])
    finally:
        process.terminate()
        process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", help=argparse.SUPPRESS)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.phase:
        print(json.dumps(phase(args.phase)))
        return
    display = free_display()
    relative = run_phase(
        ["Xvfb", f":{display}", "-screen", "0", "1600x1000x24"], display, "relative"
    )
    display = free_display()
    port = 5900 + display
    absolute = run_phase(
        ["Xvnc", f":{display}", "-rfbport", str(port), "-SecurityTypes", "None"]
        + ["-geometry", "1600x1000", "-depth", "24"],
        display,
        f"absolute:{port}",
    )
    reports = [relative, absolute]
    (OUTPUT / "report.json").write_text(json.dumps(reports, indent=2))
    toggles = sum(
        step["step"].startswith(("grab", "release"))
        for report in reports
        for step in report["steps"]
    )
    print(
        f"PASS: {toggles} Tab toggles with real relative (Xvfb) and absolute (Xvnc) pointer "
        "motion between them; no toggle moved the view, mouse-look is 1:1, pitch clamps, "
        "no roll"
    )


if __name__ == "__main__":
    main()
