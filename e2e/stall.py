"""Real pilot CLI + MuJoCo: input stalls release the deadman, and only a short one re-arms.

The release reaches the robot at once, a blocked link loop counts as a stall too, and the
app closed mid-drive is gone within two seconds with the robot neutral.

LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a uv run python e2e/stall.py

Drawing has its own thread (e2e/app.py holds the display for 650 ms with input still live),
so the stall is injected where a GC pause or a GIL-holding save would hit: the input sampler.
"""

import json
import logging
import multiprocessing
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pygame
from pygame._sdl2 import Window

from ito import clock
from ito.app.__main__ import main as pilot_main
from ito.app.pilot import SHORT_STALL
from ito.desktop.dispatch import DisplayDispatch
from ito.driver import pairing

OUT = Path("e2e/out/stall")
stall_for = 0.0  # Seconds the next input sample waits, as if the sampler thread froze.
stall_ended = None  # When the last injected stall let sampling continue.
sample = DisplayDispatch.sample


def stalling_sample(self, dt, on_sample):
    global stall_for, stall_ended
    if stall_for:
        pause, stall_for, stall_ended = stall_for, 0.0, None
        time.sleep(pause)
        stall_ended = clock.now()
    sample(self, dt, on_sample)


DisplayDispatch.sample = stalling_sample


class Lines(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def key(code, down=None):
    for kind in (pygame.KEYDOWN, pygame.KEYUP) if down is None else (down,):
        pygame.event.post(pygame.event.Event(kind, key=code))


def main():
    global stall_for
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CONFIG_HOME"] = os.environ["APPDATA"] = str(OUT / "config")
    with socket.socket() as port:
        port.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{port.getsockname()[1]}"
    code_file = OUT / "pairing-code"
    code = pairing.rotate(code_file)
    driver_log = (OUT / "driver.log").open("w")
    robot = subprocess.Popen(
        [sys.executable, "-m", "drivers.mujoco.cli", "--port", address.split(":")[1]]
        + ["--pairing-file", str(code_file)],
        env=os.environ | {"MUJOCO_GL": "osmesa", "LP_NUM_THREADS": "2"},
        stdout=driver_log,
        stderr=driver_log,
    )
    lines = Lines()
    logging.getLogger("ito.app.pilot").addHandler(lines)
    began = clock.now()
    stage, changed = "connecting", began
    seen = []  # (robot state, pilot armed, left command) from a stall until it settles.
    results = {}
    closed_at = None

    def drive(app, window, value):
        nonlocal stage, changed
        now = clock.now()
        assert now - began < 120, (stage, app.state.status, app.telemetry)
        t = app.telemetry
        status = app.state.status

        def moving():
            return status.armed and status.robot_state == "active" and t["left_command"] > 0

        def still():
            return not status.armed and t["left_command"] == t["right_command"] == 0

        def settled(condition, limit=5):
            """Driver status lags input by a few hundred ms on a loaded software GPU."""
            assert now - changed < limit, (stage, status, t)
            return condition()

        def stalling(seconds, name):
            nonlocal stage, changed
            global stall_for
            seen.clear()
            stall_for = seconds
            stage, changed = name, now

        if stage == "connecting" and app.matched_frames > 12 and t:
            Window.from_display_module().focus()
            key(pygame.K_r)
            key(pygame.K_w, pygame.KEYDOWN)
            stage, changed = "walking", now
        elif stage == "walking" and settled(moving) and now - changed > 1:
            stalling(0.4, "short")
        elif stage in ("short", "long"):
            # The sampler is frozen; the drawing thread watches the deadman drop and return.
            seen.append((status.robot_state, status.armed, t["left_command"]))
            if stall_ended is not None and (stage == "long" or moving()):
                assert any(not armed for _, armed, _ in seen), seen
                assert any(state != "active" for state, _, _ in seen), seen
                results[f"{stage}_stall_ms"] = round((stall_ended - changed) * 1000)
                results[f"{stage}_robot_states"] = sorted({state for state, _, _ in seen})
                if stage == "short":
                    assert any("re-armed after a" in line for line in lines.lines), lines.lines
                    results["short_rearmed_after_stall_ms"] = round((now - stall_ended) * 1000)
                stage, changed = f"{stage}_after", now
            else:
                settled(lambda: True, 8)
        elif stage == "short_after" and now - changed > 1.5:
            assert moving(), (status, t)
            stalling(2.0, "long")
        elif stage == "long_after" and now - changed > 2:
            # W is still held and input is live, yet the robot waits for an explicit resume.
            assert value.movement[2] > 0 and value.active, value
            assert still() and not app.stalled, (status, t)
            assert sum("input stalled over" in line for line in lines.lines) == 1, lines.lines
            key(pygame.K_r)
            stage, changed = "long_resumed", now
        elif stage == "long_resumed" and settled(moving) and now - changed > 1:
            # Input stays live while the link's own loop is blocked longer than a short stall,
            # as by a synchronous decoder join: the loop never sees the input go stale.
            app.loop.call_soon_threadsafe(time.sleep, 1.5)
            stage, changed = "link", now
        elif stage == "link" and now - changed > 3.5:
            assert value.movement[2] > 0 and value.active, value
            assert still(), (status, t)
            assert sum("input stalled over" in line for line in lines.lines) == 2, lines.lines
            key(pygame.K_r)
            stage, changed = "link_resumed", now
        elif stage == "link_resumed" and settled(moving):
            stage, changed = "focus", now
        elif stage == "focus" and now - changed > 1:
            pygame.event.post(pygame.event.Event(pygame.WINDOWFOCUSLOST))
            stage, changed = "focus_lost", now
        elif stage == "focus_lost" and now - changed > 0.3:
            pygame.event.post(pygame.event.Event(pygame.WINDOWFOCUSGAINED))
            key(pygame.K_w, pygame.KEYDOWN)  # Focus loss forgets held keys.
            stage, changed = "refocused", now
        elif stage == "refocused" and now - changed > 2:
            assert value.movement[2] > 0 and value.active, value
            assert still() and status.focus_hold, (status, t)
            assert any("disarmed: focus lost" in line for line in lines.lines), lines.lines
            key(pygame.K_r)
            stage, changed = "focus_resumed", now
        elif stage == "focus_resumed" and settled(moving):
            key(pygame.K_w, pygame.KEYUP)
            key(pygame.K_SPACE)
            stage, changed = "done", now
        elif stage == "done" and settled(still):
            assert sum("re-armed after a" in line for line in lines.lines) == 1, lines.lines
            results["pilot_log"] = [line for line in lines.lines if "armed" in line]
            key(pygame.K_r)
            key(pygame.K_w, pygame.KEYDOWN)
            stage, changed = "driving", now
        elif stage == "driving" and settled(moving) and now - changed > 1:
            # Closed mid-drive, splats streaming and the reconstruction worker busy.
            nonlocal closed_at
            closed_at = clock.now()
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            stage = "quit"

    try:
        result = pilot_main(
            [address, "--code", code, "--size", "480", "360", "--fps", "60"],
            on_frame=drive,
        )
        closing = clock.now() - closed_at
        assert result == 0 and stage == "quit", (result, stage)
        assert closing < 2, f"closing took {closing:.2f} s"
        results["close_s"] = round(closing, 2)
        # Nothing the app started may outlive it: Python would wait on them at exit.
        left = [t.name for t in threading.enumerate() if not t.daemon and t.name != "MainThread"]
        assert not left and not multiprocessing.active_children(), left
        time.sleep(0.3)
        # The close sent stop, and the robot heard it before the link went down.
        neutral = (OUT / "driver.log").read_text().split("Robot neutral: ")[-1].split(";")[0]
        assert neutral in ("stop", "pilot disconnected"), neutral
    finally:
        robot.terminate()
        robot.wait(timeout=8)
        driver_log.close()
    driver_lines = (OUT / "driver.log").read_text().splitlines()
    results["driver_neutral"] = [
        line.split("Robot neutral: ")[1] for line in driver_lines if "Robot neutral: " in line
    ]
    # A sampler stall's release is news to the robot at once, not after its input timeout.
    released = [line for line in results["driver_neutral"] if line.startswith("deadman released")]
    assert len(released) >= 2, results["driver_neutral"]
    print(
        "PASS stall:",
        json.dumps(results | {"short_stall_bound_ms": SHORT_STALL * 1000}),
        flush=True,
    )


if __name__ == "__main__":
    main()
