"""Real pilot CLI + MuJoCo: pilot poses go out as their samples arrive, never in pairs.

LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a uv run python e2e/pose_pacing.py [--fine]

Windows wakes timers on its 15.6 ms tick whenever it does not honour a 1 ms request (an
occluded window on Windows 11), so the run gives the link loop's sleeps that tick unless
--fine: a pacing that sleeps after every send then holds a steady sample stream back by
several milliseconds (#31). Two stretches of input: the desktop sampler as it is (about
90/s), then bursts of two samples 1.5 ms apart, as a monitor frame plus a sampler tick can
give. Passes when steady samples go out at a median under 1.5 ms after their capture and
nearly all of them go out, and no two poses go out closer than the send interval, bursts
included. With coarse timers the sleep-after-every-send pacing (6d9442a) held steady samples
4.5 ms at the median and sent 56 of 78 per second.

The window draws small and slow: llvmpipe frames hold the GIL, which would add the link
loop's own wait to every sample and hide the pacing.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pygame
from pygame._sdl2 import Window

from ito import clock
from ito.app.__main__ import main as pilot_main
from ito.app.pilot import SEND_INTERVAL
from ito.desktop.dispatch import DisplayDispatch
from ito.driver import pairing
from ito.link.peer import Peer
from ito.protocol import PilotState

OUT = Path("e2e/out/pose-pacing")
TICK = 1 / 64  # Windows' default timer period
sends = []  # (sent, capture time, deadman, stretch) of every pilot state
stretch = None  # "steady" or "burst" while measuring
samples = []  # (capture time, stretch) of every input sample
send = Peer.send
sample = DisplayDispatch.sample
fine_sleep = clock.sleep_until


def recording_send(self, message):
    accepted = send(self, message)
    if isinstance(message, PilotState) and accepted:
        sends.append((clock.now(), message.capture_time, message.deadman, stretch))
    return accepted


def bursting_sample(self, dt, on_sample):
    sample(self, dt, on_sample)
    if stretch:
        samples.append((self.value.timestamp, stretch))
    if stretch == "burst" and on_sample:
        time.sleep(0.0015)
        on_sample(replace(self.value, timestamp=clock.now(), commands=()))
        samples.append((clock.now(), stretch))


async def coarse_sleep_until(deadline):
    """Wake on the first timer tick at or after the deadline, as Windows does unasked."""
    await fine_sleep(deadline if deadline <= clock.now() else (deadline // TICK + 1) * TICK)


def main():
    global stretch
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine", action="store_true", help="1 ms timers, as Windows when asked")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CONFIG_HOME"] = os.environ["APPDATA"] = str(OUT / "config")
    Peer.send = recording_send
    DisplayDispatch.sample = bursting_sample
    if not args.fine:
        clock.sleep_until = coarse_sleep_until
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
    began = clock.now()
    stage, changed = "connecting", began

    def drive(app, window, value):
        global stretch
        nonlocal stage, changed
        now = clock.now()
        assert now - began < 90, (stage, app.state.status)
        if stage == "connecting" and app.matched_frames > 12 and app.telemetry:
            Window.from_display_module().focus()
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
            pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame.K_r))
            stage, changed = "arming", now
        elif stage == "arming" and app.state.status.armed and now - changed > 1:
            stage, stretch, changed = "steady", "steady", now
        elif stage == "steady" and now - changed > 5:
            stage, stretch, changed = "burst", "burst", now
        elif stage == "burst" and now - changed > 5:
            stretch = None
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            stage = "quit"

    try:
        result = pilot_main(
            [address, "--code", code, "--size", "320", "240", "--fps", "20"], on_frame=drive
        )
        assert result == 0 and stage == "quit", (result, stage)
    finally:
        Peer.send, DisplayDispatch.sample, clock.sleep_until = send, sample, fine_sleep
        robot.terminate()
        robot.wait(timeout=8)
        driver_log.close()
    report = {"timers": "fine" if args.fine else "coarse 15.6 ms"}
    for name in ("steady", "burst"):
        # Each sample's first send: the loop resends the latest one as a heartbeat.
        first = {}
        for t, c, armed, s in sends:
            if s == name and armed:
                first.setdefault(c, t)
        rows = sorted((t, c) for c, t in first.items())
        held = np.array([(t - c) * 1000 for t, c in rows])
        gaps = np.diff([t for t, _ in rows]) * 1000
        taken = np.diff([c for c, s in samples if s == name]) * 1000
        report[name] = dict(
            samples_per_s=round((len(taken) + 1) / 5, 1),
            sample_gap_ms_p5_p50_p95=[
                round(float(np.percentile(taken, q)), 1) for q in (5, 50, 95)
            ],
            sent_per_s=round(len(rows) / 5, 1),
            held_ms_p50=round(float(np.percentile(held, 50)), 2),
            held_ms_p95=round(float(np.percentile(held, 95)), 2),
            closest_sends_ms=round(float(gaps.min()), 2),
            sends_under_interval=int((gaps < SEND_INTERVAL * 1000 - 0.3).sum()),
        )
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    steady, burst = report["steady"], report["burst"]
    assert steady["sent_per_s"] > 0.85 * steady["samples_per_s"] > 50, report
    assert steady["held_ms_p50"] < 1.5, report
    assert steady["sends_under_interval"] == burst["sends_under_interval"] == 0, report
    assert burst["sent_per_s"] > 50, report
    print("PASS pose_pacing: steady samples go out as they arrive; bursts never send in pairs")


if __name__ == "__main__":
    main()
