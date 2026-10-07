"""Real MuJoCo + pilot audio, mute, and missing-device runs.

DISPLAY=:97 LIBGL_ALWAYS_SOFTWARE=1 uv run python e2e/audio.py
"""

import json
import os
import socket
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import pygame

from ito.app.__main__ import main as pilot_main

OUT = Path("e2e/out/audio")


def measure(path, frequency):
    with wave.open(str(path)) as source:
        assert source.getframerate() == 48000 and source.getnchannels() == 1
        samples = np.frombuffer(source.readframes(source.getnframes()), "<i2").astype(float)
    # Both directions must contain audible tone, sustained mute, and resumed tone.
    blocks = samples[: len(samples) // 4800 * 4800].reshape(-1, 4800)
    energy = np.sqrt(np.mean(blocks**2, axis=1))
    audible = energy > 1000
    silent = energy < 10
    assert audible.sum() > 15 and silent.sum() > 8, (path, energy)
    quiet = np.flatnonzero(silent)
    assert audible[: quiet[-1]].any() and audible[quiet[-1] + 1 :].sum() > 5, energy
    peaks = np.fft.rfftfreq(4800, 1 / 48000)[
        np.abs(np.fft.rfft(blocks[audible], axis=1)).argmax(axis=1)
    ]
    assert np.max(np.abs(peaks - frequency)) <= 10, peaks
    return {
        "frequency_hz": float(np.median(peaks)),
        "audible_blocks": int(audible.sum()),
        "silent_blocks": int(silent.sum()),
    }


def run(devices=False):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = str(sock.getsockname()[1])
    label = "devices" if devices else "tones"
    with (OUT / f"{label}-driver.log").open("w") as log:
        driver = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "drivers.mujoco.cli",
                "--port",
                port,
                "--audio-source",
                "device" if devices else "tone:440",
                "--audio-sink",
                "device" if devices else str(OUT / "robot.wav"),
            ],
            env=os.environ | {"MUJOCO_GL": "osmesa", "LD_LIBRARY_PATH": "/opt/data/lib/osmesa"},
            stdout=log,
            stderr=log,
        )
        start = time.monotonic()
        connected = None
        stage = 0
        click_pending = None

        def click(x, y):
            nonlocal click_pending
            pygame.mouse.set_pos((x, y))
            click_pending = (x, y, 3)

        def drive(app, window, value):
            nonlocal connected, stage, click_pending
            if click_pending:
                x, y, remaining = click_pending
                if remaining <= 1:
                    event = pygame.MOUSEBUTTONDOWN if remaining else pygame.MOUSEBUTTONUP
                    pygame.event.post(pygame.event.Event(event, button=1, pos=(x, y)))
                click_pending = (x, y, remaining - 1) if remaining else None
            assert time.monotonic() - start < 35, app.state.status
            if app.state.status.link != "CONNECTED":
                return
            if connected is None:
                connected = time.monotonic()
            elapsed = time.monotonic() - connected
            if devices:
                if elapsed > 3:
                    assert "unavailable" in app.state.status.audio, app.state.status
                    assert "unavailable" in app.state.status.robot_audio, app.state.status
                    assert app.state.video is not None
                    pygame.event.post(pygame.event.Event(pygame.QUIT))
                    stage = 5
                return
            if stage == 0 and elapsed > 2:
                click(30, 265)  # Shared ImGui checkbox, before the request status line.
                stage = 1
            elif stage == 1 and elapsed > 4:
                assert app.state.status.mic_muted
                click(30, 282)
                stage = 10
            elif stage == 10 and elapsed > 4.3:
                click(165, 282)
                stage = 2
            elif stage == 2 and elapsed > 6:
                assert app.state.status.speaker_muted and not app.state.status.mic_muted
                click(165, 282)
                stage = 3
            elif stage == 3 and elapsed > 8:
                assert not app.state.status.speaker_muted
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F12))
                stage = 4
            elif stage == 4 and elapsed > 9:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
                stage = 5

        try:
            assert (
                pilot_main(
                    [
                        f"127.0.0.1:{port}",
                        "--size",
                        "800",
                        "600",
                        "--fps",
                        "60",
                        "--reconstruction",
                        "video",
                        "--capture-dir",
                        str(OUT),
                        "--audio-source",
                        "device" if devices else "tone:880",
                        "--audio-sink",
                        "device" if devices else str(OUT / "pilot.wav"),
                    ],
                    on_frame=drive,
                )
                == 0
            )
            assert stage == 5
        finally:
            driver.terminate()
            driver.wait(timeout=10)
        assert driver.returncode == 0
        assert "Traceback" not in (OUT / f"{label}-driver.log").read_text()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CONFIG_HOME"] = str(OUT / "config")
    run()
    report = {"pilot": measure(OUT / "pilot.wav", 440), "robot": measure(OUT / "robot.wav", 880)}
    run(devices=True)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("PASS: two-way Opus, mic/speaker mute and recovery, no-device video/link continuity")


if __name__ == "__main__":
    main()
