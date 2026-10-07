"""Real MuJoCo + pilot audio: tones, live devices, closing, and missing devices.

Each phase runs the pilot app in its own process, so PortAudio sees that phase's devices:
  tones    two-way Opus through WAV sinks, mic/speaker mute and recovery
  devices  pilot mic and speakers on real devices; mute toggles, e-stop, close the window
  stalled  the same close while the audio service stops responding
  missing  no audio devices; video and the link carry on

On a machine without sound hardware, a JACK dummy server provides paced devices:
  . /opt/data/lib/portaudio/env.sh   # libportaudio2 and jackd2 from Debian, unpacked
DISPLAY=:97 LIBGL_ALWAYS_SOFTWARE=1 uv run python e2e/audio.py
"""

import argparse
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import psutil
import pygame

from ito import clock, diagnostics
from ito.app.__main__ import main as pilot_main
from ito.driver import pairing

OUT = Path("e2e/out/audio")
JACK = f"ito-e2e-audio-{os.getpid()}"
CLOSE_LIMIT_S = 1.5  # Window closed to pilot_main returning, devices and link included.
STALLED_CLOSE_LIMIT_S = 4.0


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


class Errors(logging.Handler):
    def __init__(self):
        super().__init__(logging.ERROR)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def click(position):
    """A pilot's click holds the button for several frames."""
    pygame.event.post(
        pygame.event.Event(pygame.MOUSEMOTION, pos=position, rel=(0, 0), buttons=(0, 0, 0))
    )
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=position, button=1))
    release = pygame.event.Event(pygame.MOUSEBUTTONUP, pos=position, button=1)
    threading.Timer(0.12, pygame.event.post, (release,)).start()


def phase(name):
    """One pilot session against a fresh MuJoCo driver; returns this phase's report."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = str(sock.getsockname()[1])
    code_file = OUT / f"{name}-pairing-code"
    code = pairing.rotate(code_file)
    tones = name == "tones"
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    errors = Errors()
    logging.getLogger().addHandler(errors)
    report = {}
    with (OUT / f"{name}-driver.log").open("w") as log:
        driver = subprocess.Popen(
            [sys.executable, "-m", "drivers.mujoco.cli", "--port", port]
            + ["--pairing-file", str(code_file), "--audio-source", "tone:440"]
            + ["--audio-sink", "device" if name == "missing" else str(OUT / f"{name}-robot.wav")],
            env=os.environ | {"MUJOCO_GL": "osmesa", "LD_LIBRARY_PATH": library_path()},
            stdout=log,
            stderr=log,
        )
        start = clock.now()
        stage, changed = "connecting", start
        closing = None
        jitter = None

        def go(next_stage):
            nonlocal stage, changed
            stage, changed = next_stage, clock.now()

        def drive(app, window, value):
            nonlocal closing, jitter
            if diagnostics.current().enabled:
                report["diagnostic_run"] = diagnostics.current().run_id
            waited = clock.now() - changed
            status = app.state.status
            layout = window.overlay.layout
            assert clock.now() - start < 40, (stage, status)
            if stage == "connecting":
                if status.link == "CONNECTED" and waited > 3:
                    if name == "missing":
                        assert "unavailable" in status.audio, status
                        assert "unavailable" in status.robot_audio, status
                        assert app.state.video is not None
                        go("close")
                    else:
                        expected = "mic ready | speaker ready"
                        assert tones or expected in status.audio, status
                        if name == "devices":
                            report["audio_before_jitter"] = dict(app.audio.counters)

                            async def scheduling_jitter():
                                import asyncio

                                for _ in range(30):
                                    await asyncio.sleep(0.065)
                                    time.sleep(0.035)  # noqa: ASYNC251 — inject a real scheduling stall

                            import asyncio

                            jitter = asyncio.run_coroutine_threadsafe(scheduling_jitter(), app.loop)
                        click(layout["mute_mic"])
                        go("mic muted")
            elif stage == "mic muted" and waited > (
                3.5 if name == "devices" else 2 if tones else 0.6
            ):
                assert status.mic_muted and not status.speaker_muted, status
                click(layout["mute_mic"])
                go("mic unmuted")
            elif stage == "mic unmuted" and waited > 0.6:
                assert not status.mic_muted, status
                click(layout["mute_speaker"])
                go("speaker muted")
            elif stage == "speaker muted" and waited > (2 if tones else 0.6):
                assert status.speaker_muted and not status.mic_muted, status
                if tones:
                    click(layout["mute_speaker"])
                    go("speaker unmuted")
                else:
                    # The owner's run: mute toggles and an e-stop, then close the window.
                    click(layout["mute_mic"])
                    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_e))
                    pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame.K_e))
                    go("e-stop")
            elif stage == "speaker unmuted" and waited > 2:
                assert not status.speaker_muted, status
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F12))
                go("close")
            elif stage == "e-stop" and status.e_stop and waited > 0.5:
                assert status.mic_muted and status.speaker_muted, status
                if name == "devices":
                    assert jitter.done(), "Jitter exercise did not finish"
                    jitter.result()
                    report["audio_after_jitter"] = dict(app.audio.counters)
                    before, after = report["audio_before_jitter"], app.audio.counters
                    assert after["played"] > before["played"] + 20, report
                    assert after["underruns"] - before["underruns"] <= 1, report
                if name == "stalled":
                    freeze()
                go("close")
            elif stage == "close" and waited > 1:
                closing = clock.now()
                pygame.event.post(pygame.event.Event(pygame.QUIT))
                go("closed")

        try:
            arguments = [f"127.0.0.1:{port}", "--code", code, "--size", "800", "600"]
            arguments += ["--fps", "60", "--reconstruction", "video", "--capture-dir", str(OUT)]
            if tones:
                arguments += ["--audio-source", "tone:880"]
                arguments += ["--audio-sink", str(OUT / "pilot.wav")]
            result = pilot_main(arguments, on_frame=drive)
            closed = clock.now()
            assert not errors.messages, errors.messages
            if "diagnostic_run" in report:
                path = OUT / "config/ito/diagnostics.jsonl"
                records = [json.loads(line) for line in path.read_text().splitlines()]
                records = [r for r in records if r["run_id"] == report["diagnostic_run"]]
                assert any(r["event"] == "diagnostics_closed" for r in records)
                if name == "stalled":
                    assert any(r["event"] == "audio_close_timeout" for r in records)
                ends = [
                    r for r in records if r["event"] == "shutdown_stage" and r["state"] == "end"
                ]
                assert {"audio", "audio_devices", "webrtc", "link_thread", "window"} <= {
                    r["stage"] for r in ends
                }, ends
            assert result == 0 and stage == "closed", (result, stage)
            report["close_s"] = closed - closing
            limit = STALLED_CLOSE_LIMIT_S if name == "stalled" else CLOSE_LIMIT_S
            assert report["close_s"] < limit, report
            if name != "stalled":
                # A closed window leaves nothing of the session behind.
                leftover = [
                    t.name for t in threading.enumerate() if t is not threading.main_thread()
                ]
                assert not leftover, leftover
                children = [p for p in psutil.Process().children() if p.pid != driver.pid]
                assert not children, [p.cmdline() for p in children]
        finally:
            driver.terminate()
            driver.wait(timeout=10)
    assert driver.returncode == 0, driver.returncode
    assert "Traceback" not in (OUT / f"{name}-driver.log").read_text()
    return report


def library_path():
    """MuJoCo's software GL for the driver, plus PortAudio when env.sh provides it."""
    paths = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    return ":".join(dict.fromkeys(["/opt/data/lib/osmesa", *paths]))


def freeze():
    """Stop the audio service mid-session, as a wedged device driver would."""
    server = psutil.Process(int(os.environ["ITO_E2E_JACKD"]))
    server.suspend()
    # Longer than the pilot app waits for its link; the e2e resumes it after the phase.
    thaw = threading.Timer(15, server.resume)
    thaw.daemon = True
    thaw.start()


def run(name, env=None):
    started = clock.now()
    process = subprocess.run(
        [sys.executable, __file__, "--phase", name],
        env=os.environ | (env or {}),
        capture_output=True,
        text=True,
        timeout=120,
    )
    (OUT / f"{name}.log").write_text(process.stdout + process.stderr)
    # The process exits once its window closes, even with a stalled audio service.
    assert process.returncode == 0, f"{name} failed; see {OUT / f'{name}.log'}"
    lines = [line for line in process.stdout.splitlines() if line.startswith("REPORT ")]
    return json.loads(lines[-1][7:]) | {"process_s": clock.now() - started}


def devices_available():
    probe = "import sounddevice as sd; sd.check_input_settings(); sd.check_output_settings()"
    return subprocess.run([sys.executable, "-c", probe], capture_output=True).returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("tones", "devices", "stalled", "missing"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CONFIG_HOME"] = str(OUT / "config")
    os.environ["JACK_NO_START_SERVER"] = "1"  # Only the server started here may provide devices.
    if args.phase:
        print("REPORT " + json.dumps(phase(args.phase)), flush=True)
        return
    report = {"tones": run("tones")}
    report["tones"] |= {
        "pilot": measure(OUT / "pilot.wav", 440),
        "robot": measure(OUT / "tones-robot.wav", 880),
    }
    jackd = None
    if shutil.which("jackd"):
        jackd = subprocess.Popen(
            ["jackd", "-r", "-n", JACK, "-d", "dummy", "-r", "48000", "-p", "960"],
            env=os.environ | {"JACK_NO_AUDIO_RESERVATION": "1"},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        os.environ |= {"JACK_DEFAULT_SERVER": JACK, "ITO_E2E_JACKD": str(jackd.pid)}
        time.sleep(1)
    try:
        assert devices_available(), (
            "The devices phase needs an audio input and output: a sound card, or "
            "`. /opt/data/lib/portaudio/env.sh` for a JACK dummy server"
        )
        report["devices"] = run("devices")
        if jackd:
            report["stalled"] = run("stalled")
    finally:
        if jackd:
            os.killpg(jackd.pid, signal.SIGCONT)
            jackd.terminate()
            jackd.wait(timeout=10)
            del os.environ["JACK_DEFAULT_SERVER"], os.environ["ITO_E2E_JACKD"]
    # Without a JACK server or sound card, PortAudio finds no devices.
    report["missing"] = run("missing", {"JACK_DEFAULT_SERVER": JACK + "-absent"})
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(
        "PASS: two-way Opus, mute and recovery, prompt clean close with live devices "
        "(after mute toggles and e-stop, and with a stalled audio service), no-device continuity"
    )


if __name__ == "__main__":
    main()
