"""Reconnects, idle media, robot faults, and the pilot app closing cleanly under faults.

LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a uv run python e2e/lifecycle.py
"""

import asyncio
import contextlib
import json
import os
import signal
import socket
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import psutil
from webrtc import ROOT, events, line

from ito import clock
from ito.driver import pairing
from ito.link import connect
from ito.protocol import Command, PilotState, Status

CODE = "246813"
# From the pilot closing the window to the app's process and all its children being gone.
CLOSE_BUDGET = 2.0
CLI = "from ito.driver.cli import main; main()"
# Fault injection: the driver's watchdog task dies while a pilot is driving.
WATCHDOG_DIES = (
    """
import asyncio
import ito.driver.server as server
watchdog = server.Driver._run
async def dies(self):
    task = asyncio.create_task(watchdog(self))
    while self.state != "active":
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.3)
    task.cancel()
    raise RuntimeError("injected watchdog failure")
server.Driver._run = dies
"""
    + CLI
)


@asynccontextmanager
async def robot(script=CLI, **options):
    with tempfile.TemporaryDirectory(prefix="ito-lifecycle-") as directory:
        journal = Path(directory) / "robot.jsonl"
        code_file = Path(directory) / "pairing-code"
        pairing.write(code_file, CODE)
        log = (Path(directory) / "driver.log").open("w")
        env = os.environ | {"PYTHONPATH": str(ROOT) + os.pathsep + str(ROOT / "e2e")}
        driver = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            script,
            "robot:Robot",
            "--adapter-args",
            json.dumps({"journal": str(journal), **options}),
            "--port",
            "0",
            "--pairing-file",
            str(code_file),
            stdout=asyncio.subprocess.PIPE,
            stderr=log,
            env=env,
            cwd=ROOT,
        )
        try:
            address = (await line(driver)).rsplit(" ", 1)[1]
            yield address, journal, driver
            log.flush()
            driver.log = (Path(directory) / "driver.log").read_text()
        finally:
            if driver.returncode is None:
                driver.send_signal(signal.SIGTERM)
                async with asyncio.timeout(5):
                    await driver.wait()
            log.close()
            assert driver.returncode == 0 or script != CLI


async def status(peer, predicate):
    async with asyncio.timeout(5):
        while True:
            message = await peer.messages.get()
            if isinstance(message, Status) and predicate(message):
                return message


async def fresh_peer(address):
    async with asyncio.timeout(5):
        while True:
            try:
                return await connect(address, code=CODE)
            except ConnectionError:
                await asyncio.sleep(0.05)


async def drive(peer, start=0, duration=0.4):
    sequence = start
    deadline = clock.now() + duration
    while clock.now() < deadline:
        sequence += 1
        assert peer.connected
        peer.send(PilotState(sequence=sequence, capture_time=clock.now(), deadman=True))
        await asyncio.sleep(0.015)
    return sequence


async def run():
    async with robot() as (address, journal, driver):
        monitor = await asyncio.create_subprocess_exec(
            sys.executable,
            str(ROOT / "e2e" / "link_monitor.py"),
            address,
            CODE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=ROOT,
        )
        try:
            assert json.loads(await line(monitor))["type"] == "robot"
            metrics = json.loads(await line(monitor))
            assert metrics["frames"]["video"] >= 5 and metrics["frames"]["audio"] >= 5
            assert metrics["driver"]["state"] == "stopped"
            monitor.send_signal(signal.SIGINT)
            async with asyncio.timeout(5):
                await monitor.wait()
            assert monitor.returncode == 0, (await monitor.stderr.read()).decode()
        finally:
            if monitor.returncode is None:
                monitor.kill()
                await monitor.wait()
        # A failed media offer must release its peer and camera resources.
        with contextlib.suppress(ConnectionError):
            bad = await connect(address, video_tracks=0, code=CODE)
            await bad.close()
            raise AssertionError("offer without receiving camera unexpectedly succeeded")
        async with await fresh_peer(address) as peer:
            video = await peer.tracks.get()
            audio = await peer.tracks.get()
            if video.kind != "video":
                video, audio = audio, video
            assert video.id == peer.description.cameras[0].track_id
            # Leave media unread longer than the idle timeout: clocks keep the link alive,
            # and bounded draining means the first frame we read is still live.
            await asyncio.sleep(5.2)
            assert peer.connected
            frame = await video.recv()
            assert float(frame.pts * frame.time_base) > 4.8
            assert not any(e["event"] == "apply" for e in events(journal))
            assert peer.send(Command(sequence=0, action="resume"))
            await drive(peer)
            await status(peer, lambda s: s.state == "active")
            assert peer.send(Command(sequence=1, action="e-stop"))
            await status(peer, lambda s: s.state == "e-stopped")

        async with await fresh_peer(address) as peer:
            # Neither reconnect nor live deadman input clears the e-stop latch.
            sequence = await drive(peer)
            latched = await status(peer, lambda s: s.state == "e-stopped")
            count = latched.telemetry["applied"]
            await drive(peer, sequence)
            assert (await status(peer, lambda s: s.state == "e-stopped")).telemetry[
                "applied"
            ] == count
            # A state captured before resume, but delivered afterward, must not actuate.
            delayed = PilotState(sequence=1000, capture_time=clock.now(), deadman=True)
            assert peer.send(Command(sequence=0, action="resume"))
            await status(peer, lambda s: s.command_sequence == 0 and s.state == "neutral")
            assert peer.send(delayed)
            await asyncio.sleep(0.15)
            assert len([e for e in events(journal) if e["event"] == "apply"]) == count
            sequence = await drive(peer, 1000)
            await status(peer, lambda s: s.state == "active")
            # An older sequence with a fresh timestamp cannot override current input.
            older = PilotState(sequence=999, capture_time=clock.now(), deadman=False)
            peer.pilot.send(older.model_dump_json())
            await drive(peer, sequence, 0.15)
            assert not peer.closed.is_set()
            # Losing reliable control neutralizes immediately, even if pilot state is alive.
            closed_at = clock.now()
            peer.control.close()
            async with asyncio.timeout(2):
                await peer.closed.wait()
            entries = events(journal)
            neutral = [e for e in entries if e["event"] == "neutral" and e["time"] >= closed_at]
            assert neutral and neutral[0]["time"] - closed_at < 0.1
            assert not any(
                e["event"] == "apply" and e["time"] > neutral[0]["time"] for e in entries
            )
        assert driver.returncode is None

    for failure in ("fail_apply", "fail_telemetry", "fail_neutral_once"):
        async with robot(**{failure: True}) as (address, journal, driver):
            async with await connect(address, code=CODE) as peer:
                assert peer.send(Command(sequence=0, action="resume"))
                await drive(peer)
                await status(peer, lambda s: s.state == "fault")
                assert peer.send(Command(sequence=1, action="resume"))
                await status(peer, lambda s: s.command_sequence == 1 and s.state == "fault")
                await drive(peer, 1000)
                assert driver.returncode is None
                entries = events(journal)
                if failure == "fail_neutral_once":
                    failed_at = next(e["time"] for e in entries if e["event"] == "neutral_failed")
                    assert any(e["event"] == "neutral" and e["time"] > failed_at for e in entries)
                    assert not any(e["event"] == "apply" for e in entries)
                if failure == "fail_telemetry":
                    detected = next(e["time"] for e in entries if e["event"] == "telemetry_failed")
                    assert not any(e["event"] == "apply" and e["time"] > detected for e in entries)
                if failure == "fail_apply":
                    assert not any(e["event"] == "apply" for e in entries)
                    failures = [e for e in entries if e["event"] == "apply_failed"]
                    assert len(failures) == 1
                    assert any(
                        e["event"] == "neutral" and e["time"] >= failures[0]["time"]
                        for e in entries
                    )
    async with robot(WATCHDOG_DIES) as (address, journal, driver):
        async with await connect(address, code=CODE) as peer:
            assert peer.send(Command(sequence=0, action="resume"))
            sequence = 0
            async with asyncio.timeout(10):
                # Keep driving as a pilot would, until the driver gives up.
                while driver.returncode is None:
                    sequence += 1
                    state = PilotState(sequence=sequence, capture_time=clock.now(), deadman=True)
                    peer.send(state)
                    with contextlib.suppress(TimeoutError):
                        async with asyncio.timeout(0.015):
                            await driver.wait()
        entries = events(journal)
        assert any(e["event"] == "apply" for e in entries), "the injected driver never drove"
        assert entries[-1]["event"] == "neutral", entries[-3:]
    assert driver.returncode == 1 and "safety watchdog stopped" in driver.log, driver.log
    closes = {}
    for scenario in ("streaming", "connecting", "stuck_link", "reconstruction", "sim"):
        closes[scenario] = await closing(scenario)
    print(
        "PASS: idle live media, failed offers, reconnect/e-stop latch, "
        "fresh resume, channel loss, adapter faults, watchdog failure, "
        "app close (seconds to exit):",
        json.dumps(closes),
    )


@asynccontextmanager
async def mujoco(directory):
    """The simulated robot a pilot would drive, on a port of its own."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    code_file = directory / "pairing-code"
    pairing.write(code_file, CODE)
    with (directory / "driver.log").open("w") as log:
        driver = await asyncio.create_subprocess_exec(
            sys.executable,
            *("-m", "drivers.mujoco.cli", "--port", str(port), "--pairing-file", str(code_file)),
            stdout=log,
            stderr=log,
            env=os.environ | {"MUJOCO_GL": "osmesa", "LP_NUM_THREADS": "2"},
            cwd=ROOT,
        )
    try:
        yield f"127.0.0.1:{port}"
    finally:
        driver.send_signal(signal.SIGTERM)
        async with asyncio.timeout(8):
            await driver.wait()


@asynccontextmanager
async def silent_robot():
    """Accepts the pilot's connection and never answers it."""
    held = []
    server = await asyncio.start_server(lambda r, w: held.append(w), "127.0.0.1", 0)
    try:
        yield f"127.0.0.1:{server.sockets[0].getsockname()[1]}"
    finally:
        for writer in held:
            writer.close()
        server.close()


def zombie(process):
    try:
        return process.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


async def closing(scenario):
    """Close the real app's window with a fault in place; it must be gone within budget."""
    with tempfile.TemporaryDirectory(prefix=f"ito-close-{scenario}-") as name:
        directory = Path(name)
        robot_context = (
            silent_robot()
            if scenario == "connecting"
            else contextlib.nullcontext(None)
            if scenario == "sim"
            else mujoco(directory)
        )
        async with robot_context as address:
            target = ["--sim"] if address is None else [address, "--code", CODE]
            env = os.environ | {
                "ITO_E2E_CLOSE": scenario,
                "ITO_E2E_MARK": str(directory / "mark"),
                "XDG_CONFIG_HOME": str(directory / "config"),
                "MUJOCO_GL": "osmesa",
                "PYTHONPATH": str(ROOT),
            }
            with (directory / "app.log").open("w") as log:
                app = await asyncio.create_subprocess_exec(
                    sys.executable,
                    str(ROOT / "e2e" / "closing_app.py"),
                    *target,
                    *("--size", "480", "360", "--fps", "60"),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=log,
                    env=env,
                    cwd=directory,
                )
            try:
                while not (said := await line(app, 90)).startswith("closed "):
                    pass  # pygame's greeting
                closed = float(said.split()[1])
                family = psutil.Process(app.pid).children(recursive=True)
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(max(0, closed + CLOSE_BUDGET - clock.now())):
                        await app.wait()
                took = clock.now() - closed
                _, alive = psutil.wait_procs(family, timeout=0.2)
                alive = [p for p in alive if not zombie(p)]  # Exited; init has yet to reap it.
                text = (directory / "app.log").read_text()
                assert app.returncode is not None, f"{scenario}: still running after {took:.1f} s"
                assert not alive, (scenario, [p.cmdline() for p in alive])
                expected = 1 if scenario == "stuck_link" else 0
                assert app.returncode == expected, (scenario, app.returncode, text[-3000:])
                assert "Traceback" not in text, (scenario, text[-3000:])
                if scenario == "stuck_link":
                    assert "Pilot link stuck at" in text, text[-3000:]
                if scenario == "reconstruction":
                    assert "Reconstruction process" in text and "killed" in text, text[-3000:]
            finally:
                if app.returncode is None:
                    for process in psutil.Process(app.pid).children(recursive=True):
                        process.kill()
                    app.kill()
                    await app.wait()
        if address is not None and scenario != "connecting":
            # The robot is left neutral and keeps serving: it never depends on a clean close.
            neutral = (directory / "driver.log").read_text().split("Robot neutral: ")[-1]
            reason = neutral.split(";")[0]
            assert reason in ("pilot disconnected", "stop", "input timeout"), (scenario, reason)
        return round(took, 2)


if __name__ == "__main__":
    asyncio.run(run())
