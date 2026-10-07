"""Run: uv run python e2e/mujoco_driver.py. CLI, real WebRTC, OSMesa, and physical motion."""

import asyncio
import contextlib
import json
import math
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path

import av
import numpy as np

from ito.driver import pairing
from ito.link import connect
from ito.protocol import Command, FrameMetadata, PilotState, Pose, Status

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "e2e" / "out" / "mujoco"


def head(yaw=0.0, pitch=0.0):
    sy, cy = math.sin(yaw / 2), math.cos(yaw / 2)
    sp, cp = math.sin(pitch / 2), math.cos(pitch / 2)
    return Pose(orientation=(cy * sp, sy * cp, -sy * sp, cy * cp))


def rotation(pose):
    x, y, z, w = pose.orientation
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def save(name, frame, metadata):
    with av.open(str(OUT / f"{name}.png"), mode="w", format="image2") as output:
        stream = output.add_stream("png", rate=1)
        stream.width, stream.height, stream.pix_fmt = frame.width, frame.height, "rgb24"
        for packet in stream.encode(frame.reformat(format="rgb24")):
            output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    np.save(OUT / f"{name}-depth.npy", depth(metadata))
    (OUT / f"{name}.json").write_text(metadata.model_dump_json(indent=2))


def depth(metadata):
    value = metadata.depth
    return np.frombuffer(value.to_bytes(), dtype="<u2").reshape(value.height, value.width) / 1000


class Pilot:
    def __init__(self, peer):
        self.peer = peer
        self.metadata = {}
        self.metadata_ready = asyncio.Event()
        self.statuses = deque(maxlen=1000)
        self.observations = deque(maxlen=300)
        self.sequence = 0
        self.last_sent = 0.0
        self.frames = 0
        self.unmatched = 0
        self.last_capture = -1
        self.last_sequence = -1
        self.origin = None
        self.latencies = []
        self.tasks = [asyncio.create_task(self.messages()), asyncio.create_task(self.video())]

    async def messages(self):
        while True:
            message = await self.peer.messages.get()
            if isinstance(message, FrameMetadata):
                assert message.video_pts is not None
                assert message.capture_time > self.last_capture
                assert message.sequence > self.last_sequence
                self.last_capture, self.last_sequence = message.capture_time, message.sequence
                origin = message.capture_time - message.video_pts / 90000
                if self.origin is None:
                    self.origin = origin
                assert abs(origin - self.origin) < 0.001
                assert message.camera_pose is not None and message.depth is not None
                self.metadata[message.video_pts] = message
                self.metadata_ready.set()
                if len(self.metadata) > 300:
                    del self.metadata[next(iter(self.metadata))]
            elif isinstance(message, Status):
                self.statuses.append((time.monotonic(), message))

    async def video(self):
        track = await self.peer.tracks.get()
        assert track.id == self.peer.description.cameras[0].track_id
        while True:
            frame = await track.recv()
            assert frame.time_base.numerator == 1 and frame.time_base.denominator == 90000
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(2):
                    while frame.pts not in self.metadata:
                        self.metadata_ready.clear()
                        await self.metadata_ready.wait()
            metadata = self.metadata.get(frame.pts)
            if metadata is None:
                self.unmatched += 1
                continue
            self.frames += 1
            self.latencies.append(
                time.monotonic() - self.peer.clock.remote_to_local(metadata.capture_time)
            )
            self.observations.append((time.monotonic(), frame, metadata))

    def healthy(self):
        for task in self.tasks:
            if task.done():
                task.result()
                raise AssertionError("pilot media consumer stopped")

    async def drive(self, duration, *, yaw=0.0, pitch=0.0, forward=0.0, turn=0.0, deadman=True):
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            self.healthy()
            self.last_sent = time.monotonic()
            self.peer.send(
                PilotState(
                    sequence=self.sequence,
                    capture_time=self.last_sent,
                    deadman=deadman,
                    head=head(yaw, pitch),
                    axes={"move_y": forward, "move_x": turn},
                )
            )
            self.sequence += 1
            await asyncio.sleep(1 / 60)

    async def latest(self, name):
        requested = time.monotonic()
        async with asyncio.timeout(5):
            while not self.observations or self.observations[-1][2].capture_time < requested:
                self.healthy()
                await asyncio.sleep(0.01)
        _, frame, metadata = self.observations[-1]
        assert frame.width == 320 and frame.height == 240
        d = depth(metadata)
        assert np.isfinite(d).all() and np.mean(d > 0) > 0.95
        assert 0.1 < np.median(d) < 12
        assert d.std() > 0.2
        save(name, frame, metadata)
        return frame.to_ndarray(format="rgb24").astype(float), metadata

    async def status(self, state, after=0.0):
        async with asyncio.timeout(3):
            while True:
                self.healthy()
                for timestamp, status in reversed(self.statuses):
                    if timestamp >= after and status.state == state:
                        return timestamp, status
                await asyncio.sleep(0.01)

    async def close(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.peer.close()


async def run():
    OUT.mkdir(parents=True, exist_ok=True)
    executable = str(Path(sys.executable).with_name("ito-driver-mujoco"))
    for arguments, expected in (
        (["--width", "321"], "video size must be even"),
        (["--fps", "0"], "fps must be between"),
        (["--pan", "missing"], "missing"),
        (["/no/such/ito-model.xml"], "ito-model.xml"),
    ):
        process = await asyncio.create_subprocess_exec(
            executable,
            *arguments,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ | {"MUJOCO_GL": "osmesa"},
        )
        async with asyncio.timeout(15):
            stdout, stderr = await process.communicate()
        assert process.returncode == 1, (arguments, stdout, stderr)
        assert expected in stderr.decode() and b"Traceback" not in stderr, stderr
    log = (OUT / "driver.log").open("w")
    code_file = OUT / "pairing-code"
    code = pairing.rotate(code_file)
    driver = await asyncio.create_subprocess_exec(
        executable,
        "--gl",
        "osmesa",
        "--port",
        "0",
        "--pairing-file",
        str(code_file),
        stdout=asyncio.subprocess.PIPE,
        stderr=log,
        cwd=ROOT,
        env=os.environ | {"LIBGL_ALWAYS_SOFTWARE": "1", "LP_NUM_THREADS": "2"},
    )
    pilot = None
    try:
        async with asyncio.timeout(30):
            line = await driver.stdout.readline()
        assert line.startswith(b"Ito driver listening"), (OUT / "driver.log").read_text()
        address = line.decode().strip().rsplit(" ", 1)[1]
        peer = await connect(address, receive_audio=False, code=code)
        pilot = Pilot(peer)
        assert peer.description.capabilities == (
            "depth",
            "camera-pose",
            "head-pan-tilt",
            "differential-drive",
        )
        await pilot.drive(0.8)
        initial_rgb, initial = await pilot.latest("initial")
        intrinsics = peer.description.cameras[0].intrinsics
        assert abs(intrinsics.fy - 240 / (2 * math.tan(math.radians(65) / 2))) < 0.001
        d = depth(initial)
        # Back-project rug into the startup robot frame: top is at Y=0.016-0.19 m.
        floor_y = []
        for row in (220, 230):
            for col in (80, 160, 240):
                local = (
                    np.array(
                        [
                            (col + 0.5 - intrinsics.cx) / intrinsics.fx,
                            -(row + 0.5 - intrinsics.cy) / intrinsics.fy,
                            -1,
                        ]
                    )
                    * d[row, col]
                )
                world = rotation(initial.camera_pose) @ local + initial.camera_pose.position
                floor_y.append(world[1])
        assert sum(abs(y + 0.174) < 0.008 for y in floor_y) >= 4, floor_y

        await pilot.drive(1.2, yaw=0.7, pitch=0.3)
        head_rgb, turned_head = await pilot.latest("head-left-up")
        _, status = await pilot.status("active")
        assert abs(status.telemetry["head_pan"] - 0.7) < 0.08, status
        assert abs(status.telemetry["head_tilt"] - 0.3) < 0.08, status
        forward = -rotation(turned_head.camera_pose)[:, 2]
        assert forward[0] < -0.4 and forward[1] > 0.15, forward
        assert np.mean(abs(head_rgb - initial_rgb)) > 15
        assert np.mean(abs(depth(turned_head) - depth(initial))) > 0.2

        await pilot.drive(1.2, yaw=2.3, pitch=1.0)
        await pilot.latest("head-limits")
        _, status = await pilot.status("active")
        assert 1.3 < status.telemetry["head_pan"] < 1.42, status
        assert 0.58 < status.telemetry["head_tilt"] < 0.67, status
        await pilot.drive(1.0)
        _, before = await pilot.latest("before-drive")
        await pilot.drive(1.8, forward=0.8)
        moving_rgb, moving = await pilot.latest("forward")
        displacement = np.array(moving.camera_pose.position) - before.camera_pose.position
        assert displacement[2] < -0.6 and abs(displacement[0]) < 0.15, (
            displacement,
            pilot.statuses[-1],
        )
        assert np.mean(abs(moving_rgb - initial_rgb)) > 8

        # Stop sending pilot state while keeping WebRTC connected.
        stopped_sending = pilot.last_sent
        received, neutral = await pilot.status("neutral", after=stopped_sending)
        assert neutral.reason == "input timeout", neutral
        timeout_ms = (received - stopped_sending) * 1000
        assert timeout_ms < 450, timeout_ms  # status is published at 10 Hz
        assert neutral.telemetry["left_command"] == neutral.telemetry["right_command"] == 0
        await asyncio.sleep(0.7)
        _, stopped = await pilot.latest("timeout-neutral")
        await asyncio.sleep(0.4)
        _, still = await pilot.latest("timeout-still")
        drift = np.linalg.norm(np.array(still.camera_pose.position) - stopped.camera_pose.position)
        assert drift < 0.015, drift
        _, status = await pilot.status("neutral", after=received + 0.4)
        assert abs(status.telemetry["left_velocity"]) < 0.1, status
        assert abs(status.telemetry["right_velocity"]) < 0.1, status

        await pilot.drive(1.0, turn=0.8)
        _, turned_base = await pilot.latest("turn-right")
        _, status = await pilot.status("active")
        assert status.telemetry["base_yaw"] < -0.45, status
        # The base steers right while the head keeps looking in the startup direction.
        forward = -rotation(turned_base.camera_pose)[:, 2]
        assert abs(forward[0]) < 0.15, forward
        await pilot.drive(0.3, forward=0.8)
        sent = time.monotonic()
        assert peer.send(Command(sequence=0, action="e-stop"))
        await pilot.drive(0.7, forward=1.0, yaw=-0.7)
        _, estop = await pilot.status("e-stopped", after=sent + 0.3)
        assert estop.telemetry["left_command"] == estop.telemetry["right_command"] == 0
        _, held = await pilot.latest("e-stop")
        await pilot.drive(0.5, forward=1.0, yaw=-0.7)
        _, held_again = await pilot.latest("e-stop-held")
        assert (
            np.linalg.norm(np.array(held.camera_pose.position) - held_again.camera_pose.position)
            < 0.015
        )
        assert (
            abs(np.dot(held.camera_pose.orientation, held_again.camera_pose.orientation)) > 0.9999
        )
        assert peer.send(Command(sequence=1, action="resume"))
        await pilot.drive(0.4, forward=-0.5)
        _, status = await pilot.status("active", after=time.monotonic() - 0.2)
        assert status.telemetry["left_command"] < 0
        await pilot.drive(0.3, deadman=False)
        _, status = await pilot.status("neutral", after=time.monotonic() - 0.2)
        assert status.reason == "deadman released"
        pilot.healthy()
        assert pilot.frames > 60 and pilot.unmatched == 0
        assert peer.rejected_messages == 0
        result = {
            "matched_rgb_depth_pose_frames": pilot.frames,
            "timeout_status_ms": round(timeout_ms, 1),
            "stationary_drift_m": round(float(drift), 5),
            "forward_displacement_m": [round(float(v), 3) for v in displacement],
            "capture_to_decode_median_ms": round(float(np.median(pilot.latencies)) * 1000, 1),
            "samples": str(OUT.relative_to(ROOT)),
        }
        await pilot.close()
        pilot = None
        await asyncio.sleep(0.4)
        # A new pilot gets a fresh video epoch and working camera, not a stopped relay.
        pilot = Pilot(await connect(address, receive_audio=False, code=code))
        await pilot.drive(0.5)
        await pilot.latest("reconnected")
        assert pilot.frames > 2 and pilot.unmatched == 0
        result["reconnected_frames"] = pilot.frames
        (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    finally:
        if pilot:
            await pilot.close()
        if driver.returncode is None:
            driver.send_signal(signal.SIGTERM)
            try:
                async with asyncio.timeout(10):
                    await driver.wait()
            except TimeoutError:
                driver.kill()
                await driver.wait()
        log.close()
        assert driver.returncode == 0, (OUT / "driver.log").read_text()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run())
