"""The display reads snapshots; one background loop owns the link and worker lifetime."""

import asyncio
import contextlib
import logging
import math
import threading
import time
from collections import deque
from dataclasses import replace

from ito.desktop import DesktopState, PilotStatus
from ito.link import connect
from ito.protocol import Command, FrameMetadata, PilotState, Pose, Status
from ito.reconstruction import Reconstruction
from ito.render import pose

from . import settings
from .frames import FrameJoin

log = logging.getLogger(__name__)


class Pilot:
    def __init__(self, address, *, defaults=None, overrides=None, camera=None, cameras=1):
        self.address, self.camera_name, self.cameras = address, camera, cameras
        self.settings = defaults or settings.Settings()
        self.defaults = self.settings
        self.overrides = overrides or {}
        self.state = DesktopState(status=PilotStatus(link="CONNECTING"))
        self.telemetry = {}
        self.latest_input = None
        self.commands = deque()
        self.budget_change = None
        self.worker = None
        self.worker_lock = threading.Lock()
        self.failure = None
        self.stop = threading.Event()
        self.thread = None
        self.loop = self.task = None
        self.matched_frames = 0
        self.connections = 0
        self.camera_pose = pose()
        self.settings_revision = 0
        self.last_frame = 0.0

    @property
    def max_splats(self):
        return self.settings.max_splats

    def set_max_splats(self, value):
        settings.Settings.model_validate(self.settings.model_dump() | {"max_splats": value})
        self.budget_change = value

    def input(self, value):
        self.latest_input = value
        for command in value.commands:
            if len(self.commands) >= 32:
                self.commands.clear()
                self.commands.append("e_stop")
                break
            self.commands.append(command)

    def poll(self):
        if not self.worker_lock.acquire(False):
            return None
        try:
            if self.worker is not None:
                try:
                    return self.worker.poll()
                except RuntimeError as exc:
                    self.failure = str(exc)
            return None
        finally:
            self.worker_lock.release()

    def _status(self, link, detail=None, *, status=None, peer=None):
        old = self.state.status
        name = peer.description.name if peer else old.robot
        self.state = replace(
            self.state,
            status=PilotStatus(
                link=link,
                latency_ms=peer.clock.rtt * 1000 if peer and peer.clock.rtt is not None else None,
                robot=name,
                e_stop=status.state == "e-stopped" if status else old.e_stop,
                detail=detail or (f"{status.state}: {status.reason}" if status else ""),
                input_latency_ms=self.telemetry.get("pilot_input_latency_ms") if peer else None,
            ),
        )

    def _save(self, name):
        try:
            settings.save(self.address, name, self.settings)
        except OSError as exc:
            log.warning("Cannot save pilot settings: %s", exc)
            self._status("CONNECTED", f"Could not save settings: {exc}")

    def _submit(self, pair, peer):
        if pair is None:
            return
        rgb, depth, camera, captured = FrameJoin.arrays(pair, peer.clock)
        if not -0.1 <= time.monotonic() - captured <= 2:
            return
        if self.worker.submit(rgb, depth, camera, captured):
            self.matched_frames += 1
            self.last_frame = time.monotonic()
            self.camera_pose = camera
            anchor = camera.copy()
            # A pan/tilt camera has already followed the head. Remove its measured
            # rotation before applying the current local head pose at display rate.
            if "head-pan-tilt" in peer.description.capabilities:
                pan, tilt = self.telemetry.get("head_pan"), self.telemetry.get("head_tilt")
                if isinstance(pan, (int, float)) and isinstance(tilt, (int, float)):
                    anchor[:3, :3] = camera[:3, :3] @ pose(yaw=pan, pitch=tilt)[:3, :3].T
            self.state = replace(self.state, robot_camera=anchor)

    async def _session(self, peer):
        description = peer.description
        camera = next(
            (
                c
                for c in description.cameras
                if self.camera_name is None or c.name == self.camera_name
            ),
            None,
        )
        if camera is None:
            raise ValueError("Driver has no selected camera")
        selected = settings.load(self.address, description.name, self.defaults)
        self.settings = settings.Settings.model_validate(selected.model_dump() | self.overrides)
        self.overrides = {}
        self.settings_revision += 1
        self._save(description.name)
        self.telemetry = {}
        self.failure = None
        # Resume is never replayed across a connection; safety requests survive it.
        safety = []
        while self.commands:
            action = self.commands.popleft()
            if action != "resume":
                safety.append(action)
        self.commands.extend(safety)
        with self.worker_lock:
            self.worker = Reconstruction(camera.intrinsics, max_splats=self.max_splats)
        self.connections += 1
        self._status("CONNECTED", "Press R to resume piloting", peer=peer)
        joined = FrameJoin()
        tasks = []
        armed = False
        sequence = command_sequence = 0
        pending = None
        last_status = time.monotonic()
        self.last_frame = last_status

        async def messages():
            nonlocal last_status, pending
            while True:
                message = await peer.messages.get()
                if isinstance(message, FrameMetadata) and message.camera == camera.name:
                    self._submit(joined.described(message), peer)
                elif isinstance(message, Status):
                    last_status = time.monotonic()
                    self.telemetry = message.telemetry
                    if pending and message.command_sequence == pending.sequence:
                        pending = None
                    self._status("CONNECTED", status=message, peer=peer)

        async def video(track):
            while True:
                frame = await track.recv()
                if track.id == camera.track_id:
                    self._submit(joined.decoded(frame), peer)

        async def tracks():
            while True:
                track = await peer.tracks.get()
                tasks.append(asyncio.create_task(video(track)))

        tasks.extend([asyncio.create_task(messages()), asyncio.create_task(tracks())])
        deadline = time.monotonic()
        try:
            while not self.stop.is_set():
                now = time.monotonic()
                if not peer.connected or now - last_status > 2:
                    raise ConnectionError("Driver status lost; input disarmed")
                if self.failure:
                    raise RuntimeError(self.failure)
                if now - self.last_frame > 5:
                    raise ConnectionError("No synchronized camera frames for five seconds")
                for task in tasks:
                    if task.done():
                        task.result()
                        raise ConnectionError("Camera stream ended")
                if self.budget_change is not None:
                    value, self.budget_change = self.budget_change, None
                    self.settings = self.settings.model_copy(update={"max_splats": value})
                    self.worker.set_max_splats(value)
                    self._save(description.name)
                while pending or self.commands:
                    if pending is None:
                        action = self.commands.popleft().replace("_", "-")
                        armed = action == "resume"
                        pending = Command(sequence=command_sequence, action=action)
                        command_sequence += 1
                    if peer.send(pending):
                        pending = None
                    else:
                        break  # Retry backpressure; accepted commands use reliable SCTP.
                value = self.latest_input
                if value is not None:
                    fresh = value.active and now - value.timestamp < 0.2
                    if not fresh:
                        armed = False
                    matrix = value.head
                    yaw = math.atan2(float(matrix[0, 2]), float(matrix[2, 2]))
                    pitch = math.asin(max(-1, min(1, -float(matrix[1, 2]))))
                    sy, cy = math.sin(yaw / 2), math.cos(yaw / 2)
                    sp, cp = math.sin(pitch / 2), math.cos(pitch / 2)
                    peer.send(
                        PilotState(
                            sequence=sequence,
                            capture_time=value.timestamp,
                            deadman=armed and fresh,
                            head=Pose(
                                position=tuple(map(float, matrix[:3, 3])),
                                orientation=(cy * sp, sy * cp, -sy * sp, cy * cp),
                            ),
                            buttons={name: True for name in value.buttons},
                            axes={
                                self.settings.move_x: value.movement[0],
                                self.settings.move_y: value.movement[2],
                            },
                        )
                    )
                    sequence += 1
                deadline = max(deadline + 1 / 60, now)
                await asyncio.sleep(max(0, deadline - time.monotonic()))
        finally:
            peer.send(Command(sequence=command_sequence, action="stop"))
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            with self.worker_lock:
                worker, self.worker = self.worker, None
                worker.close()

    async def _run(self):
        self.loop = asyncio.get_running_loop()
        self.task = asyncio.current_task()
        delay = 0.5
        while not self.stop.is_set():
            try:
                self._status(
                    "CONNECTING" if not self.connections else "RECONNECTING",
                    "Connecting to " + self.address,
                )
                async with await connect(
                    self.address, video_tracks=self.cameras, connect_timeout=8
                ) as peer:
                    delay = 0.5
                    await self._session(peer)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning("Pilot connection: %s", exc)
                self._status("RECONNECTING", str(exc))
            if not self.stop.is_set():
                await asyncio.sleep(delay)
                delay = min(4, delay * 2)

    def start(self):
        def run():
            with contextlib.suppress(asyncio.CancelledError):
                asyncio.run(self._run())

        self.thread = threading.Thread(target=run, name="ito-link")
        self.thread.start()
        return self

    def close(self):
        self.stop.set()
        if self.loop and self.task:
            with contextlib.suppress(RuntimeError):
                self.loop.call_soon_threadsafe(self.task.cancel)
        if self.thread:
            self.thread.join(timeout=12)
            if self.thread.is_alive():
                raise RuntimeError("Pilot link did not shut down")

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
