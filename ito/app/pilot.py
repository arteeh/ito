"""The display reads snapshots; one background loop owns the link and worker lifetime."""

import asyncio
import contextlib
import logging
import sys
import threading
import traceback
from collections import deque
from dataclasses import replace

from ito import clock, diagnostics
from ito.desktop import DesktopState, PilotStatus
from ito.link import PairingError, connect
from ito.link.audio import Audio
from ito.protocol import Command, FrameMetadata, Paired, PilotState, Pose, Status
from ito.reconstruction import Reconstruction
from ito.render import pose
from ito.render.pose import quaternion

from . import settings
from .frames import FrameJoin, camera_matrix

log = logging.getLogger(__name__)
# A failed RGB-D worker restarts after this delay, doubling up to RESTART_LONGEST.
RESTART_FIRST, RESTART_LONGEST = 1.0, 30.0


class Pilot:
    def __init__(
        self,
        address,
        *,
        defaults=None,
        overrides=None,
        camera=None,
        cameras=1,
        audio_source="device",
        audio_sink="device",
        persist=True,
        code=None,
        credential=None,
    ):
        self.audio_options = (audio_source, audio_sink)
        self.audio = None
        self.mic_muted = self.speaker_muted = False
        self.address, self.camera_name, self.cameras = address, camera, cameras
        self.persist = persist  # The simulated robot's address changes every run.
        self.code = code  # Used once, when the robot does not know this pilot yet.
        self.credential = credential
        self.refusal = None  # Why the robot refused to pair; the pilot must act.
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
        self.retiring = []
        self.failure = None
        self.backend = "rgbd"
        self.reconstruction_status = ""
        self.tracked_frames = 0
        self.tracking = False
        self.restart_at = None  # When a failed RGB-D worker starts again.
        self.stop = threading.Event()
        self.thread = None
        self.loop = self.task = None
        self.matched_frames = 0
        self.connections = 0
        self.camera_pose = pose()
        self.last_tracking = 0.0
        self.settings_revision = 0
        self.last_frame = 0.0
        self.frame_heads = deque(maxlen=256)

    @property
    def max_splats(self):
        return self.settings.max_splats

    def set_max_splats(self, value):
        settings.Settings.model_validate(self.settings.model_dump() | {"max_splats": value})
        self.budget_change = value

    def input(self, value):
        self.latest_input = value
        for command in value.commands:
            if command in {"mute_mic", "mute_speaker"}:
                name = "mic_muted" if command == "mute_mic" else "speaker_muted"
                setattr(self, name, not getattr(self, name))
                if self.audio:
                    setattr(self.audio, name, getattr(self, name))
                continue
            if len(self.commands) >= 32:
                self.commands.clear()
                self.commands.append("e_stop")
                break
            self.commands.append(command)

    def poll(self):
        if not self.worker_lock.acquire(False):
            return None
        try:
            if self.worker is not None and not self.failure:
                try:
                    return self.worker.poll()
                except RuntimeError as exc:
                    self.failure = str(exc)
                    self.reconstruction_status = str(exc) + "; showing flat camera feed"
                    self.state = replace(self.state, flat_video=True)
            return None
        finally:
            self.worker_lock.release()

    def _status(self, link, detail=None, *, status=None, peer=None):
        old = self.state.status
        diagnostics.event("link_state", interval=0 if old.link != link else 1, state=link)
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
                reconstruction=self.reconstruction_status,
                audio=self._audio_status(peer),
                robot_audio=self.telemetry.get("audio", "") if peer else "",
                robot_microphone=bool(peer) and "microphone" in peer.description.capabilities,
                robot_speaker=bool(peer) and "speaker" in peer.description.capabilities,
                mic_muted=self.mic_muted,
                speaker_muted=self.speaker_muted,
            ),
        )

    def _audio_status(self, peer):
        """The pilot's devices that have a counterpart on the robot."""
        if not peer or not self.audio:
            return ""
        capabilities = peer.description.capabilities
        parts = [f"mic {self.audio.input_status}"] if "speaker" in capabilities else []
        if "microphone" in capabilities:
            parts.append(f"speaker {self.audio.output_status}")
        return "Audio: " + " | ".join(parts) if parts else ""

    def _save(self, name):
        if not self.persist:
            return False
        try:
            settings.save(self.address, name, self.settings)
            settings.remember(self.address, name, self.credential)
            return True
        except OSError as exc:
            log.warning("Cannot save pilot settings: %s", exc)
            self._status("CONNECTED", f"Could not save settings: {exc}")
            return False

    def _submit(self, pair, peer):
        if pair is None:
            return
        captured = peer.clock.remote_to_local(pair[1].capture_time)
        if captured <= self.state.video_time or not -0.1 <= clock.now() - captured <= 2:
            diagnostics.event(
                "frame_rejected",
                interval=1,
                sequence=pair[1].sequence,
                out_of_order=captured <= self.state.video_time,
                age_ms=(clock.now() - captured) * 1000,
            )
            return
        rgb, depth, camera, _ = FrameJoin.arrays(pair, peer.clock)
        self.last_frame = clock.now()
        self.state = replace(self.state, video=rgb, video_time=captured)
        if self.backend == "rgbd" and camera is not None:
            self._anchor(camera, pair[1].head_angles, pair[1].body_yaw)
        if self.worker is None or self.failure:
            return
        try:
            accepted = self.worker.submit(rgb, depth, camera, captured)
        except ValueError as exc:
            self.failure = str(exc)
            self.reconstruction_status = str(exc) + "; showing flat camera feed"
            self.state = replace(self.state, flat_video=True)
            return
        if accepted:
            self.matched_frames += 1
            self.last_frame = clock.now()
            self.frame_heads.append((captured, pair[1].head_angles, pair[1].body_yaw))

    def _retire(self, worker):
        # Never on the link loop: a suspended or wedged worker takes seconds to kill.
        closer = threading.Thread(target=worker.close, name="ito-reconstruction-close")
        closer.start()
        self.retiring = [thread for thread in self.retiring if thread.is_alive()] + [closer]

    def _anchor(self, camera, head_angles=None, body_yaw=0):
        self.camera_pose = camera
        anchor = camera.copy()
        # Camera and measured joints must belong to the same exposure. Status is
        # asynchronous: subtracting its newer joints makes a still base oscillate.
        if head_angles is not None:
            pan, tilt = head_angles
            anchor[:3, :3] = camera[:3, :3] @ pose(yaw=pan, pitch=tilt)[:3, :3].T
        # The body catches up to the pilot's startup-relative gaze while walking.
        # Remove that heading too: the current local head must only be applied once.
        anchor[:3, :3] = anchor[:3, :3] @ pose(yaw=-body_yaw)[:3, :3]
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
        paired = peer.credential_received.is_set()
        if paired:
            self.credential = peer.credential
        # The robot retires its code once this pilot can come back without it.
        if self._save(description.name) or (paired and not self.persist):
            if paired and peer.send(Paired(pilot=self.credential.pilot)):
                self.code = None
        self.telemetry = {}
        self.failure = None
        # Resume is never replayed across a connection; safety requests survive it.
        safety = []
        while self.commands:
            action = self.commands.popleft()
            if action != "resume":
                safety.append(action)
        self.commands.extend(safety)
        choice = self.settings.reconstruction
        self.backend = (
            ("rgbd" if {"depth", "camera-pose"} <= set(description.capabilities) else "slam")
            if choice == "auto"
            else choice
        )
        self.reconstruction_status = (
            "Flat camera feed" if self.backend == "video" else "Starting " + self.backend
        )
        self.tracked_frames = 0
        self.tracking = False
        self.restart_at = None
        restart_delay = RESTART_FIRST
        self.frame_heads.clear()
        self.state = replace(self.state, flat_video=self.backend != "rgbd", video=None)

        def reconstruct():
            with self.worker_lock:
                self.worker = Reconstruction(
                    camera.intrinsics,
                    max_splats=self.max_splats,
                    backend=self.backend,
                    origin=camera_matrix(camera.extrinsics),
                )

        if self.backend != "video":
            reconstruct()
        # Talking to a robot without a speaker, or listening to one without a microphone,
        # would only hold the pilot's devices open.
        self.audio.start(
            capture="speaker" in description.capabilities,
            playback="microphone" in description.capabilities,
        )
        self.connections += 1
        self._status("CONNECTED", "Resume to begin piloting", peer=peer)
        joined = FrameJoin()
        tasks = []
        armed = False
        previous_fresh = None
        sequence = command_sequence = 0
        pending = None
        last_status = clock.now()
        self.last_frame = last_status

        async def messages():
            nonlocal last_status, pending
            while True:
                message = await peer.messages.get()
                if isinstance(message, FrameMetadata) and message.camera == camera.name:
                    self._submit(joined.described(message), peer)
                elif isinstance(message, Status):
                    last_status = clock.now()
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
                if track.kind == "video":
                    tasks.append(asyncio.create_task(video(track)))
                else:
                    track.stop()

        tasks.extend([asyncio.create_task(messages()), asyncio.create_task(tracks())])
        deadline = clock.now()
        try:
            while not self.stop.is_set():
                now = clock.now()
                if not peer.connected or now - last_status > 2:
                    raise ConnectionError("Driver status lost; input disarmed")
                if self.worker and not self.failure:
                    message = self.worker.status()
                    if message:
                        self.reconstruction_status = message
                    if self.backend == "slam":
                        # A stalled worker may hold the pose lock; staleness alone pauses.
                        tracked = self.worker.pose()
                        if tracked:
                            transform, count, self.tracking, captured = tracked
                            if count != self.tracked_frames:
                                self.last_tracking = now
                            self.tracked_frames = count
                        live = self.tracking and now - self.last_tracking < 2
                        if live and tracked:
                            angles, body_yaw = next(
                                (
                                    (angles, yaw)
                                    for stamp, angles, yaw in self.frame_heads
                                    if stamp == captured
                                ),
                                (None, 0),
                            )
                            self._anchor(transform, angles, body_yaw)
                        elif self.tracking and not live:
                            self.reconstruction_status = (
                                "SLAM tracking paused; showing flat camera feed"
                            )
                        self.state = replace(self.state, flat_video=not live)
                if self.failure:
                    self.reconstruction_status = self.failure + "; showing flat camera feed"
                    if self.restart_at is not None:
                        wait = max(0, self.restart_at - now)
                        self.reconstruction_status += f"; restarting in {wait:.0f} s"
                    self.state = replace(self.state, flat_video=True)
                if self.failure and self.worker:
                    # A failed worker must not tear down the robot link or stop video/input.
                    with self.worker_lock:
                        failed, self.worker = self.worker, None
                    self._retire(failed)
                    if self.backend == "rgbd":
                        # Posed RGB-D has no missing model or device to wait for: try again.
                        self.restart_at = now + restart_delay
                        restart_delay = min(RESTART_LONGEST, restart_delay * 2)
                if self.restart_at is not None and now >= self.restart_at:
                    self.restart_at = self.failure = None
                    self.reconstruction_status = "Restarting rgbd"
                    reconstruct()
                    self.state = replace(self.state, flat_video=False)
                if now - self.last_frame > 5:
                    raise ConnectionError("No synchronized camera frames for five seconds")
                for task in tasks:
                    if task.done():
                        task.result()
                        raise ConnectionError("Camera stream ended")
                if self.budget_change is not None:
                    value, self.budget_change = self.budget_change, None
                    self.settings = self.settings.model_copy(update={"max_splats": value})
                    if self.worker:
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
                    if fresh != previous_fresh:
                        diagnostics.event("input_freshness_transition", fresh=fresh)
                        previous_fresh = fresh
                    if not fresh:
                        if armed:
                            log.warning(
                                "Pilot input disarmed: active=%s sample_age_ms=%.1f",
                                value.active,
                                (now - value.timestamp) * 1000,
                            )
                        armed = False
                    diagnostics.event(
                        "input_freshness",
                        interval=1,
                        fresh=fresh,
                        active=value.active,
                        armed=armed,
                        age_ms=(now - value.timestamp) * 1000,
                        sequence=sequence,
                    )
                    if self.audio:
                        diagnostics.event(
                            "audio_state",
                            interval=1,
                            input=self.audio.input_status,
                            output=self.audio.output_status,
                            mic_muted=self.mic_muted,
                            speaker_muted=self.speaker_muted,
                            **self.audio.counters,
                        )
                    matrix = value.head
                    peer.send(
                        PilotState(
                            sequence=sequence,
                            capture_time=value.timestamp,
                            deadman=armed and fresh,
                            head=Pose(
                                position=tuple(map(float, matrix[:3, 3])),
                                orientation=quaternion(matrix),
                            ),
                            hands=value.hands if fresh else {},
                            trackers=value.trackers if fresh else {},
                            buttons={name: True for name in value.buttons},
                            axes={
                                **value.axes,
                                self.settings.move_x: value.movement[0],
                                self.settings.move_y: value.movement[2],
                            },
                        )
                    )
                    sequence += 1
                deadline = max(deadline + 1 / 60, now)
                await asyncio.sleep(max(0, deadline - clock.now()))
        finally:
            peer.send(Command(sequence=command_sequence, action="stop"))
            for task in tasks:
                task.cancel()
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                with self.worker_lock:
                    worker, self.worker = self.worker, None
                if worker:
                    self._retire(worker)

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
                self.audio = Audio(*self.audio_options)
                self.audio.mic_muted = self.mic_muted
                self.audio.speaker_muted = self.speaker_muted
                async with await connect(
                    self.address,
                    video_tracks=self.cameras,
                    connect_timeout=8,
                    audio_io=self.audio,
                    code=self.code,
                    credential=self.credential,
                ) as peer:
                    delay = 0.5
                    await self._session(peer)
            except asyncio.CancelledError:
                break
            except PairingError as exc:
                log.warning("Pilot connection: %s", exc)
                if self.credential is not None:
                    # The robot's code was rotated, so it forgot this pilot.
                    self.credential = None
                    if self.persist:
                        settings.forget_credential(self.address)
                    if self.code is not None:
                        continue
                # Retrying the same code cannot succeed and counts against the robot's limit.
                self.refusal = str(exc)
                self._status("REFUSED", self.refusal)
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

        # Daemon: a link stuck in teardown must not keep the closed app alive.
        self.thread = threading.Thread(target=run, name="ito-link", daemon=True)
        self.thread.start()
        return self

    def close(self):
        self.stop.set()
        if self.loop and self.task:
            with contextlib.suppress(RuntimeError):
                self.loop.call_soon_threadsafe(self.task.cancel)
        if self.thread:
            with diagnostics.stage("link_thread"):
                self.thread.join(timeout=12)
            if self.thread.is_alive():
                stack = sys._current_frames().get(self.thread.ident)
                log.error("Pilot link stuck at:\n%s", "".join(traceback.format_stack(stack)))
                raise RuntimeError("Pilot link did not shut down")
        # Cancellation during connection setup can precede the session's cleanup block.
        with self.worker_lock:
            worker, self.worker = self.worker, None
        if worker is not None:
            self._retire(worker)
        deadline = clock.now() + 8
        for closer in self.retiring:
            with diagnostics.stage("reconstruction"):
                closer.join(timeout=max(0, deadline - clock.now()))
            if closer.is_alive():
                raise RuntimeError("Reconstruction worker did not shut down")

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
