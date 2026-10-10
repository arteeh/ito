"""The display reads snapshots; one background loop owns the link and worker lifetime."""

import asyncio
import contextlib
import logging
import math
import sys
import threading
import traceback
from collections import deque
from dataclasses import replace

import numpy as np

from ito import clock, diagnostics
from ito.desktop import DesktopState, PilotStatus
from ito.link import PairingError, connect
from ito.link.audio import Audio
from ito.protocol import Command, FrameMetadata, Paired, PilotState, Pose, Status
from ito.rates import CounterRate, Rate
from ito.reconstruction import Reconstruction
from ito.render import pose
from ito.render.pose import quaternion

from . import settings
from .frames import FrameJoin, camera_matrix

log = logging.getLogger(__name__)
# SLAM's 3D view drops to the flat feed after this long without tracking, and returns only
# once tracking has run this steadily, so the view does not flicker at the edge of tracking.
TRACKING_LOST = 2.0
TRACKING_STEADY = 1.0
TRACKING_GAP = 0.5
# A failed worker restarts after this delay, doubling up to RESTART_LONGEST; one that ran
# RESTART_HEALTHY seconds before failing starts the count again.
RESTART_FIRST, RESTART_LONGEST, RESTART_HEALTHY = 1.0, 30.0, 60.0
# Input older than INPUT_FRESH releases the deadman. A pilot who never let go is re-armed when
# input returns within SHORT_STALL; a longer gap may have changed the scene, so they resume.
INPUT_FRESH, SHORT_STALL = 0.2, 1.0
# A closing app waits this long for the link to tear down, then this long more for each
# reconstruction worker; anything still running is abandoned so the app can exit.
LINK_WAIT, WORKER_WAIT = 1.0, 0.8


class ShutdownStuck(RuntimeError):
    """Part of the pilot did not stop; only ending the process releases it."""


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
        self.armed = False  # Resume sent and input fresh since, or back within SHORT_STALL.
        self.stalled = False  # Disarmed by an input stall alone; fresh input soon re-arms.
        self.resumed = False  # The last safety command this pilot gave was resume.
        self.focus_hold = False  # The robot is stopped only because the window lost focus.
        self.budget_change = None
        self.worker = None
        self.worker_lock = threading.Lock()
        self.retiring = []
        self.failure = None
        self.backend = "rgbd"
        self.reconstruction_status = ""
        self.tracked_frames = 0
        self.slam_restarts = {}  # The worker's map restarts by cause, as last logged.
        self.tracking = False
        self.tracking_steady_since = 0.0
        self.slam_view = False  # SLAM's 3D view is shown rather than the flat feed.
        self.restart_at = None  # When a failed worker starts again.
        self.stop = threading.Event()
        self.thread = None
        self.loop = self.task = None
        self.matched_frames = 0
        self.connections = 0
        self.camera_pose = pose()
        self.last_tracking = 0.0
        # Where the robot thought recent submitted frames looked (capture time, rotation), and
        # the turn from that onto SLAM's map, from the newest tracked one: the live frame
        # hangs where the map's splats of the same view are, not a heading error away.
        self.priors = deque(maxlen=64)
        self.correction = np.eye(3)
        self.settings_revision = 0
        self.last_frame = 0.0
        self.extrinsics = pose()
        self.camera_rate = Rate()  # joined frames handed to reconstruction
        self.scene_rate = CounterRate()  # frames reconstruction integrated into the scene

    @property
    def max_splats(self):
        return self.settings.max_splats

    def set_max_splats(self, value):
        settings.Settings.model_validate(self.settings.model_dump() | {"max_splats": value})
        self.budget_change = value

    def input(self, value):
        for command in value.commands:
            if command in {"mute_mic", "mute_speaker"}:
                name = "mic_muted" if command == "mute_mic" else "speaker_muted"
                setattr(self, name, not getattr(self, name))
                if self.audio:
                    setattr(self.audio, name, getattr(self, name))
                continue
            if command == "focus_stop":
                # Commands precede this sample's inactive input, so armed is still current.
                self.focus_hold = self.resumed and (self.armed or self.stalled)
            else:
                if command == "rearm":
                    # Only a stop that focus loss alone caused is undone by clicking back in;
                    # the driver still waits for fresh deadman input after the resume.
                    if not self.focus_hold or self.state.status.e_stop:
                        continue
                    command = "resume"
                self.focus_hold = False
            if command in {"stop", "focus_stop", "e_stop", "resume"}:
                self.resumed = command == "resume"
            if len(self.commands) >= 32:
                self.commands.clear()
                self.commands.append("e_stop")
                break
            self.commands.append(command)
        self.latest_input = value

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
        rates = (None, None, None)
        worker = self.worker
        if peer:
            rates = (
                self.telemetry.get("pilot_input_hz"),
                self.camera_rate.hz(),
                self.scene_rate.hz(worker.integrated.value) if worker is not None else None,
            )
            diagnostics.event(
                "rates", interval=1, pose_hz=rates[0], camera_hz=rates[1], scene_hz=rates[2]
            )
        self.state = replace(
            self.state,
            status=PilotStatus(
                link=link,
                latency_ms=peer.clock.rtt * 1000 if peer and peer.clock.rtt is not None else None,
                robot=name,
                e_stop=status.state == "e-stopped" if status else old.e_stop,
                robot_state=status.state if status else old.robot_state if link == old.link else "",
                detail=detail or (f"{status.state}: {status.reason}" if status else ""),
                input_latency_ms=self.telemetry.get("pilot_input_latency_ms") if peer else None,
                rates=rates,
                reconstruction=self.reconstruction_status,
                audio=self._audio_status(peer),
                robot_audio=self.telemetry.get("audio", "") if peer else "",
                robot_microphone=bool(peer) and "microphone" in peer.description.capabilities,
                robot_speaker=bool(peer) and "speaker" in peer.description.capabilities,
                mic_muted=self.mic_muted,
                speaker_muted=self.speaker_muted,
                armed=self.armed,
                focus_hold=self.focus_hold,
            ),
        )

    def _log_restarts(self, restarts):
        """One diagnostic event per cause whose count of SLAM map restarts went up."""
        if not restarts:
            return
        for cause, count in restarts.items():
            if count > self.slam_restarts.get(cause, 0):
                diagnostics.event(
                    "slam_restart",
                    cause=cause,
                    count=count - self.slam_restarts.get(cause, 0),
                    totals=restarts,
                )
        self.slam_restarts = restarts

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
        measured = False
        if self.backend == "slam":
            # SLAM places the camera itself; it needs only the driver's best guess of
            # this exposure's world orientation to align its map with gravity and heading.
            # A driver that measures its camera pose or gaze also holds SLAM to its heading.
            measured = camera is not None or pair[1].head_angles is not None
            camera = self._prior(camera, pair[1])
            self.priors.append((captured, camera[:3, :3]))
        orientation = None
        if camera is not None:
            orientation = np.eye(4)
            orientation[:3, :3] = self.correction @ camera[:3, :3]
        self.state = replace(
            self.state, video=rgb, video_time=captured, video_orientation=orientation
        )
        if self.backend == "rgbd" and camera is not None:
            self._anchor(camera)
        if self.worker is None or self.failure:
            return
        self.camera_rate.tick()
        try:
            accepted = self.worker.submit(rgb, depth, camera, captured, measured=measured)
        except ValueError as exc:
            self.failure = str(exc)
            self.reconstruction_status = str(exc) + "; showing flat camera feed"
            self.state = replace(self.state, flat_video=True)
            return
        if accepted:
            self.matched_frames += 1
            self.last_frame = clock.now()

    def _retire(self, worker):
        # Never on the link loop: a busy or wedged worker takes a while to stop or kill.
        closer = threading.Thread(target=worker.close, name="ito-reconstruction-close")
        closer.start()
        retiring = [(c, w) for c, w in self.retiring if c.is_alive() or w.process.is_alive()]
        self.retiring = retiring + [(closer, worker)]

    def _prior(self, camera, metadata):
        """World-from-camera as far as the robot knows it at this exposure.

        A driver that knows the camera pose says so; otherwise its body heading and
        measured head pan and tilt turn the startup camera mount; otherwise the mount.
        """
        if camera is not None:
            return camera
        result = self.extrinsics.copy()
        if metadata.head_angles is not None:
            pan, tilt = metadata.head_angles
            result[:3, :3] = pose(yaw=pan, pitch=tilt)[:3, :3] @ result[:3, :3]
        return pose(yaw=metadata.body_yaw) @ result

    def _anchor(self, camera):
        """Only the camera's position anchors the pilot's eye.

        The pilot's head is the whole view rotation, in the gravity-aligned world anchored
        where the robot started. The robot turns its head and body to look where the pilot
        looks, but those turns never rotate the view: the pilot is never turned by the robot.
        """
        self.camera_pose = camera
        self.state = replace(self.state, robot_camera=pose(camera[:3, 3]))

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
        self.tracking = self.slam_view = False
        self.restart_at = None
        self.priors.clear()
        self.correction = np.eye(3)
        restart_delay = RESTART_FIRST
        worker_started = 0.0
        self.extrinsics = camera_matrix(camera.extrinsics)
        self.state = replace(
            self.state,
            flat_video=self.backend != "rgbd",
            video=None,
            video_orientation=None,
            video_intrinsics=camera.intrinsics,
            video_fov=2 * math.atan(camera.intrinsics.width / (2 * camera.intrinsics.fx)),
        )

        def reconstruct():
            nonlocal worker_started
            worker_started = clock.now()
            self.slam_restarts = {}
            with self.worker_lock:
                self.worker = Reconstruction(
                    camera.intrinsics,
                    max_splats=self.max_splats,
                    backend=self.backend,
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
        self.armed = self.focus_hold = self.stalled = False
        previous_fresh = None
        last_fresh = None  # Capture time of the newest sample that held the deadman.

        def disarm(reason):
            nonlocal last_fresh
            if self.armed or self.stalled:
                log.warning("Pilot input disarmed: %s; resume to pilot again", reason)
                diagnostics.event("input_disarmed", reason=reason)
            self.armed = self.stalled = False
            last_fresh = None

        sequence = command_sequence = 0
        pending = None
        last_status = clock.now()
        self.last_frame = last_status

        async def messages():
            nonlocal last_status, pending
            while True:
                message = await peer.messages.get()
                if isinstance(message, FrameMetadata) and message.camera == camera.name:
                    for pair in joined.described(message):
                        self._submit(pair, peer)
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
                    for pair in joined.decoded(frame):
                        self._submit(pair, peer)

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
                                prior = next(
                                    (r for t, r in reversed(self.priors) if t == captured), None
                                )
                                if prior is not None:
                                    self.correction = transform[:3, :3] @ prior.T
                                if now - self.last_tracking > TRACKING_GAP:
                                    self.tracking_steady_since = now
                                self.last_tracking = now
                            self.tracked_frames = count
                        self._log_restarts(self.worker.restarts())
                        live = self.tracking and now - self.last_tracking < TRACKING_LOST
                        if live and tracked:
                            self._anchor(transform)
                        elif self.tracking and not live:
                            self.reconstruction_status = (
                                "SLAM tracking paused; showing flat camera feed"
                            )
                        steady = (
                            live
                            and now - self.last_tracking < TRACKING_GAP
                            and now - self.tracking_steady_since >= TRACKING_STEADY
                        )
                        self.slam_view = live if self.slam_view else steady
                        self.state = replace(self.state, flat_video=not self.slam_view)
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
                    # Posed RGB-D has no missing model or device to wait for, nor has SLAM
                    # that already tracked (a CUDA fault, a diverged solve): try again.
                    if self.backend == "rgbd" or self.tracked_frames:
                        if now - worker_started >= RESTART_HEALTHY:
                            restart_delay = RESTART_FIRST
                        self.restart_at = now + restart_delay
                        restart_delay = min(RESTART_LONGEST, restart_delay * 2)
                if self.restart_at is not None and now >= self.restart_at:
                    self.restart_at = self.failure = None
                    self.reconstruction_status = "Restarting " + self.backend
                    self.tracked_frames = 0
                    self.tracking = self.slam_view = False
                    reconstruct()
                    self.state = replace(self.state, flat_video=self.backend != "rgbd")
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
                        given = self.commands.popleft()
                        action = ("stop" if given == "focus_stop" else given).replace("_", "-")
                        if action == "resume":
                            # Held input counts from here; a stall before resume is not one.
                            self.armed, self.stalled, last_fresh = True, False, None
                        else:
                            disarm("focus lost" if given == "focus_stop" else action)
                        pending = Command(sequence=command_sequence, action=action)
                        command_sequence += 1
                    if peer.send(pending):
                        pending = None
                    else:
                        break  # Retry backpressure; accepted commands use reliable SCTP.
                value = self.latest_input
                if value is not None:
                    fresh = value.active and now - value.timestamp < INPUT_FRESH
                    if fresh != previous_fresh:
                        diagnostics.event("input_freshness_transition", fresh=fresh)
                        previous_fresh = fresh
                    # The gap since the last sample that held the deadman, whichever thread
                    # stalled: a link loop that was blocked never saw the input go stale.
                    gap = None
                    if last_fresh is not None:
                        gap = (value.timestamp if fresh else now) - last_fresh
                    if not value.active:
                        disarm("window inactive")
                    elif gap is not None and gap > SHORT_STALL:
                        disarm(f"input stalled over {SHORT_STALL * 1000:.0f} ms")
                    elif not fresh and self.armed:
                        age_ms = (now - value.timestamp) * 1000
                        log.warning(
                            "Pilot input stalled, deadman released: sample_age_ms=%.1f", age_ms
                        )
                        diagnostics.event("input_stalled", age_ms=age_ms)
                        self.armed, self.stalled = False, last_fresh is not None
                    elif fresh and self.stalled:
                        log.info("Pilot re-armed after a %.0f ms input stall", gap * 1000)
                        diagnostics.event("input_rearmed", stall_ms=gap * 1000)
                        self.armed, self.stalled = True, False
                    if fresh and self.armed:
                        last_fresh = value.timestamp
                    diagnostics.event(
                        "input_freshness",
                        interval=1,
                        fresh=fresh,
                        active=value.active,
                        armed=self.armed,
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
                    head = Pose(
                        position=tuple(map(float, matrix[:3, 3])),
                        orientation=quaternion(matrix),
                    )
                    if fresh:
                        state = PilotState(
                            sequence=sequence,
                            capture_time=value.timestamp,
                            deadman=self.armed,
                            head=head,
                            hands=value.hands,
                            trackers=value.trackers,
                            buttons={name: True for name in value.buttons},
                            axes={
                                **value.axes,
                                self.settings.move_x: value.movement[0],
                                self.settings.move_y: value.movement[2],
                            },
                        )
                    else:
                        # The release is news as of now; resent with the stale sample's time,
                        # the driver drops it as old and waits out its own input timeout.
                        state = PilotState(
                            sequence=sequence, capture_time=now, deadman=False, head=head
                        )
                    peer.send(state)
                    sequence += 1
                shown = self.state.status
                if (shown.armed, shown.focus_hold) != (self.armed, self.focus_hold):
                    self.state = replace(
                        self.state,
                        status=replace(shown, armed=self.armed, focus_hold=self.focus_hold),
                    )
                deadline = max(deadline + 1 / 60, now)
                await asyncio.sleep(max(0, deadline - clock.now()))
        finally:
            self.armed = False
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
        """Bounded: the pilot closed the app, and nothing on the link may keep it open.

        The robot never depends on this: every exit path sends stop first, and a driver
        that hears nothing more neutralizes within its input timeout.
        """
        self.stop.set()
        if self.audio:
            # A device still closing is left for the operating system to release.
            self.audio.close_timeout = 0.3
        if self.loop and self.task:
            with contextlib.suppress(RuntimeError):
                self.loop.call_soon_threadsafe(self.task.cancel)
        stuck = []
        if self.thread:
            with diagnostics.stage("link_thread"):
                self.thread.join(timeout=LINK_WAIT)
            if self.thread.is_alive():
                stack = sys._current_frames().get(self.thread.ident)
                log.error("Pilot link stuck at:\n%s", "".join(traceback.format_stack(stack)))
                stuck.append("the pilot link")
        # Cancellation during connection setup, or a link stuck in teardown, can precede the
        # session's cleanup block. A worker left running would keep the app from exiting:
        # multiprocessing joins live worker processes at interpreter exit, without a timeout.
        with self.worker_lock:
            worker, self.worker = self.worker, None
        if worker is not None:
            self._retire(worker)
        deadline = clock.now() + WORKER_WAIT
        for closer, worker in self.retiring:
            with diagnostics.stage("reconstruction"):
                closer.join(timeout=max(0, deadline - clock.now()))
            if closer.is_alive() or worker.process.is_alive():
                stuck.append(f"reconstruction process {worker.process.pid}")
        if stuck:
            raise ShutdownStuck(" and ".join(stuck) + " did not shut down")

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
