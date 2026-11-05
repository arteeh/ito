import asyncio
import contextlib
import logging
import math
import sys
import threading
import traceback
from collections import deque
from dataclasses import replace

from ito import clock, diagnostics
from ito.desktop import DesktopState, PilotStatus
from ito.desktop.settings import load as app_settings
from ito.link import PairingError, connect, media
from ito.protocol import Command, FrameMetadata, Motion, Paired, PilotState, Pose, Sensors, Status
from ito.rates import CounterRate, Rate
from ito.render import pose
from ito.render.pose import quaternion
from ito.tofmesh.worker import CLOSE_WAIT, TofMesh

from . import settings
from .frames import FrameJoin, camera_matrix

log = logging.getLogger(__name__)
RESTART_FIRST, RESTART_LONGEST, RESTART_HEALTHY = 1.0, 30.0, 60.0
# ToF mesh: the eye is the robot camera's position averaged over the last Microduck gait period,
# moved on by half the change from the period before's average. A walking duck's head sways
# ~1.6 cm sideways at 2.3 Hz (sim, 1 Nov), which shook near things in the view; a whole period's
# average has none of it. One average alone trails a steady walk by half a period; the change
# between two consecutive ones is the walk's speed without the sway, and puts it back. A start
# or stop is followed within two periods.
EYE_AVERAGE = 0.42
INPUT_FRESH, SHORT_STALL = 0.2, 1.0
# Pilot state goes out as each input sample arrives (a headset's 72-90 Hz, a monitor's refresh),
# and at least at HEARTBEAT when samples stop: the robot must hear the deadman go stale.
# Only a sample that comes within SEND_INTERVAL of the last send waits, and goes out with
# any newer one as one pose, so near-simultaneous samples are not sent as a pair. The
# window is short on purpose: the driver keeps only the newest pose anyway, and a wider
# one (1/120 s) made a desktop pilot's samples wait out the interval, which doubled
# input-to-robot latency on Windows (#31).
SEND_INTERVAL, HEARTBEAT = 0.002, 1 / 60
# Quitting waits for the workers as long as their close() may take (worker.CLOSE_WAIT);
# a healthy one is gone in about 0.6 s.
LINK_WAIT, WORKER_WAIT = 1.0, CLOSE_WAIT


def _targets(zones):
    """A RangeGrid's per-zone targets as one row per zone, nearest first, NaN-padded."""
    depth = max(1, max(map(len, zones), default=0))
    return [[*zone, *[math.nan] * (depth - len(zone))] for zone in zones]


class ShutdownStuck(RuntimeError):
    pass


class Pilot:
    def __init__(
        self,
        address,
        *,
        defaults=None,
        overrides=None,
        code=None,
        credential=None,
    ):
        self.address = address
        self.code = code
        self.credential = credential
        self.refusal = None
        self.description = None
        self.settings = defaults or settings.Settings(**app_settings().model_dump())
        self.defaults = self.settings
        self.overrides = overrides or {}
        self.state = DesktopState(status=PilotStatus(link="CONNECTING"))
        self.telemetry = {}
        self.latest_input = None
        self.input_arrived = None
        self.commands = deque()
        self.armed = False
        self.stalled = False
        self.resumed = False
        self.focus_hold = False
        self.worker = None
        self.worker_lock = threading.Lock()
        self.retiring = []
        self.failure = None
        self.reconstruction_status = ""
        self.restart_at = None
        self.stop = threading.Event()
        self.thread = None
        self.loop = self.task = None
        self.connections = 0
        self.camera_pose = pose()
        self.eye_positions = deque()  # (time, camera position) for EYE_AVERAGE
        self.settings_revision = 0
        self.last_frame = 0.0
        self.camera_rate = Rate()
        self.frame_cameras = deque(maxlen=16)  # (capture time, camera pose) of shown frames
        self.motion_time = -1.0
        self.backend_rate = CounterRate()

    def input(self, value):
        for command in value.commands:
            if command == "focus_stop":
                self.focus_hold = self.resumed and (self.armed or self.stalled)
            else:
                if command == "rearm":
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
        loop, arrived = self.loop, self.input_arrived
        if loop and arrived:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(arrived.set)

    def statistics(self):
        if not self.worker_lock.acquire(False):
            return {}
        try:
            return self.worker.statistics() if self.worker is not None else {}
        finally:
            self.worker_lock.release()

    def poll(self):
        if not self.worker_lock.acquire(False):
            return None
        try:
            if self.worker is not None and not self.failure:
                try:
                    return self.worker.poll()
                except RuntimeError as exc:
                    message, _, cause = str(exc).partition("\n")
                    log.warning("Reconstruction failed: %s\n%s", message, cause.rstrip())
                    self.failure = message
                    self.reconstruction_status = message + "; showing flat camera feed"
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
                self.backend_rate.hz(worker.integrated.value) if worker is not None else None,
            )
            diagnostics.event(
                "rates", interval=1, pose_hz=rates[0], camera_hz=rates[1], backend_hz=rates[2]
            )
        self.state = replace(
            self.state,
            status=PilotStatus(
                link=link,
                link_lagging=bool(peer)
                and (
                    (peer.clock.rtt is not None and peer.clock.rtt >= INPUT_FRESH)
                    or (self.telemetry.get("pilot_input_latency_ms") or 0) >= INPUT_FRESH * 1000
                    or (status is not None and status.reason == "input timeout")
                ),
                latency_ms=peer.clock.rtt * 1000 if peer and peer.clock.rtt is not None else None,
                robot=name,
                e_stop=status.state == "e-stopped" if status else old.e_stop,
                robot_state=status.state if status else old.robot_state if link == old.link else "",
                detail=detail or (f"{status.state}: {status.reason}" if status else ""),
                rates=rates,
                reconstruction=self.reconstruction_status,
                armed=self.armed,
                focus_hold=self.focus_hold,
            ),
        )

    def _save(self, name):
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
        camera = FrameJoin.camera(pair)
        self.last_frame = clock.now()
        self.frame_cameras.append((captured, camera))
        self.state = replace(self.state, video=pair[0], video_time=captured)
        if camera is not None:
            self._anchor(camera, captured)

    def camera_frame(self, image, captured):
        """From the window, which straightens each decoded frame on the GPU: the frame captured
        at `captured`, strided to the ToF mesh's colour size (worker.frame_size)."""
        loop = self.loop
        if loop is not None:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self._colour, image, captured)

    def _colour(self, image, captured):
        camera = next((c for t, c in reversed(self.frame_cameras) if t == captured), None)
        if self.worker is None or self.failure:
            return
        self.camera_rate.tick()
        self.worker.submit(image, camera, captured)

    def _motion(self, message, peer):
        fresh = []
        for sample in message.samples:
            local = peer.clock.remote_to_local(sample.time)
            if local <= self.motion_time:
                continue
            self.motion_time = local
            fresh.append(
                (local, camera_matrix(sample.camera_pose), sample.angular_velocity, sample.state)
            )
        worker = self.worker
        if fresh and worker is not None and not self.failure:
            worker.motion(fresh)
        return fresh

    def _sensors(self, message, peer):
        """Head IMU and ToF ranges: the mesh's geometry."""
        worker = self.worker
        if worker is None or self.failure:
            return
        local = peer.clock.remote_to_local
        worker.imu(
            [
                (
                    local(s.time),
                    s.orientation,
                    camera_matrix(imu.mount)[:3, :3],
                    s.angular_velocity,
                )
                for imu in message.imu
                for s in imu.samples
                if s.orientation is not None
            ]
        )
        if not message.ranges:
            return
        received = clock.now()
        worker.ranges(
            [
                dict(
                    time=local(grid.time),
                    received=received,
                    mount=camera_matrix(grid.mount),
                    shape=(grid.rows, grid.cols),
                    field_of_view=grid.field_of_view,
                    distances=_targets(grid.distances),
                )
                for grid in message.ranges
            ]
        )

    def _retire(self, worker):
        closer = threading.Thread(target=worker.close, name="ito-reconstruction-close")
        closer.start()
        retiring = [(c, w) for c, w in self.retiring if c.is_alive() or w.process.is_alive()]
        self.retiring = retiring + [(closer, worker)]

    def _anchor(self, camera, captured):
        """Only the camera's position anchors the pilot's eye.

        The pilot's head is the whole view rotation, in the gravity-aligned world anchored
        where the robot started. The robot turns its head and body to look where the pilot
        looks, but those turns never rotate the view: the pilot is never turned by the robot.
        The eye is the average position over the EYE_AVERAGE seconds before the frame
        `captured`, moved on by half its change from the EYE_AVERAGE before.
        """
        self.camera_pose = camera
        kept = self.eye_positions
        if kept and captured < kept[-1][0]:
            kept.clear()  # a new session's clock
        kept.append((captured, camera[:3, 3].copy()))
        while captured - kept[0][0] > 2 * EYE_AVERAGE:
            kept.popleft()
        last = [p for t, p in kept if captured - t <= EYE_AVERAGE]
        before = [p for t, p in kept if captured - t > EYE_AVERAGE]
        eye = sum(last) / len(last)
        if before:
            eye = eye + (eye - sum(before) / len(before)) / 2
        self.state = replace(
            self.state,
            robot_camera=pose(eye),
            robot_heading=math.atan2(float(camera[0, 2]), float(camera[2, 2])),
        )

    async def _session(self, peer):
        self.description = description = peer.description
        if not description.cameras:
            raise ValueError("Driver has no camera")
        camera = description.cameras[0]
        selected = settings.load(self.address, description.name, self.defaults)
        self.settings = settings.Settings.model_validate(selected.model_dump() | self.overrides)
        self.overrides = {}
        self.settings_revision += 1
        paired = peer.credential_received.is_set()
        if paired:
            self.credential = peer.credential
        if self._save(description.name):
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
        self.reconstruction_status = "Starting the ToF mesh"
        self.motion_time = -1.0
        self.restart_at = None
        restart_delay = RESTART_FIRST
        worker_started = 0.0
        self.state = replace(
            self.state,
            flat_video=True,
            video=None,
            video_fov=2 * math.atan(camera.intrinsics.width / (2 * camera.intrinsics.fx)),
            video_intrinsics=camera.intrinsics,
            video_lens=camera.lens,
        )

        def reconstruct():
            nonlocal worker_started
            worker_started = clock.now()
            with self.worker_lock:
                s = self.settings
                depth = None
                if s.mono_depth == "pda":
                    depth = dict(
                        model=s.pda_model, inside=s.pda_inside / 100, outside=s.pda_outside / 100
                    )
                self.worker = TofMesh(
                    camera.intrinsics,
                    s.mesh_memory,
                    capture_delay=camera.capture_delay,
                    depth=depth,
                )

        reconstruct()
        self.connections += 1
        self._status("CONNECTED", "Resume to begin piloting", peer=peer)
        joined = FrameJoin()
        tasks = []
        self.armed = self.focus_hold = self.stalled = False
        previous_fresh = None
        last_fresh = None

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
                elif isinstance(message, Motion):
                    self._motion(message, peer)
                elif isinstance(message, Sensors) and message.camera == camera.name:
                    self._sensors(message, peer)
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
        sent = -math.inf
        self.input_arrived = arrived = asyncio.Event()
        try:
            while not self.stop.is_set():
                now = clock.now()
                if not peer.connected or now - last_status > 2:
                    raise ConnectionError("Driver status lost; input disarmed")
                if self.worker and not self.failure:
                    message = self.worker.status()
                    if message:
                        self.reconstruction_status = message
                    # The flat feed until the first triangles arrive.
                    self.state = replace(self.state, flat_video=not self.worker.has_mesh)
                if self.failure:
                    self.reconstruction_status = self.failure + "; showing flat camera feed"
                    if self.restart_at is not None:
                        wait = max(0, self.restart_at - now)
                        self.reconstruction_status += f"; restarting in {wait:.0f} s"
                    self.state = replace(self.state, flat_video=True)
                if self.failure and self.worker:
                    with self.worker_lock:
                        failed, self.worker = self.worker, None
                    self._retire(failed)
                    if now - worker_started >= RESTART_HEALTHY:
                        restart_delay = RESTART_FIRST
                    self.restart_at = now + restart_delay
                    restart_delay = min(RESTART_LONGEST, restart_delay * 2)
                if self.restart_at is not None and now >= self.restart_at:
                    self.restart_at = self.failure = None
                    self.reconstruction_status = "Restarting the ToF mesh"
                    reconstruct()
                    self.state = replace(self.state, flat_video=True)
                if now - self.last_frame > 5:
                    raise ConnectionError("No synchronized camera frames for five seconds")
                for task in tasks:
                    if task.done():
                        task.result()
                        raise ConnectionError("Camera stream ended")
                while pending or self.commands:
                    if pending is None:
                        given = self.commands.popleft()
                        action = ("stop" if given == "focus_stop" else given).replace("_", "-")
                        if action == "resume":
                            self.armed, self.stalled, last_fresh = True, False, None
                        else:
                            disarm("focus lost" if given == "focus_stop" else action)
                        pending = Command(sequence=command_sequence, action=action)
                        command_sequence += 1
                    if peer.send(pending):
                        pending = None
                    else:
                        break
                value = self.latest_input
                if value is not None:
                    fresh = value.active and now - value.timestamp < INPUT_FRESH
                    if fresh != previous_fresh:
                        diagnostics.event("input_freshness_transition", fresh=fresh)
                        previous_fresh = fresh
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
                    sent = clock.now()
                    sequence += 1
                shown = self.state.status
                if (shown.armed, shown.focus_hold) != (self.armed, self.focus_hold):
                    self.state = replace(
                        self.state,
                        status=replace(shown, armed=self.armed, focus_hold=self.focus_hold),
                    )
                deadline = max(deadline + HEARTBEAT, now)
                if not arrived.is_set():
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(arrived.wait(), max(0, deadline - clock.now()))
                if arrived.is_set():
                    await clock.sleep_until(sent + SEND_INTERVAL)
                    arrived.clear()
                    deadline = clock.now()
        finally:
            self.armed = False
            self.input_arrived = None
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
                async with await connect(
                    self.address,
                    connect_timeout=8,
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
                    self.credential = None
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
        media.receive_h264(media.VideoDecoder)

        # Windows' default (proactor) loop never finishes closing a UDP socket that has a send
        # in flight (CPython gh-156920, still in 3.12): ICE's sockets carry the link's traffic,
        # so leaving a session often waited out peer.TEARDOWN_WAIT (#27). The selector loop
        # closes them at once.
        factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None

        def run():
            with contextlib.suppress(asyncio.CancelledError), clock.fine_timers():
                asyncio.run(self._run(), loop_factory=factory)

        self.thread = threading.Thread(target=run, name="ito-link", daemon=True)
        self.thread.start()
        return self

    def close(self):
        """Bounded: the pilot closed the app, and nothing on the link may keep it open.

        The robot never depends on this: every exit path sends stop first, and a driver
        that hears nothing more neutralizes within its input timeout.
        """
        self.stop.set()
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
        # aiortc's video decoder threads are not daemons and end only in a finished teardown;
        # the interpreter would wait for them forever.
        decoders = [
            t for t in threading.enumerate() if t.name.endswith("-decoder") and not t.daemon
        ]
        if decoders:
            stuck.append("the WebRTC decoder threads")
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
