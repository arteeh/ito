"""Use the robot's own gaze IK and balance policy; never command individual leg servos."""

import asyncio
import contextlib
import math
import os
from collections import deque

from ito import clock
from ito.driver import Adapter
from ito.driver.walking import Walker
from ito.protocol import Camera as CameraDescription
from ito.protocol import (
    DegreeOfFreedom,
    Motion,
    MotionNoise,
    MotionSample,
    Pose,
    RobotDescription,
)
from ito.rates import Rate

from . import frames, gaze
from .camera import Camera, Track
from .remote import Remote, RemoteError

# Alpha MJCF travel limits. robot.look applies the robot's own IK and joint limits as well.
HEAD_LIMITS = {
    "neck_pitch": (-math.pi / 2, math.pi / 3),
    "head_pitch": (-math.pi / 2, math.pi / 2),
    "head_yaw": (-math.radians(170), math.radians(170)),
    "head_roll": (-math.radians(25), math.radians(25)),
}
# How far the gaze pans before the body turns. The walking policy was trained with head yaw
# commands within +-1.40 rad (microduck_rl velocity recipe), and robot.look's IK flips the
# neck through a singular pose at +-90 degrees: past that the head flails and rolls 30 degrees.
GAZE_PAN = (-1.40, 1.40)
PAN_LIMIT = 1.50  # The most the pan trim may aim, short of that flip.
# mediad's frames reach the driver about 90 ms after robotd reports the head pose they were
# taken at (encoding, WebRTC, decoding). Measured on the simulated twin by matching
# MASt3R-SLAM's turns to the robot's: taking the pose from receive time instead puts a
# brisk turn's frames 5-10 degrees off and makes SLAM restart its map against the robot.
FRAME_DELAY = 0.09
# The trunk sways about 1.5 degrees at the gait's ~2.4 Hz. Aiming the head from the
# instantaneous heading feeds that sway into the pan command, which the head reaches 0.1-0.3 s
# late, so it adds to the camera's swing instead of cancelling it. The head is aimed from the
# heading averaged over one gait period, moved on by the turn rate over two.
GAIT_PERIOD = 0.42
# robot.look aims the camera at a point; this far out, the camera moving a few centimetres
# as the head turns changes the aim by a fraction of a degree.
GAZE_DISTANCE = 10.0
# Frame metadata keeps the receive-stamped 20 Hz pose it always carried, so the vision-only
# tracker can still be compared against robot-first tracking on the same build. Motion
# carries every 50 Hz tick of robotd's loop, stamped when robotd read its sensors.
METADATA_POSE_INTERVAL = 0.05
# Head FK on the trunk IMU against MuJoCo truth over 800 poses of a walking drive (#33):
# p95 1.32 degrees rotation, 1.77 cm per metre walked, both with receive stamps. One sigma is
# about half a p95; robotd's game-rotation IMU drifts in yaw well under a degree a minute.
# In a turn the pose is off along the turn, as if read a few milliseconds from its t_ns:
# against truth, an interquartile spread of 6-9 ms over the sweep180 and tour drives.
MOTION_NOISE = MotionNoise(
    rotation_deg=0.66,
    rotation_deg_per_s=0.2,
    translation_m=0.005,
    translation_fraction=0.018,
    gravity_deg=0.66,
    time_s=0.006,
)
# robot.look's IK aims with head pitch and yaw and keeps head_roll level. The pilot's head
# roll goes to head_roll, the last joint, turning about the camera's own axis, so it leaves
# the aim alone. Rolls within the dead band are the pilot's posture, not a gesture.
ROLL_DEAD_BAND = math.radians(3)
# A head-IMU sample this close to a robot.state tick turns the camera at that tick.
HEAD_IMU_MATCH = 0.025
# Odometry speeds that count as walking or turning in a motion sample's state.
WALKING_SPEED, TURNING_RATE = 0.02, 0.05


def steady_heading(samples):
    """Body heading now without the gait's sway, from (time, yaw) samples over two periods.

    The mean over the last period cancels the sway but lags a turning body by half a period;
    the turn rate fitted over both periods brings it up to date.
    """
    t0, yaw0 = samples[-1]
    times = [t - t0 for t, _ in samples]
    yaws = [math.remainder(yaw - yaw0, math.tau) for _, yaw in samples]
    count = len(samples)
    mean_t, mean_yaw = sum(times) / count, sum(yaws) / count
    spread = sum((t - mean_t) ** 2 for t in times)
    rate = (
        sum((t - mean_t) * (y - mean_yaw) for t, y in zip(times, yaws, strict=True)) / spread
        if spread > 1e-9
        else 0.0
    )
    recent = [(t, y) for t, y in zip(times, yaws, strict=True) if t >= -GAIT_PERIOD]
    late = sum(t for t, _ in recent) / len(recent)
    return math.remainder(yaw0 + sum(y for _, y in recent) / len(recent) - rate * late, math.tau)


def yaw_command(rate):
    """The robot.move yaw command that turns the body at about this rate, rad/s.

    Velstand v8 (microduck 0.17's policy set) tracks the true command from about 0.5 rad/s up
    (microduck_rl docs/velstand_policy.md). v7's dead zone needed `0.33 + |rate| / 0.6`, which
    makes v8 turn about twice as fast as asked.
    """
    if abs(rate) <= 0.05:
        return 0.0
    return rate


def motion_state(data):
    """What the body is doing this tick, for how far its odometry can be trusted."""
    safety = data.get("safety") or {}
    if safety.get("fallen"):
        return "fallen"
    if safety.get("picked_up"):
        return "held"
    vx, vy, vyaw = (data.get("move") or {}).get("applied") or (0.0, 0.0, 0.0)
    if math.hypot(vx, vy) > WALKING_SPEED:
        return "walking"
    return "turning" if abs(vyaw) > TURNING_RATE else "standing"


class SourceClock:
    """robotd's tick time (CLOCK_MONOTONIC, ns) on the driver's clock.

    On the robot and in the simulator perf_counter is the same CLOCK_MONOTONIC, so the time
    is taken as it is and the link's clock offset carries it to the pilot. A driver running
    elsewhere sees a different clock: the smallest transport delay over the last ten seconds
    then stands in for the offset. A robotd without t_ns gets the arrival time.
    """

    def __init__(self):
        self.shared = None
        self.lags = deque(maxlen=500)

    def __call__(self, t_ns, received):
        if not t_ns:
            return received, "driver receive time"
        source = t_ns / 1e9
        if self.shared is None:
            self.shared = -0.005 <= received - source <= 0.5
        if self.shared:
            return source, "robot t_ns"
        self.lags.append(received - source)
        return source + min(self.lags), "robot t_ns, offset by arrival"


class MicroduckAdapter(Adapter):
    def __init__(self, robot="ws://127.0.0.1:8443", input_timeout=0.25, head_imu=False):
        if not robot.startswith(("ws://", "wss://")):
            raise ValueError("--robot must be mediad's ws://host:8443 address")
        if not math.isfinite(input_timeout) or not 0.02 <= input_timeout <= 0.5:
            raise ValueError("Microduck input timeout must be between 0.02 and 0.5 seconds")
        self.remote = Remote(robot, self._notification)
        self.input_timeout = input_timeout
        self.head_imu = head_imu
        # The policy's trained forward/backward range is +-0.4 m/s and it stands still below
        # about 0.35. It has no usable sideways step (none up to its trained 0.3 m/s), so
        # sideways input steers the body into that direction instead.
        self.walker = Walker(GAZE_PAN, speed=0.4, lateral_speed=0.0, turn_speed=0.8)
        self.camera = None
        self._description = None
        self._tasks = []
        self._error = None
        self._latest = None
        self._wake = asyncio.Event()
        self._telemetry = {}
        self._state_at = 0.0
        self._yaw_origin = None
        self._origin = None  # Trunk x, y and yaw at startup, in robotd's odometry world.
        # Recent measured camera poses: (receive time, position, orientation, body yaw).
        self._poses = deque(maxlen=40)
        self._pose_due = 0.0
        self._source_clock = SourceClock()
        self._motion = deque(maxlen=16)  # Samples not yet sent, oldest first.
        self._previous_motion = None  # (time, xyzw) of the last motion sample.
        self._motion_rate = Rate()
        self._head_gyro = deque(maxlen=8)  # (source time, gyro in sensor axes)
        self._headings = deque()  # (time, body yaw) over the last two gait periods.
        # (time, world up in the trunk frame) over the last gait period, and the camera's
        # position in the trunk: where robot.look's levelled gaze starts.
        self._ups = deque()
        self._camera_in_trunk = None
        self._trim = gaze.TiltTrim()
        # The same learned trim for the pan, standing or turning in place (one table): there
        # robot.look lands a look about 15% short (velstand v8: 51 degrees for 60, 68 for 80;
        # v7 about 10%). Walking, the trunk's yaw sway would be learned as error.
        self._pan_trim = gaze.TiltTrim()
        self._looks = Rate()  # robot.look calls answered, and their round trips.
        self._look_ms = deque(maxlen=50)
        self._measured_at = None
        self._neutral_done = asyncio.Event()
        self._gaze = None  # The joints robot.look chose last, which roll rides on.

    def _exposure(self):
        """The simulator renders a frame at an instant (or averages --blur renders); the
        IMX219's auto-exposure time reaches nobody."""
        if self.camera.source == "sim" and os.environ.get("ITO_MICRODUCK_BLUR", "1") == "1":
            return 0.0
        return None

    @property
    def description(self):
        if self._description is None:
            raise RuntimeError("Microduck has not supplied its camera calibration")
        return self._description

    async def start(self):
        try:
            await self.remote.start()
            # `hello` belongs to updaterd, not robotd. Ask the services we actually use;
            # a simulator or a robot without its updater still has control and video.
            model = await self.remote.call("robot.model")
            if model.get("asset") != "alpha":
                raise RuntimeError("Microduck driver requires the alpha head model")
            self.joint_names = model["joint_names"]
            self.camera = Camera(await self.remote.call("media.video"), self._frame_metadata)
            self._description = RobotDescription(
                name="Microduck",
                motion_noise=MOTION_NOISE,
                cameras=(
                    CameraDescription(
                        name="head",
                        track_id="microduck-rgb",
                        intrinsics=self.camera.intrinsics,
                        # Frames carry their arrival time; mediad dates none at the sensor.
                        capture_delay=FRAME_DELAY,
                        exposure=self._exposure(),
                    ),
                ),
                capabilities=("head", "walking", "posture", "beak"),
                degrees_of_freedom=tuple(
                    DegreeOfFreedom(
                        name=name,
                        unit="radians",
                        minimum=lo,
                        maximum=hi,
                    )
                    for name, (lo, hi) in HEAD_LIMITS.items()
                ),
            )
            self._telemetry.update(
                camera_calibration=self.camera.source, camera_timestamp="driver receive time"
            )
            await self.remote.call("robot.stop")
            if self.head_imu:
                try:
                    await self.remote.call("head_imu.stream")
                except RemoteError as exc:
                    raise RuntimeError(
                        f"Microduck head IMU unavailable ({exc}); enable [head_imu] in tofd "
                        "on the robot, or run without --head-imu"
                    ) from exc
            # Every tick of robotd's 50 Hz loop: robot-first tracking on the pilot PC
            # interpolates the camera's motion between them.
            await self.remote.call("robot.subscribe", {})
            self._tasks = [
                asyncio.create_task(self._guard(work))
                for work in (
                    self.camera.receive(self.remote.video),
                    self._commands(),
                    self._health(),
                )
            ]
            async with asyncio.timeout(5):
                while not self.camera.latest or not self._state_at:
                    self._check()
                    await asyncio.sleep(0.02)
        except Exception:
            await self.close()
            raise

    def _check(self):
        if self._error:
            raise RuntimeError(f"Microduck fault: {self._error}")
        self.remote.check()

    async def _guard(self, work):
        try:
            await work
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._error = exc
            self.neutral()
            if self.camera:
                self.camera.close()
            with contextlib.suppress(Exception):
                await self.remote.call("robot.stop")

    def media_tracks(self):
        self._check()
        return [Track(self.camera, self.publish_frame)]

    def apply(self, state):
        self._check()
        if clock.now() - self._state_at > 1:
            raise RuntimeError("Microduck telemetry stalled")
        if not state.deadman:
            self.neutral()
            return
        self._latest = (clock.now(), state)
        self._neutral_done.clear()
        self._wake.set()

    def neutral(self):
        self._latest = None
        self._wake.set()

    async def _commands(self):
        while True:
            try:
                async with asyncio.timeout(self.input_timeout):
                    await self._wake.wait()
            except TimeoutError:
                if self._neutral_done.is_set():
                    continue
                self._latest = None
            self._wake.clear()
            latest = self._latest
            if not latest or clock.now() - latest[0] >= self.input_timeout:
                await self.remote.call("robot.stop")
                await self.remote.call("robot.pose", {"active": False})
                # Hold head and beak: releasing a carried object is not a safe neutral.
                self._trim.release()
                self._pan_trim.release()
                self._neutral_done.set()
                continue
            state = latest[1]
            axes, buttons = state.axes, state.buttons
            move = self.walker(state, steady_heading(self._headings))
            # The walking policy needs enough command range to enter its stepping gait.
            # Continuous intents go as notifications: robotd applies the newest on its next
            # tick, so waiting for an answer to each would only hold the next pose back.
            vyaw = yaw_command(move.turn)
            self.remote.notify("robot.move", {"vx": move.forward, "vy": move.left, "vyaw": vyaw})
            beak = max(0.0, axes.get("right_trigger", 0.0), float(buttons.get("g", False)))
            self.remote.notify("robot.mouth", {"open": beak})
            crouch = buttons.get("c", False)
            self.remote.notify("robot.pose", {"z": -0.025 if crouch else 0.0, "active": crouch})
            if not state.head:
                self._trim.release()
                self._pan_trim.release()
                continue
            # Levelled by the trunk's lean averaged over a gait period: chasing the gait's
            # pitch sway with a head that lags it would add to the camera's swing. A robotd
            # without IMU and head frames in robot.state gets the trunk's own axes.
            up = (
                tuple(sum(u[i] for _, u in self._ups) for i in range(3))
                if self._ups
                else (0.0, 0.0, 1.0)
            )
            gait = gaze.gait(move.forward, move.left, vyaw)
            aim = self._trim(move.tilt, clock.now(), gait)
            # Short of the IK's singular pose at +-90 degrees, where the head flails.
            in_place = "standing" if gait in ("standing", "turning") else gait
            pan = self._pan_trim(move.pan, clock.now(), in_place)
            pan = min(PAN_LIMIT, max(-PAN_LIMIT, pan))
            x, y, z = gaze.target(
                pan,
                aim,
                up,
                self._camera_in_trunk or (0.0, 0.0, 0.0),
                GAZE_DISTANCE,
            )
            # robot.look answers with the IK's joints; one in flight at a time, always the
            # newest gaze. A stop arriving meanwhile is sent as soon as it answers.
            # robot.look levels head_roll, so a rolled pilot's robot.head goes right behind
            # it, on the joints the last look chose: robotd applies both in one go, never
            # a level head between them, and the aim trails by one look at most.
            low, high = HEAD_LIMITS["head_roll"]
            excess = abs(move.roll) - ROLL_DEAD_BAND
            roll = min(high, max(low, math.copysign(excess, move.roll))) if excess > 0 else 0.0
            rolled = ()
            if roll and self._gaze is not None:
                rolled = (("robot.head", self._gaze | {"head_roll": roll}),)
            sent = clock.now()
            answer = await self.remote.call(
                "robot.look",
                {"x": x, "y": y, "z": z, "neck_pitch": 0.0},
                deadline=self.input_timeout,
                then=rolled,
            )
            head = answer.get("head") if isinstance(answer, dict) else None
            if isinstance(head, dict) and set(head) >= set(HEAD_LIMITS):
                self._gaze = {name: float(head[name]) for name in HEAD_LIMITS}
            self._looks.tick()
            self._look_ms.append((clock.now() - sent) * 1000)
            # The gait and the tilt aimed at, trim included, show how the trim follows the
            # gait's head droop (#28) next to the measured camera, neck and head pitch.
            self._telemetry.update(
                look_hz=self._looks.hz(),
                look_ms=sorted(self._look_ms)[len(self._look_ms) // 2],
                gait=gait,
                pilot_tilt=move.tilt,
                aimed_tilt=aim,
                pilot_pan=move.pan,
                aimed_pan=pan,
                pilot_roll=move.roll,
                aimed_roll=roll if rolled else 0.0,
            )

    async def _health(self):
        while True:
            health = await self.remote.call("robot.health")
            self._telemetry["healthy"] = health["healthy"]
            self._telemetry["health_reason"] = str(health.get("reason") or "")[:256]
            for name in ("battery", "motors"):
                for key, value in (health.get(name) or {}).items():
                    if type(value) in (float, int, bool):
                        self._telemetry[f"{name}_{key}"] = value
            await asyncio.sleep(1)
            if self._state_at and clock.now() - self._state_at > 1:
                raise RuntimeError("Microduck telemetry stalled")

    def _notification(self, method, data):
        if method == "head_imu.frame":
            source, _ = self._source_clock(data.get("t_ns"), clock.now())
            self._head_gyro.append((source, tuple(data["gyro"])))
            return
        if method != "robot.state":
            return
        values = {"policy": data["policy"]}
        for name, angle in zip(self.joint_names, data["joints"], strict=True):
            if name in HEAD_LIMITS or name == "mouth":
                values[name] = angle
        for name, target in zip(self.joint_names, data["targets"], strict=True):
            if name in HEAD_LIMITS or name == "mouth":
                values[f"{name}_target"] = target
        for prefix, vector in data["move"].items():
            if prefix in {"requested", "applied"}:
                values.update(
                    {
                        f"{prefix}_{axis}": v
                        for axis, v in zip(("vx", "vy", "vyaw"), vector, strict=True)
                    }
                )
        for key in ("fallen", "limp", "picked_up"):
            if key in data["safety"]:
                values[key] = data["safety"][key]
        position = data.get("odom", {}).get("position")
        if position is not None:
            values.update({f"base_{axis}": v for axis, v in zip("xyz", position, strict=True)})
        for key, value in data.get("odom", {}).items():
            if type(value) in (float, int, bool):
                values[f"base_{key}"] = value
        # robotd's contact odometry supplies IMU heading; keep Ito's startup reference.
        yaw = data["odom"]["yaw"]
        if self._yaw_origin is None:
            self._yaw_origin = yaw
        values["base_yaw"] = math.remainder(yaw - self._yaw_origin, 2 * math.pi)
        now = clock.now()
        self._headings.append((now, values["base_yaw"]))
        while now - self._headings[0][0] > 2 * GAIT_PERIOD:
            self._headings.popleft()
        for key, vector in (data.get("imu") or {}).items():
            for index, value in enumerate(vector):
                values[f"imu_{key}_{index}"] = value
        camera = (data.get("frames") or {}).get("camera")
        if camera and data.get("imu"):
            self._camera_in_trunk = tuple(camera["pos"])
            self._ups.append((now, gaze.up_in_trunk(data["imu"]["quat"])))
            while now - self._ups[0][0] > GAIT_PERIOD:
                self._ups.popleft()
        if camera and data.get("imu") and position is not None:
            if self._origin is None:
                self._origin = (position[0], position[1], self._yaw_origin)
            pose = frames.camera_in_world(
                data["imu"]["quat"], position, (camera["pos"], camera["quat"]), self._origin
            )
            if now >= self._pose_due:
                self._pose_due = max(self._pose_due + METADATA_POSE_INTERVAL, now - 0.05)
                self._poses.append((now, *pose, values["base_yaw"]))
            self._moved(data, now, pose, camera)
            for name, angle in zip(("yaw", "pitch", "roll"), frames.angles(pose[1]), strict=True):
                values[f"camera_{name}"] = angle
            if self._measured_at is not None:
                self._trim.measured(values["camera_pitch"], now, now - self._measured_at)
                if self._pan_trim.gait == "standing":
                    # The camera's pan off the trunk's heading, as the pilot's pan is measured.
                    pan = math.remainder(values["camera_yaw"] - values["base_yaw"], math.tau)
                    self._pan_trim.measured(pan, now, now - self._measured_at)
            self._measured_at = now
        self._telemetry.update(values)
        self._state_at = clock.now()

    def _moved(self, data, received, pose, camera):
        """Queue this tick's camera pose for the pilot, stamped when robotd read its sensors."""
        source, stamp = self._source_clock(data.get("t_ns"), received)
        previous = self._previous_motion
        if previous is not None and source <= previous[0]:
            return
        gyro = None
        mount = (data.get("frames") or {}).get("head_imu") if self.head_imu else None
        if self._head_gyro and mount:
            when, reading = min(self._head_gyro, key=lambda g: abs(g[0] - source))
            if abs(when - source) <= HEAD_IMU_MATCH:
                gyro = frames.gyro_in_camera(reading, mount["quat"], camera["quat"])
        gyro_from_fk = gyro is None
        if gyro_from_fk:
            gyro = (
                frames.angular_velocity(previous[1], pose[1], source - previous[0])
                if previous is not None
                else (0.0, 0.0, 0.0)
            )
        self._previous_motion = (source, pose[1])
        self._motion.append(
            MotionSample(
                time=source,
                camera_pose=Pose(position=pose[0], orientation=pose[1]),
                angular_velocity=gyro,
                state=motion_state(data),
            )
        )
        if self.publish_motion(Motion(samples=tuple(self._motion))):
            self._motion.clear()
        self._motion_rate.tick()
        self._telemetry.update(
            motion_hz=self._motion_rate.hz(),
            motion_clock=stamp,
            motion_gyro="head FK" if gyro_from_fk else "head IMU",
        )

    def _frame_metadata(self):
        """What the robot measured about its camera when the frame arriving now was taken.

        The camera pose is robotd's forward kinematics at the measured head joints on the IMU's
        trunk orientation: the heading SLAM cannot see on a plain wall, and the roll and pitch
        that level its map.
        """
        if not self._poses:
            return {}
        position, orientation, body_yaw = frames.at(self._poses, clock.now() - FRAME_DELAY)
        return dict(camera_pose=Pose(position=position, orientation=orientation), body_yaw=body_yaw)

    def telemetry(self):
        self._check()
        return self._telemetry.copy()

    async def close(self):
        self.neutral()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        with contextlib.suppress(Exception):
            await self.remote.call("robot.stop")
            await self.remote.call("robot.pose", {"active": False})
        if self.camera:
            self.camera.close()
        await self.remote.close()
