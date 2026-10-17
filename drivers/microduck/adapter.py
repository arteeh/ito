import asyncio
import contextlib
import logging
import math
import os
from collections import deque

import numpy as np

from ito import clock
from ito.driver import Adapter
from ito.driver.walking import Walker
from ito.protocol import Camera as CameraDescription
from ito.protocol import (
    DegreeOfFreedom,
    Depth,
    Motion,
    MotionNoise,
    MotionSample,
    Pose,
    RobotDescription,
    Sensors,
    TruePose,
)
from ito.rates import Rate

from . import frames, gaze, sensors, truth
from .camera import Camera, Track
from .remote import Remote, RemoteError

log = logging.getLogger(__name__)

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
PAN_LIMIT = 1.50
# mediad's frames reach the driver about 90 ms after robotd reports the head pose they were
# taken at (encoding, WebRTC, decoding). Measured on the simulated twin by matching
# MASt3R-SLAM's turns to the robot's: taking the pose from receive time instead puts a
# brisk turn's frames 5-10 degrees off and makes SLAM restart its map against the robot.
FRAME_DELAY = 0.09
GAIT_PERIOD = 0.42
GAZE_DISTANCE = 10.0
METADATA_POSE_INTERVAL = 0.05
MOTION_NOISE = MotionNoise(
    rotation_deg=0.66,
    rotation_deg_per_s=0.2,
    translation_m=0.005,
    translation_fraction=0.018,
    gravity_deg=0.66,
    time_s=0.006,
)
ROLL_DEAD_BAND = math.radians(3)
# robotd's idle look-around starts 2 s after the last head intent; hold the head well inside it.
HEAD_HOLD = 0.5
HEAD_IMU_MATCH = 0.025
WALKING_SPEED, TURNING_RATE = 0.02, 0.05


def steady_heading(samples):
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
    if abs(rate) <= 0.05:
        return 0.0
    return rate


def motion_state(data):
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
    def __init__(self, robot="ws://127.0.0.1:8443", input_timeout=0.25, head_imu=None, tof=True):
        if not robot.startswith(("ws://", "wss://")):
            raise ValueError("--robot must be mediad's ws://host:8443 address")
        if not math.isfinite(input_timeout) or not 0.02 <= input_timeout <= 0.5:
            raise ValueError("Microduck input timeout must be between 0.02 and 0.5 seconds")
        self.remote = Remote(robot, self._notification)
        self.input_timeout = input_timeout
        self.head_imu = head_imu
        self.tof = tof
        self._imu_chip = None
        self._tof_chip = None
        self._tof_fov = None
        self._imu_pending = deque(maxlen=64)
        self._ranges_pending = deque(maxlen=8)
        self._imu_rate = Rate()
        self._tof_rate = Rate()
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
        self._origin = None
        self._poses = deque(maxlen=40)
        self._pose_due = 0.0
        self._source_clock = SourceClock()
        self._motion = deque(maxlen=16)
        self._previous_motion = None
        self._motion_rate = Rate()
        self._head_gyro = deque(maxlen=8)
        self._headings = deque()
        self._ups = deque()
        self._camera_in_trunk = None
        self._trim = gaze.TiltTrim()
        self._pan_trim = gaze.TiltTrim()
        self._looks = Rate()
        self._look_ms = deque(maxlen=50)
        self._measured_at = None
        self._neutral_done = asyncio.Event()
        self._gaze = None
        self._head_at = 0.0
        self._held = None

    def _exposure(self):
        told = os.environ.get("ITO_MICRODUCK_EXPOSURE")
        if told:
            return float(told)
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
            self._tof_fov = math.radians(model.get("tof_fov_deg") or 45.0)
            address = os.environ.get("ITO_MICRODUCK_TRUTH")
            self._truth = truth.SimTruth(address) if address else None
            self.camera = Camera(
                await self.remote.call("media.video"),
                self._frame_metadata,
                truth=self._true_pose if self._truth else None,
            )
            self._description = RobotDescription(
                name="Microduck",
                motion_noise=MOTION_NOISE,
                cameras=(
                    CameraDescription(
                        name="head",
                        track_id="microduck-rgb",
                        intrinsics=self.camera.intrinsics,
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
            if self.head_imu is not False:
                self._imu_chip = await self._stream("head_imu.stream", required=self.head_imu)
            if self.tof:
                self._tof_chip = await self._stream("tof.stream")
            self._telemetry.update(head_imu=self._imu_chip or "none", tof=self._tof_chip or "none")
            await self.remote.call("robot.subscribe", {})
            self._tasks = [
                asyncio.create_task(self._guard(work))
                for work in (
                    self.camera.receive(self.remote.video),
                    self._commands(),
                    self._hold_head(),
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

    async def _stream(self, method, required=False):
        try:
            answer = await self.remote.call(method)
        except RemoteError as exc:
            answer = {"unavailable": str(exc)}
        chip = answer.get("sensor")
        if chip:
            log.info("Microduck %s: %s at %s Hz", method, chip, answer.get("hz"))
            return chip
        reason = answer.get("unavailable") or "no sensor fitted"
        log.info("Microduck %s unavailable, carrying on without it: %s", method, reason)
        if required:
            raise RuntimeError(f"Microduck {method} unavailable: {reason}")
        self._telemetry[f"{method.split('.')[0]}_unavailable"] = str(reason)[:256]
        return None

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
            up = (
                tuple(sum(u[i] for _, u in self._ups) for i in range(3))
                if self._ups
                else (0.0, 0.0, 1.0)
            )
            gait = gaze.gait(move.forward, move.left, vyaw)
            aim = self._trim(move.tilt, clock.now(), gait)
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
            low, high = HEAD_LIMITS["head_roll"]
            excess = abs(move.roll) - ROLL_DEAD_BAND
            roll = min(high, max(low, math.copysign(excess, move.roll))) if excess > 0 else 0.0
            rolled = ()
            if roll and self._gaze is not None:
                rolled = (("robot.head", self._gaze | {"head_roll": roll}),)
            sent = clock.now()
            self._head_at = sent
            answer = await self.remote.call(
                "robot.look",
                {"x": x, "y": y, "z": z, "neck_pitch": 0.0},
                deadline=self.input_timeout,
                then=rolled,
            )
            head = answer.get("head") if isinstance(answer, dict) else None
            if isinstance(head, dict) and set(head) >= set(HEAD_LIMITS):
                self._gaze = {name: float(head[name]) for name in HEAD_LIMITS}
            if self._gaze is not None:
                self._held = self._gaze | ({"head_roll": roll} if rolled else {})
            self._looks.tick()
            self._look_ms.append((clock.now() - sent) * 1000)
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

    async def _hold_head(self):
        """Keep the head where the pilot left it.

        robotd (daemon 0.17, robotd/src/idle_head.rs) takes the head over for an idle
        look-around (glances, gaze nudges, yawns) once the robot stands still and no head
        intent has arrived for 2 s. Ito sends looks only while the pilot holds the deadman,
        so a pilot who stops, lets go or has not moved yet saw the duck look around on its
        own. Re-sending the last aim well inside those 2 s keeps the camera where the
        pilot pointed it; before the first look that is the head's straight-ahead pose.
        """
        while True:
            await asyncio.sleep(HEAD_HOLD / 2)
            if clock.now() - self._head_at < HEAD_HOLD:
                continue
            held = self._held or dict.fromkeys(HEAD_LIMITS, 0.0)
            self._head_at = clock.now()
            self.remote.notify("robot.head", held)

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
            self._imu_pending.append((source, data))
            self._imu_rate.tick()
            return
        if method == "tof.frame":
            source, _ = self._source_clock(data.get("t_ns"), clock.now())
            self._ranges_pending.append((source, data))
            self._tof_rate.tick()
            return
        if method != "robot.state":
            return
        self._publish_sensors(data.get("frames") or {})
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
                    pan = math.remainder(values["camera_yaw"] - values["base_yaw"], math.tau)
                    self._pan_trim.measured(pan, now, now - self._measured_at)
            self._measured_at = now
        self._telemetry.update(values)
        self._state_at = clock.now()

    def _publish_sensors(self, mounts):
        camera = mounts.get("camera")
        if not camera or not (self._imu_pending or self._ranges_pending):
            return
        imu, grids = (), ()
        if self._imu_pending and mounts.get("head_imu"):
            imu = (
                sensors.imu(self._imu_chip, list(self._imu_pending), mounts["head_imu"], camera),
            )
        if self._ranges_pending and mounts.get("tof"):
            grids = tuple(
                sensors.ranges(self._tof_chip, when, frame, mounts["tof"], camera, self._tof_fov)
                for when, frame in self._ranges_pending
            )
        if not imu and not grids:
            return
        if self.publish_sensors(Sensors(camera="head", imu=imu, ranges=grids)):
            self._imu_pending.clear()
            self._ranges_pending.clear()
        self._telemetry.update(head_imu_hz=self._imu_rate.hz(), tof_hz=self._tof_rate.hz())

    def _moved(self, data, received, pose, camera):
        source, stamp = self._source_clock(data.get("t_ns"), received)
        previous = self._previous_motion
        if previous is not None and source <= previous[0]:
            return
        gyro = None
        mount = (data.get("frames") or {}).get("head_imu") if self._imu_chip else None
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
        if not self._poses:
            return {}
        position, orientation, body_yaw = frames.at(self._poses, clock.now() - FRAME_DELAY)
        return dict(camera_pose=Pose(position=position, orientation=orientation), body_yaw=body_yaw)

    def _true_pose(self, rgb, rtp):
        row = self._truth.match(rgb, rtp)
        if row is None or self._origin is None:
            return {}
        position, orientation = truth.in_world(row, self._origin)
        depth = None
        if "depth" in row:
            height, width = row["depth_shape"]
            sent = Depth(width=width, height=height, data=row["depth"])
            mm = np.frombuffer(sent.to_bytes(), "<u2").reshape(height, width)
            depth = Depth.from_bytes(
                width, height, self.camera.rectify_depth(mm).astype("<u2").tobytes()
            )
        return dict(
            true_pose=TruePose(
                time=row["time"],
                camera_pose=Pose(position=position, orientation=orientation),
                depth=depth,
            )
        )

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
