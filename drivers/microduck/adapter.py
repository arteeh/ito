"""Use the robot's own gaze IK and balance policy; never command individual leg servos."""

import asyncio
import contextlib
import math

from ito import clock
from ito.driver import Adapter
from ito.driver.walking import Walker
from ito.protocol import Camera as CameraDescription
from ito.protocol import DegreeOfFreedom, Pose, RobotDescription

from . import frames
from .camera import Camera, Track
from .remote import Remote

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


def yaw_command(rate):
    """The robot.move yaw command that turns the body at about this rate, rad/s.

    Pollen's velstand stands still for yaw commands below about 1 rad/s (dead zone documented
    in microduck_rl docs/velstand_policy.md); their runtime remap makes it track 0.5-1 rad/s.
    """
    if abs(rate) <= 0.05:
        return 0.0
    return math.copysign(0.33 + abs(rate) / 0.6, rate)


class MicroduckAdapter(Adapter):
    def __init__(self, robot="ws://127.0.0.1:8443", input_timeout=0.25):
        if not robot.startswith(("ws://", "wss://")):
            raise ValueError("--robot must be mediad's ws://host:8443 address")
        if not math.isfinite(input_timeout) or not 0.02 <= input_timeout <= 0.5:
            raise ValueError("Microduck input timeout must be between 0.02 and 0.5 seconds")
        self.remote = Remote(robot, self._notification)
        self.input_timeout = input_timeout
        # The policy's trained forward/backward range is +-0.4 m/s and it stands still below
        # about 0.35. It has no usable sideways step (none up to its trained 0.3 m/s), so
        # sideways input steers the body into that direction instead.
        self.walker = Walker(GAZE_PAN, speed=0.4, lateral_speed=0.0, turn_speed=0.8)
        self.camera = None
        self._description = None
        self._tasks = []
        self._error = None
        self._latest = None
        self._generation = 0
        self._wake = asyncio.Event()
        self._telemetry = {}
        self._state_at = 0.0
        self._yaw_origin = None
        self._origin = None  # Trunk x, y and yaw at startup, in robotd's odometry world.
        self._camera_pose = None
        self._neutral_done = asyncio.Event()

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
                cameras=(
                    CameraDescription(
                        name="head", track_id="microduck-rgb", intrinsics=self.camera.intrinsics
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
            await self.remote.call("robot.subscribe", {"hz": 20})
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
        self._generation += 1
        self._neutral_done.clear()
        self._wake.set()

    def neutral(self):
        self._latest = None
        self._generation += 1
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
            generation, latest = self._generation, self._latest
            if not latest or clock.now() - latest[0] >= self.input_timeout:
                await self.remote.call("robot.stop")
                await self.remote.call("robot.pose", {"active": False})
                # Hold head and beak: releasing a carried object is not a safe neutral.
                self._neutral_done.set()
                continue
            state = latest[1]
            axes, buttons = state.axes, state.buttons
            move = self.walker(state, self._telemetry["base_yaw"])
            # The walking policy needs enough command range to enter its stepping gait.
            commands = [
                (
                    "robot.move",
                    {
                        "vx": move.forward,
                        "vy": move.left,
                        "vyaw": yaw_command(move.turn),
                    },
                )
            ]
            if state.head:
                commands.append(
                    (
                        "robot.look",
                        {
                            "x": 2 * math.cos(move.pan) * math.cos(move.tilt),
                            "y": 2 * math.sin(move.pan) * math.cos(move.tilt),
                            "z": 2 * math.sin(move.tilt),
                            "neck_pitch": 0.0,
                        },
                    )
                )
            commands.extend(
                [
                    (
                        "robot.mouth",
                        {
                            "open": max(
                                0.0, axes.get("right_trigger", 0.0), float(buttons.get("g", False))
                            )
                        },
                    ),
                    (
                        "robot.pose",
                        {
                            "z": -0.025 if buttons.get("c", False) else 0.0,
                            "active": buttons.get("c", False),
                        },
                    ),
                ]
            )
            for method, params in commands:
                # Newest wins, including a stop arriving during an outstanding RPC.
                if generation != self._generation:
                    break
                if clock.now() - latest[0] >= self.input_timeout:
                    self.neutral()
                    break
                await self.remote.call(method, params, deadline=self.input_timeout)
            await asyncio.sleep(0.02)

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
        for key, vector in (data.get("imu") or {}).items():
            for index, value in enumerate(vector):
                values[f"imu_{key}_{index}"] = value
        camera = (data.get("frames") or {}).get("camera")
        if camera and data.get("imu") and position is not None:
            if self._origin is None:
                self._origin = (position[0], position[1], self._yaw_origin)
            self._camera_pose = frames.camera_in_world(
                data["imu"]["quat"], position, (camera["pos"], camera["quat"]), self._origin
            )
            for name, angle in zip(
                ("yaw", "pitch", "roll"), frames.angles(self._camera_pose[1]), strict=True
            ):
                values[f"camera_{name}"] = angle
        self._telemetry.update(values)
        self._state_at = clock.now()

    def _frame_metadata(self):
        """What the robot measured about its camera as a frame arrives.

        The camera pose is robotd's forward kinematics at the measured head joints on the IMU's
        trunk orientation: the heading SLAM cannot see on a plain wall, and the roll and pitch
        that level its map.
        """
        if self._camera_pose is None:
            return {}
        position, orientation = self._camera_pose
        return dict(
            camera_pose=Pose(position=position, orientation=orientation),
            body_yaw=self._telemetry["base_yaw"],
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
