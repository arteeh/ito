"""Use the robot's own gaze IK and balance policy; never command individual leg servos."""

import asyncio
import contextlib
import math
import time

from ito.driver import Adapter
from ito.protocol import Camera as CameraDescription
from ito.protocol import DegreeOfFreedom, RobotDescription

from .camera import Camera, Track
from .remote import Remote

# Alpha MJCF travel limits. robot.look applies the robot's own IK and joint limits as well.
HEAD_LIMITS = {
    "neck_pitch": (-math.pi / 2, math.pi / 3),
    "head_pitch": (-math.pi / 2, math.pi / 2),
    "head_yaw": (-math.radians(170), math.radians(170)),
    "head_roll": (-math.radians(25), math.radians(25)),
}


class MicroduckAdapter(Adapter):
    def __init__(self, robot="ws://127.0.0.1:8443", input_timeout=0.25):
        if not robot.startswith(("ws://", "wss://")):
            raise ValueError("--robot must be mediad's ws://host:8443 address")
        if not math.isfinite(input_timeout) or not 0.02 <= input_timeout <= 0.5:
            raise ValueError("Microduck input timeout must be between 0.02 and 0.5 seconds")
        self.remote = Remote(robot, self._notification)
        self.input_timeout = input_timeout
        self.camera = None
        self._description = None
        self._tasks = []
        self._error = None
        self._latest = None
        self._generation = 0
        self._wake = asyncio.Event()
        self._telemetry = {}
        self._state_at = 0.0
        self._neutral_done = asyncio.Event()

    @property
    def description(self):
        if self._description is None:
            raise RuntimeError("Microduck has not supplied its camera calibration")
        return self._description

    async def start(self):
        try:
            await self.remote.start()
            hello = await self.remote.call("hello", {"api_version": 40})
            if hello.get("api_version", 0) < 25:
                raise RuntimeError(
                    "Microduck firmware must support camera geometry and robot.model"
                )
            model = await self.remote.call("robot.model")
            if model.get("asset") != "alpha":
                raise RuntimeError("Microduck driver requires the alpha head model")
            self.joint_names = model["joint_names"]
            self.camera = Camera(await self.remote.call("media.video"))
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
        if time.monotonic() - self._state_at > 1:
            raise RuntimeError("Microduck telemetry stalled")
        if not state.deadman:
            self.neutral()
            return
        self._latest = (time.monotonic(), state)
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
            if not latest or time.monotonic() - latest[0] >= self.input_timeout:
                await self.remote.call("robot.stop")
                await self.remote.call("robot.pose", {"active": False})
                # Hold head and beak: releasing a carried object is not a safe neutral.
                self._neutral_done.set()
                continue
            state = latest[1]
            axes, buttons = state.axes, state.buttons
            commands = [
                (
                    "robot.move",
                    {
                        "vx": axes.get("move_y", 0.0) * 0.15,
                        "vy": -axes.get("strafe", 0.0) * 0.10,
                        "vyaw": -axes.get("move_x", 0.0) * 0.6,
                    },
                )
            ]
            if state.head:
                x, y, z, w = state.head.orientation
                norm = math.sqrt(x * x + y * y + z * z + w * w)
                x, y, z, w = (v / norm for v in (x, y, z, w))
                # Ito -Z forward/+Y up -> trunk +X forward/+Y left/+Z up.
                commands.append(
                    (
                        "robot.look",
                        {
                            "x": 2 * (1 - 2 * (x * x + y * y)),
                            "y": 4 * (x * z + y * w),
                            "z": 4 * (x * w - y * z),
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
                if time.monotonic() - latest[0] >= self.input_timeout:
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
            if self._state_at and time.monotonic() - self._state_at > 1:
                raise RuntimeError("Microduck telemetry stalled")

    def _notification(self, method, data):
        if method != "robot.state":
            return
        values = {"policy": data["policy"]}
        for name, angle in zip(self.joint_names, data["joints"], strict=True):
            if name in HEAD_LIMITS:
                values[name] = angle
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
        for key, vector in (data.get("imu") or {}).items():
            for index, value in enumerate(vector):
                values[f"imu_{key}_{index}"] = value
        self._telemetry.update(values)
        self._state_at = time.monotonic()

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
