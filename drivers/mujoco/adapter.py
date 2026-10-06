"""MuJoCo physics has its own deadman; GL rendering never holds the simulation lock."""

import asyncio
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ito.driver import Adapter
from ito.protocol import Camera, DegreeOfFreedom, Intrinsics, PilotState, Pose, RobotDescription

from .camera import CameraTrack

ROOM = Path(__file__).with_name("assets") / "room.xml"
# MuJoCo robot axes (+X forward, +Y left, +Z up) to Ito (+X right, +Y up, -Z forward).
ITO_FROM_MJ = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])


class MujocoAdapter(Adapter):
    def __init__(
        self,
        model: str = str(ROOM),
        camera: str = "head",
        base: str = "base",
        pan: str = "head_pan",
        tilt: str = "head_tilt",
        left: str = "left_drive",
        right: str = "right_drive",
        width: int = 320,
        height: int = 240,
        fps: float = 30,
        wheel_radius: float = 0.14,
        axle_width: float = 0.52,
        speed: float = 0.7,
        turn_speed: float = 1.2,
        input_timeout: float = 0.25,
        gl: str | None = None,
        rgb_only: bool = False,
    ):
        for name, value, low, high in (
            ("fps", fps, 1, 60),
            ("wheel_radius", wheel_radius, 0.001, 10),
            ("axle_width", axle_width, 0.001, 20),
            ("speed", speed, 0.001, 10),
            ("turn_speed", turn_speed, 0.001, 10),
            ("input_timeout", input_timeout, 0.02, 5),
        ):
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be between {low} and {high}")
        if not (16 <= width <= 640 and 16 <= height <= 480 and width % 2 == height % 2 == 0):
            raise ValueError("video size must be even, between 16x16 and 640x480")
        backend = gl or os.environ.get("MUJOCO_GL", "osmesa" if sys.platform == "linux" else "glfw")
        if backend not in {"osmesa", "egl", "glfw"}:
            raise ValueError("MUJOCO_GL must be osmesa, egl, or glfw")
        if "mujoco" in sys.modules and os.environ.get("MUJOCO_GL") != backend:
            raise ValueError("select MUJOCO_GL before importing mujoco")
        os.environ["MUJOCO_GL"] = backend
        try:
            import mujoco
        except (ImportError, AttributeError, OSError) as exc:
            raise RuntimeError(
                f"MuJoCo {backend} unavailable: install libosmesa6 for software rendering "
                "or use --gl egl with an EGL driver"
            ) from exc
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(Path(model).resolve()))
        self.model.vis.global_.offwidth = max(width, self.model.vis.global_.offwidth)
        self.model.vis.global_.offheight = max(height, self.model.vis.global_.offheight)
        if not 0.0001 <= self.model.opt.timestep <= 0.01:
            raise ValueError("MJCF timestep must be between 0.0001 and 0.01 seconds")
        self.data = mujoco.MjData(self.model)
        self.camera_id = self.model.camera(camera).id
        self.base_id = self.model.body(base).id
        if self.base_id == 0:
            raise ValueError("base must be a robot body, not the world")
        if self.model.cam_mode[self.camera_id] != mujoco.mjtCamLight.mjCAMLIGHT_FIXED:
            raise ValueError("camera must be fixed to the robot")
        projection = getattr(self.model, "cam_projection", None)
        if projection is None:
            projection = self.model.cam_orthographic
        if projection[self.camera_id]:
            raise ValueError("camera must use perspective projection")
        self.head = [self._actuator(name, position=True) for name in (pan, tilt)]
        self.wheels = [self._actuator(name, position=False) for name in (left, right)]
        if len({binding[0] for binding in self.head + self.wheels}) != 4:
            raise ValueError("head and wheel actuators must be distinct")
        self.width, self.height, self.fps = width, height, fps
        self.wheel_radius, self.axle_width = wheel_radius, axle_width
        self.speed, self.turn_speed, self.input_timeout = speed, turn_speed, input_timeout
        self.rgb_only = rgb_only
        self.camera_name = camera
        self.backend = backend
        self._command: tuple[float, PilotState] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._renderer = None
        self._fault: BaseException | None = None
        self._telemetry: dict = {}
        self._tracks: list[CameraTrack] = []
        self._captured = time.monotonic()
        mujoco.mj_forward(self.model, self.data)
        self._origin = self.data.xpos[self.base_id].copy()
        self._world_rotation = ITO_FROM_MJ @ self.data.xmat[self.base_id].reshape(3, 3).T
        sensor = self.model.cam_sensorsize[self.camera_id]
        intrinsic = self.model.cam_intrinsic[self.camera_id]
        if sensor[0] > 0:
            fx, fy = intrinsic[:2] / sensor * [width, height]
            # MuJoCo principal point is measured from sensor center, with +Y up.
            cx = width / 2 - intrinsic[2] / sensor[0] * width
            cy = height / 2 + intrinsic[3] / sensor[1] * height
        else:
            fx = fy = height / (
                2 * math.tan(math.radians(float(self.model.cam_fovy[self.camera_id])) / 2)
            )
            cx, cy = width / 2, height / 2
        self._description = RobotDescription(
            name=Path(model).stem,
            cameras=(
                Camera(
                    name=camera,
                    track_id=f"{camera}-video",
                    intrinsics=Intrinsics(
                        width=width,
                        height=height,
                        fx=float(fx),
                        fy=float(fy),
                        cx=float(cx),
                        cy=float(cy),
                    ),
                    extrinsics=self.camera_pose(self.data),
                ),
            ),
            capabilities=("head-pan-tilt", "differential-drive")
            + (() if rgb_only else ("depth", "camera-pose")),
            degrees_of_freedom=tuple(
                DegreeOfFreedom(
                    name=self.model.joint(joint).name, unit="radians", minimum=low, maximum=high
                )
                for _, joint, low, high in self.head
            ),
        )

    def _actuator(self, name, *, position):
        mj, model = self.mj, self.model
        actuator = model.actuator(name)
        index = actuator.id
        if model.actuator_trntype[index] != mj.mjtTrn.mjTRN_JOINT:
            raise ValueError(f"{name}: expected a joint servo")
        joint = int(model.actuator_trnid[index, 0])
        if model.jnt_type[joint] != mj.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"{name}: expected a hinge joint")
        gain = model.actuator_gainprm[index, 0]
        bias = model.actuator_biasprm[index]
        if (
            model.actuator_dyntype[index] != mj.mjtDyn.mjDYN_NONE
            or model.actuator_gaintype[index] != mj.mjtGain.mjGAIN_FIXED
            or model.actuator_biastype[index] != mj.mjtBias.mjBIAS_AFFINE
            or not np.array_equal(model.actuator_gear[index], [1, 0, 0, 0, 0, 0])
            or gain <= 0
            or bias[0] != 0
            or (position and (bias[1] != -gain or bias[2] > 0))
            or (not position and (bias[1] != 0 or bias[2] != -gain))
        ):
            kind = "position" if position else "velocity"
            raise ValueError(f"{name}: expected a {kind} servo with unit gear")
        if position:
            if not model.jnt_limited[joint]:
                raise ValueError(f"{name}: head hinge must have joint limits")
            low, high = map(float, model.jnt_range[joint])
        else:
            low, high = -math.inf, math.inf
        if model.actuator_ctrllimited[index]:
            low = max(low, float(model.actuator_ctrlrange[index, 0]))
            high = min(high, float(model.actuator_ctrlrange[index, 1]))
        if low >= high or not low <= 0 <= high:
            raise ValueError(f"{name}: servo limits must include zero")
        return index, joint, low, high

    @property
    def description(self):
        return self._description

    def camera_pose(self, data):
        rotation = self._world_rotation @ data.cam_xmat[self.camera_id].reshape(3, 3)
        quat = np.empty(4)
        self.mj.mju_mat2Quat(quat, rotation.ravel())
        position = self._world_rotation @ (data.cam_xpos[self.camera_id] - self._origin)
        return Pose(
            position=tuple(map(float, position)), orientation=tuple(map(float, quat[[1, 2, 3, 0]]))
        )

    def apply(self, state):
        self._check_fault()
        self._command = (time.monotonic(), state) if state.deadman else None

    def neutral(self):
        self._command = None

    def _check_fault(self):
        if self._fault:
            raise RuntimeError(f"MuJoCo worker failed: {self._fault}") from self._fault

    def telemetry(self):
        self._check_fault()
        return self._telemetry.copy()

    def media_tracks(self):
        self._check_fault()
        self._tracks = [track for track in self._tracks if track.readyState == "live"]
        track = CameraTrack(self)
        self._tracks.append(track)
        return [track]

    def _controls(self, state, was_active):
        data, model = self.data, self.model
        if state is None:
            if was_active:
                for index, joint, low, high in self.head:
                    data.ctrl[index] = np.clip(data.qpos[model.jnt_qposadr[joint]], low, high)
            for index, *_ in self.wheels:
                data.ctrl[index] = 0
            return
        if state.head:
            x, y, z, w = np.array(state.head.orientation) / np.linalg.norm(state.head.orientation)
            forward = np.array(
                [-2 * (x * z + y * w), 2 * (x * w - y * z), -(1 - 2 * (x * x + y * y))]
            )
            angles = (
                math.atan2(-forward[0], -forward[2]),
                math.asin(float(np.clip(forward[1], -1, 1))),
            )
            for (index, _, low, high), angle in zip(self.head, angles, strict=True):
                data.ctrl[index] = np.clip(angle, low, high)
        forward = state.axes.get("move_y", 0.0) * self.speed
        # Stick right turns right; positive MuJoCo yaw turns left.
        turn = -state.axes.get("move_x", 0.0) * self.turn_speed
        velocities = (
            (forward - turn * self.axle_width / 2) / self.wheel_radius,
            (forward + turn * self.axle_width / 2) / self.wheel_radius,
        )
        for (index, _, low, high), velocity in zip(self.wheels, velocities, strict=True):
            data.ctrl[index] = np.clip(velocity, low, high)

    def _simulate(self):
        active = True
        deadline = time.monotonic()
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                command = self._command
                state = command[1] if command and now - command[0] < self.input_timeout else None
                with self._lock:
                    self._controls(state, active)
                    self.mj.mj_step(self.model, self.data)
                    self._captured = time.monotonic()
                    if not np.isfinite(self.data.qpos).all() or any(self.data.warning.number):
                        raise RuntimeError("unstable MJCF simulation; check model dynamics")
                    self._telemetry = {
                        "base_x": float(self.data.xpos[self.base_id][0]),
                        "base_y": float(self.data.xpos[self.base_id][1]),
                        "base_z": float(self.data.xpos[self.base_id][2]),
                        "simulation_time": float(self.data.time),
                        "active": state is not None,
                        "head_pan": float(self.data.qpos[self.model.jnt_qposadr[self.head[0][1]]]),
                        "head_tilt": float(self.data.qpos[self.model.jnt_qposadr[self.head[1][1]]]),
                        "left_command": float(self.data.ctrl[self.wheels[0][0]]),
                        "right_command": float(self.data.ctrl[self.wheels[1][0]]),
                        "left_velocity": float(
                            self.data.qvel[self.model.jnt_dofadr[self.wheels[0][1]]]
                        ),
                        "right_velocity": float(
                            self.data.qvel[self.model.jnt_dofadr[self.wheels[1][1]]]
                        ),
                    }
                active = state is not None
                # Bound catch-up after scheduling stalls instead of teleporting the robot.
                deadline = max(deadline + self.model.opt.timestep, now - 0.02)
                self._stop.wait(max(0, deadline - time.monotonic()))
        except BaseException as exc:
            self._fault = exc
            self._command = None
            self.data.ctrl[:] = 0

    def _open_renderer(self):
        try:
            self._render_data = self.mj.MjData(self.model)
            self._renderer = self.mj.Renderer(self.model, self.height, self.width)
            # Exercise GL before advertising a listening driver.
            self._renderer.update_scene(self.data, camera=self.camera_id)
            self._renderer.render()
        except Exception as exc:
            raise RuntimeError(
                f"MuJoCo {self.backend} rendering failed; install libosmesa6 and use --gl osmesa "
                "on machines without a GPU, or check the EGL driver for --gl egl"
            ) from exc

    async def start(self):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mujoco-camera")
        await asyncio.get_running_loop().run_in_executor(self._executor, self._open_renderer)
        self._thread = threading.Thread(target=self._simulate, name="mujoco-physics", daemon=True)
        self._thread.start()

    async def close(self):
        self.neutral()
        self._stop.set()
        for track in self._tracks:
            track.stop()
        if self._thread:
            await asyncio.to_thread(self._thread.join)
        if self._executor:
            if self._renderer:
                await asyncio.get_running_loop().run_in_executor(
                    self._executor, self._renderer.close
                )
            await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)
            self._executor = None
