import math

import numpy as np

from ito import clock

from .fusion import UP, PoseGraph
from .gaussian import GaussianBackend
from .motion import MotionBuffer
from .sensors import HeadSensors
from .slam_vision import RESTART_CAUSES

__all__ = ["RESTART_CAUSES", "SLAMBackend"]

# A frame smeared further than this by the camera's turn during its exposure still gets a
# pose, from the robot, but neither a visual factor nor splats: no tracker matches through
# it, and its splats would be streaks. Blur is |angular velocity| x exposure x focal length.
BLUR_PIXELS = 8.0
EXPOSURE = 0.010
DELAY_RANGE = (-0.1, 0.2)
FLOOR_RAY = math.radians(10)
FLOOR_SHARE = 0.1
FLOOR_HEIGHT = 0.05
FLOOR_LEVEL = 0.05
FLOOR_SIGMA = 0.15
RANGE_SIGMA = 0.05
SCALE_STEP = 0.01


class SLAMBackend(GaussianBackend):
    """frame_delay: the driver's estimate of how long before its capture_time a frame was
    exposed; exposure: how long the sensor integrates one, None if the driver cannot say.
    aligner: a tracking.MapAligner, the map's say in every tracked frame's pose."""

    def __init__(
        self,
        capacity,
        intrinsics,
        *,
        report,
        motion_noise,
        frame_delay=0.0,
        exposure=None,
        frontend=None,
        precision="full",
        input_resolution=100,
        precision_report=None,
        aligner=None,
        **options,
    ):
        if frontend is None:
            from .mast3r_frontend import MASt3RFrontend

            frontend = MASt3RFrontend(
                intrinsics,
                report,
                precision=precision,
                input_resolution=input_resolution,
                precision_report=precision_report,
            )
        options["device"] = "cuda"
        super().__init__(capacity, intrinsics, precision=precision, **options)
        self.report = report
        self.frontend = frontend
        self.aligner = aligner
        self.intrinsics = intrinsics
        self.aligned = 0
        self.motion = MotionBuffer(motion_noise, seconds=60.0)
        self.sensors = HeadSensors(seconds=60.0)
        bounds = (max(0.0, frame_delay + DELAY_RANGE[0]), frame_delay + DELAY_RANGE[1])
        self.graph = PoseGraph(
            self.motion, td=frame_delay, td_bounds=bounds, estimate_td=True, inertial=self.sensors
        )
        self.focal = float(intrinsics.fx)
        self.exposure = EXPOSURE if exposure is None else float(exposure)
        self.camera_pose = np.eye(4, dtype=np.float32)
        self.correction = np.eye(4, dtype=np.float32)
        self.tracked = 0
        self.matched = 0
        self.blurred = 0
        self.restarts = dict.fromkeys(RESTART_CAUSES, 0)
        self.lost = False
        self.unmatched_since = None
        self.recalls = 0
        self.waiting = False
        self.retired_keyframes = 0
        from .registration import Registration

        self.registration = Registration()
        report("MASt3R ready; waiting for camera and robot motion")

    def bind_admission(self, admission):
        super().bind_admission(admission)
        if hasattr(self.frontend, "bind_admission"):
            self.frontend.bind_admission(admission)

    def add_sensors(self, imu, ranges):
        self.sensors.add(imu, ranges)

    def add_motion(self, rows):
        self.motion.extend(rows)

    def integrate(self, rgb, depth, camera, now, measured=False, *, captured):
        """captured: the frame's stamp on the pilot clock, the clock motion samples are on."""
        started = clock.now()
        graph = self.graph
        exposure = captured - graph.td
        if not self.motion.covers(exposure):
            if not self.waiting:
                self.report("Robot-first tracking waiting for the robot's motion")
            self.waiting = True
            return
        self.waiting = False
        propagated = graph.predict(captured) if graph.latest() is not None else None
        expired = graph.expire(self.motion.times[0])
        self.retired_keyframes += len(expired)
        self.frontend.forget(expired)
        if expired and not graph.keyframes:
            graph.anchor = propagated.copy()
        node = graph.add_frame(
            captured, pose=propagated if expired and not graph.keyframes else None
        )
        predicted = graph.pose(node)
        last = graph.latest_keyframe()
        reference_time = graph.nodes[last].time if last is not None else captured
        _, covariance = self.motion.between(reference_time - graph.td, exposure)
        gyro = self.sensors.angular_velocity(exposure)
        if gyro is None:
            gyro = self.motion.angular_velocity(exposure)
        spin = float(np.linalg.norm(gyro))
        if spin * self.exposure * self.focal > BLUR_PIXELS:
            self.blurred += 1
            graph.solve_frame()
            self._publish(node)
            return
        result = self.frontend.track(rgb, node, predicted, np.sqrt(np.diag(covariance)))
        cloud = result.cloud
        if result.relative is not None:
            graph.add_visual(result.reference, node, result.relative, result.sigma)
            graph.solve_delay()
            self.matched += 1
        if cloud is not None:
            robot = self.motion.at(exposure)
            seen = []
            below = floor_scale(cloud, robot[:3, :3].T @ UP, float(robot[1, 3]))
            if below is not None:
                seen.append((math.log(below), FLOOR_SIGMA))
            ranged = self.sensors.range_scale(cloud, exposure, self.motion)
            if ranged is not None:
                seen.append((math.log(ranged), RANGE_SIGMA))
            for value, sigma in seen:
                graph.add_scale(value, sigma)
            if seen:
                graph.solve_scale()
        graph.solve_frame()
        if self.aligner is not None:
            found = self.aligner.align(rgb, graph.pose(node), self.intrinsics)
            if found is not None and found[2] > 0 and graph.add_map(node, *found[:2]):
                graph.solve_frame()
                self.aligned += 1
        poses, dropped = {}, []
        if result.keyframe:
            dropped = graph.promote()
            poses = graph.solve_window()
        scale = 1.0
        if poses or abs(graph.log_scale) > SCALE_STEP:
            scale = graph.fold_scale()
            self.frontend.correct(poses, scale)
        self.revise(poses)
        if dropped:
            self.retired_keyframes += len(dropped)
            self.frontend.forget(dropped)
        if cloud is not None:
            cloud = cloud._replace(points=cloud.points * scale)
        pose = self._publish(node)
        if cloud is not None:
            self.place(cloud, pose, now + clock.now() - started, node=node)

    def pose_at(self, captured):
        pose = self.graph.pose_at(captured)
        return None if pose is None else self.registration.map_from_estimator @ pose

    def _publish(self, node):
        graph = self.graph
        graph.publish(node)
        pose = graph.pose(node)
        mapping = self.registration.map_from_estimator
        self.camera_pose = (mapping @ pose).astype(np.float32)
        self.correction = (mapping @ graph.correction(node)).astype(np.float32)
        self.tracked += 1
        self.report(
            f"Robot-first tracking | {self.tracked} frames, "
            f"{self.matched / self.tracked:.0%} matched, {self.blurred} blurred, "
            f"{self.sensors.ranged} ToF-scaled, "
            f"frame delay {graph.td * 1000:.0f} ms"
        )
        return pose

    def place(self, cloud, pose, when, *, node):
        height, width = cloud.shape
        points = cloud.points.detach().float().cpu().numpy().reshape(height, width, 3)
        image = cloud.colors.detach().cpu().numpy().reshape(height, width, 3)
        self.observation_estimate = pose
        mapped, scale = self.registration.align(
            image, -points[..., 2], pose, cloud.intrinsics, self.views
        )
        self.registration.remember(node, pose, mapped, self.graph.nodes)
        self.camera_pose = mapped.astype(np.float32)
        self.correction = (
            self.registration.map_from_estimator @ self.graph.correction(node)
        ).astype(np.float32)
        self.observe(
            points * scale,
            image,
            cloud.valid.detach().cpu().numpy(),
            mapped,
            cloud.intrinsics,
            when,
            node=node,
            confidence=cloud.confidence.detach().cpu().numpy()
            if cloud.confidence is not None
            else None,
        )


def floor_scale(cloud, up, height):
    """Metres per front-end unit from the floor in a frame's points, or None if it is not
    in view. up: the world's up in the camera frame; height: the camera's above the floor."""
    if height < FLOOR_HEIGHT:
        return None
    torch = _torch()
    points = cloud.points[cloud.valid][::4]
    up = torch.as_tensor(up, device=points.device, dtype=points.dtype)
    below = -(points @ up)
    lower = below > math.sin(FLOOR_RAY) * torch.linalg.vector_norm(points, dim=1)
    levels = torch.log(below[lower]).cpu().numpy()
    if len(levels) < FLOOR_SHARE * len(points):
        return None
    bottom, top = np.percentile(levels, (40, 99.5))
    counts, edges = np.histogram(levels, np.arange(bottom, top + 0.04, 0.02))
    densest = int(np.argmax(np.convolve(counts, np.ones(5), "same")))
    near = np.abs(levels - edges[densest] - 0.01) < FLOOR_LEVEL
    if near.sum() < FLOOR_SHARE * len(points):
        return None
    return height / math.exp(float(np.median(levels[near])))


def _torch():
    import torch

    return torch


def heading(rotation):
    """Yaw of a world-from-camera rotation's view direction (-Z), positive to the left."""
    return math.atan2(rotation[0, 2], rotation[2, 2])
