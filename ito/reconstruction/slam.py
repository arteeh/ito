"""Robot-first tracking: the robot says where its camera is, vision corrects it, splats follow.

Composition of three narrow pieces: MotionBuffer (the robot's source-stamped camera motion),
a visual Frontend (MASt3R today) and PoseGraph (fusion). The world is the robot's odometry
frame where it started, so nothing restarts: a frame vision cannot use still gets a pose,
from the robot, and the pilot's eye keeps moving. Each frame's points are placed at its
fused pose, and the view it sees through replaces older splats there (carve).

Metric depth is calibrated from image parallax and robot motion, and reused across views.
Pure rotation retains the learned gauge; camera height never implies a visible floor.

The pilot side gets the fused camera pose and the correction (world-from-odometry) that
turns the robot's newest pose into the eye; the renderer eases changes of that correction
in, below what a pilot notices.
"""

import math

import numpy as np

from ito import clock

from .depth_support import DepthSupport
from .fusion import PoseGraph
from .metric_scale import MetricScale
from .motion import MotionBuffer
from .rgbd import RGBDBackend
from .sensors import Sensors
from .slam_vision import RESTART_CAUSES

__all__ = ["RESTART_CAUSES", "SLAMBackend"]

# A frame smeared further than this by the camera's turn during its exposure still gets a
# pose, from the robot, but neither a visual factor nor splats: no tracker matches through
# it, and its splats would be streaks. Blur is |angular velocity| x exposure x focal length.
BLUR_PIXELS = 8.0
# A camera whose driver cannot say how long it exposes: indoor auto-exposure runs about this.
EXPOSURE = 0.010
# How far the online frame-delay estimate may stray from the driver's own.
DELAY_RANGE = (-0.1, 0.2)


class SLAMBackend(RGBDBackend):
    """frame_delay: the driver's estimate of how long before its capture_time a frame was
    exposed; exposure: how long the sensor integrates one, None if the driver cannot say.
    aligner: a tracking.MapAligner, the map's say in every tracked frame's pose.
    oracle: replaces fusion's pose with a known one (e2e/slam_replay's rungs)."""

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
        oracle=None,
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
        super().__init__(capacity, intrinsics, **options)
        self.report = report
        self.frontend = frontend
        self.aligner = aligner
        self.intrinsics = intrinsics
        self.aligned = 0  # frames the map aligner placed
        self.oracle = oracle
        # A keyframe needs the robot's motion back to its exposure; a minute outlasts any
        # look around the room.
        self.sensors = Sensors()
        self.motion = MotionBuffer(motion_noise, seconds=60.0, sensors=self.sensors)
        # The frame delay is estimated online, against the robot's motion: visual turns
        # only line up with the robot's at the right delay.
        bounds = (max(0.0, frame_delay + DELAY_RANGE[0]), frame_delay + DELAY_RANGE[1])
        self.graph = PoseGraph(
            self.motion,
            td=frame_delay,
            td_bounds=bounds,
            estimate_td=exposure != 0 or frame_delay > 0,
        )
        self.depth_estimator = MetricScale(intrinsics)
        self.depth_support = DepthSupport()
        self.focal = float(intrinsics.fx)
        self.exposure = EXPOSURE if exposure is None else float(exposure)
        self.camera_pose = np.eye(4, dtype=np.float32)
        self.correction = np.eye(4, dtype=np.float32)
        self.pose_publisher = None
        self.tracked = 0  # frames given a pose
        self.matched = 0  # of those, frames vision matched
        self.blurred = 0  # of those, frames too blurred to look at
        self.restarts = dict.fromkeys(RESTART_CAUSES, 0)  # Always zero: there is no map.
        self.lost = False
        self.unmatched_since = None
        self.recalls = 0
        self.waiting = False
        report("MASt3R ready; waiting for camera and robot motion")

    def bind_admission(self, admission):
        super().bind_admission(admission)
        if hasattr(self.frontend, "bind_admission"):
            self.frontend.bind_admission(admission)

    def add_sensors(self, observations):
        self.sensors.extend(observations)

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
        self.frontend.forget(
            graph.expire(self.motion.times[0] if len(self.motion.times) else exposure - 60)
        )
        node = graph.add_frame(captured)
        oracle = None if self.oracle is None else self.oracle(captured)
        predicted = graph.pose(node) if oracle is None else oracle
        self._emit(node)
        last = graph.latest_keyframe()
        reference_time = graph.nodes[last].time if last is not None else captured
        _, covariance = self.motion.between(reference_time - graph.td, exposure)
        spin = float(np.linalg.norm(self.motion.angular_velocity(exposure)))
        if oracle is None and spin * self.exposure * self.focal > BLUR_PIXELS:
            self.blurred += 1
            graph.solve_frame()
            self._publish(node)
            return
        result = self.frontend.track(rgb, node, predicted, np.sqrt(np.diag(covariance)))
        cloud = result.cloud
        if result.relative is not None:
            relative, sigma = result.relative, result.sigma
            if self.motion.noise is None and self.depth_estimator.known:
                # Without metric odometry, range scale belongs to visual translation
                # as well as the pointmap; keep both in the same evolving gauge.
                gain = math.exp(self.depth_estimator.log_scale)
                relative, sigma = relative.copy(), sigma.copy()
                relative[:3, 3] *= gain
                sigma[3:] *= gain
            if graph.add_visual(result.reference, node, relative, sigma):
                graph.solve_delay()
                self.matched += 1
        if oracle is not None:
            graph.nodes[node].pose = np.array(oracle, dtype=np.float64)
        else:
            graph.solve_frame()
            if self.aligner is not None:
                found = self.aligner.align(rgb, graph.pose(node), self.intrinsics)
                if found is not None and found[2] > 0 and graph.add_map(node, *found[:2]):
                    graph.solve_frame()
                    self.aligned += 1
        dropped = []
        if result.keyframe:
            dropped = graph.promote()
            if oracle is None:
                graph.solve_window()
        # Reassert every retained fused pose, including when a local visual solve was
        # rejected. Otherwise its wrong reference contaminates the next observation.
        self.frontend.correct({k: graph.pose(k) for k in graph.keyframes}, 1.0)
        if dropped:
            self.frontend.forget(dropped)
        pose = self._publish(node)
        if cloud is not None:
            if hasattr(self.frontend, "refine"):
                cloud = self.frontend.refine(cloud, pose, result.reference)
            observed = self.sensors.range_scale(cloud, exposure, self.motion)
            if observed is not None:
                self.depth_estimator.observe(observed)
            cloud = self.depth_estimator.apply(rgb, cloud, self.motion, exposure)
            self.place(cloud, pose, now + clock.now() - started)

    def pose_at(self, captured):
        """World-from-camera for any recent frame stamped `captured` (pilot clock), tracked
        or not: the robot's motion between the fused poses around it."""
        return self.graph.pose_at(captured)

    def _publish(self, node):
        graph = self.graph
        graph.publish(node)
        pose = graph.pose(node)
        self.camera_pose = pose.astype(np.float32)
        self.correction = graph.correction(node).astype(np.float32)
        self.tracked += 1
        self._emit(node)
        self.report(
            f"Robot-first tracking | {self.tracked} frames, "
            f"{self.matched / self.tracked:.0%} matched, {self.blurred} blurred, "
            f"frame delay {graph.td * 1000:.0f} ms"
        )
        return pose

    def _emit(self, node):
        if self.pose_publisher is not None:
            self.pose_publisher(
                self.graph.pose(node),
                self.graph.correction(node),
                self.graph.nodes[node].time - self.graph.td,
            )

    def place(self, cloud, pose, when):
        """Splats for a frame's points seen from pose; the view replaces what it sees through."""
        torch = _torch()
        world = torch.as_tensor(pose, device="cuda", dtype=torch.float32)
        points = cloud.points @ world[:3, :3].T + world[:3, 3]
        valid = cloud.valid
        fx, fy, cx, cy = cloud.intrinsics
        depth = -cloud.points[:, 2]
        # Three pixels per splat refresh in place, so a room looked around once fits in about
        # 400k splats; smaller budgets get proportionally coarser (as the vision path).
        pixels = 3 * max(1, (400_000 / self.budget) ** 0.5)
        footprint = depth * pixels / fx
        # DLPack shares CUDA memory; only the fused slot records cross to the ring.
        self.integrate_points(
            self.xp.from_dlpack(points[valid].contiguous()),
            self.xp.from_dlpack(cloud.colors[valid].contiguous()),
            when,
            self.xp.from_dlpack(footprint[valid].contiguous()),
        )
        height, width = cloud.shape
        evidence = getattr(self, "depth_support", None)
        carve_valid = valid if evidence is None else evidence.observe(cloud, pose)
        carve_valid = carve_valid & valid
        # Keep provisional appearance in the legacy point display. Uncertain depth
        # must not erase history; rejecting its appearance leaves large moving holes.
        if getattr(cloud, "confidence", None) is not None:
            carve_valid = carve_valid & (cloud.confidence > 0)
        image = torch.where(carve_valid, depth, 0).reshape(height, width)
        self.carve(
            pose.astype(np.float32),
            fx,
            fy,
            cx,
            cy,
            self.xp.from_dlpack(image.float().contiguous()),
            when,
        )


def _torch():
    import torch

    return torch


def heading(rotation):
    """Yaw of a world-from-camera rotation's view direction (-Z), positive to the left."""
    return math.atan2(rotation[0, 2], rotation[2, 2])
