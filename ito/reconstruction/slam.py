"""Robot-first tracking: the robot says where its camera is, vision corrects it, splats follow.

Composition of three narrow pieces: MotionBuffer (the robot's source-stamped camera motion),
a visual Frontend (MASt3R today) and PoseGraph (fusion). The world is the robot's odometry
frame where it started, so nothing restarts: a frame vision cannot use still gets a pose,
from the robot, and the pilot's eye keeps moving. Each frame's points are placed at its
fused pose, and the view it sees through replaces older splats there (carve).

The pilot side gets the fused camera pose and the correction (world-from-odometry) that
turns the robot's newest pose into the eye; the renderer eases changes of that correction
in, below what a pilot notices.
"""

import math

import numpy as np

from ito import clock

from .fusion import PoseGraph
from .motion import MotionBuffer
from .rgbd import RGBDBackend
from .slam_vision import RESTART_CAUSES

__all__ = ["RESTART_CAUSES", "SLAMBackend"]

# A frame smeared further than this by the camera's turn during its exposure still gets a
# pose, from the robot, but neither a visual factor nor splats: no tracker matches through
# it, and its splats would be streaks. Blur is |angular velocity| x exposure x focal length.
BLUR_PIXELS = 8.0
# Nothing on the wire says how long the sensor exposed; indoor auto-exposure runs about this.
EXPOSURE = 0.010
# How far the online frame-delay estimate may stray from the driver's own.
DELAY_RANGE = (-0.1, 0.2)


class SLAMBackend(RGBDBackend):
    """frame_delay: the driver's estimate of how long before its capture_time a frame was
    exposed. oracle: replaces fusion's pose with a known one (e2e/slam_replay's rungs)."""

    def __init__(
        self,
        capacity,
        intrinsics,
        *,
        report,
        motion_noise,
        frame_delay=0.0,
        frontend=None,
        oracle=None,
        **options,
    ):
        if frontend is None:
            from .mast3r_frontend import MASt3RFrontend

            frontend = MASt3RFrontend(intrinsics, report)
        options["device"] = "cuda"
        super().__init__(capacity, intrinsics, **options)
        self.report = report
        self.frontend = frontend
        self.oracle = oracle
        # A keyframe needs the robot's motion back to its exposure; a minute outlasts any
        # look around the room.
        self.motion = MotionBuffer(motion_noise, seconds=60.0)
        # The frame delay is estimated online, against the robot's motion: visual turns
        # only line up with the robot's at the right delay.
        bounds = (max(0.0, frame_delay + DELAY_RANGE[0]), frame_delay + DELAY_RANGE[1])
        self.graph = PoseGraph(self.motion, td=frame_delay, td_bounds=bounds, estimate_td=True)
        self.focal = float(intrinsics.fx)
        self.camera_pose = np.eye(4, dtype=np.float32)
        self.correction = np.eye(4, dtype=np.float32)
        self.tracked = 0  # frames given a pose
        self.matched = 0  # of those, frames vision matched
        self.blurred = 0  # of those, frames too blurred to look at
        self.restarts = dict.fromkeys(RESTART_CAUSES, 0)  # Always zero: there is no map.
        self.lost = False
        self.unmatched_since = None
        self.recalls = 0
        self.waiting = False
        report("MASt3R ready; waiting for camera and robot motion")

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
        self.frontend.forget(graph.expire(self.motion.times[0]))
        node = graph.add_frame(captured)
        oracle = None if self.oracle is None else self.oracle(captured)
        predicted = graph.pose(node) if oracle is None else oracle
        last = graph.latest_keyframe()
        reference_time = graph.nodes[last].time if last is not None else captured
        _, covariance = self.motion.between(reference_time - graph.td, exposure)
        spin = float(np.linalg.norm(self.motion.angular_velocity(exposure)))
        if oracle is None and spin * EXPOSURE * self.focal > BLUR_PIXELS:
            self.blurred += 1
            graph.solve_frame()
            self._publish(node)
            return
        result = self.frontend.track(rgb, node, predicted, np.sqrt(np.diag(covariance)))
        if result.relative is not None:
            graph.add_visual(result.reference, node, result.relative, result.sigma)
            self.matched += 1
        if oracle is not None:
            graph.nodes[node].pose = np.array(oracle, dtype=np.float64)
        else:
            graph.solve_frame()
        if result.keyframe:
            dropped = graph.promote()
            if oracle is None:
                poses = graph.solve_window()
                scale = graph.fold_scale()
            else:
                poses, scale = {k: graph.pose(k) for k in graph.keyframes}, 1.0
            self.frontend.correct(poses, scale)
            self.frontend.forget(dropped)
        pose = self._publish(node)
        if result.cloud is not None:
            self.place(result.cloud, pose, now + clock.now() - started)

    def _publish(self, node):
        graph = self.graph
        pose = graph.pose(node)
        self.camera_pose = pose.astype(np.float32)
        self.correction = graph.correction(node).astype(np.float32)
        self.tracked += 1
        self.report(
            f"Robot-first tracking | {self.tracked} frames, "
            f"{self.matched / self.tracked:.0%} matched, {self.blurred} blurred, "
            f"frame delay {graph.td * 1000:.0f} ms"
        )
        return pose

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
        image = torch.where(valid, depth, 0).reshape(height, width)
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
