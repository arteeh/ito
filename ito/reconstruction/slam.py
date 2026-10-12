"""Robot-first tracking: the robot says where its camera is, vision corrects it, splats follow.

Composition of three narrow pieces: MotionBuffer (the robot's source-stamped camera motion),
a visual Frontend (MASt3R today) and PoseGraph (fusion). The world is the robot's odometry
frame where it started, so nothing restarts: a frame vision cannot use still gets a pose,
from the robot, and the pilot's eye keeps moving. Each frame's points are placed at its
fused pose, and the view it sees through replaces older splats there (carve).

The points are in metres because the robot knows how high its camera is above the floor:
where a frame sees the floor, its depth below the camera in the frame's points measures the
front end's scale. Fusion keeps a robust estimate over the last few seconds of frames for
the front end's map, and a frame that saw the floor is placed by its own.

The pilot side gets the fused camera pose and the correction (world-from-odometry) that
turns the robot's newest pose into the eye; the renderer eases changes of that correction
in, below what a pilot notices.
"""

import math

import numpy as np

from ito import clock

from .fusion import UP, PoseGraph
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
# The floor: points on rays at least this far below the horizon, the densest level among
# the lower half of them (low furniture stands above the floor, nothing below), holding at
# least this share of the frame's points, under a camera at least this high.
FLOOR_RAY = math.radians(10)
FLOOR_SHARE = 0.1
FLOOR_HEIGHT = 0.05
FLOOR_LEVEL = 0.05  # log half-width of the floor's level
FLOOR_SIGMA = 0.15  # one frame's log-scale from it, one sigma (MASt3R bends near depths)


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
        cloud = result.cloud
        if result.relative is not None:
            graph.add_visual(result.reference, node, result.relative, result.sigma)
            self.matched += 1
        floor = None
        if cloud is not None:
            robot = self.motion.at(exposure)
            floor = floor_scale(cloud, robot[:3, :3].T @ UP, float(robot[1, 3]))
            if floor is not None:
                # MASt3R's scale wanders from frame to frame (each tracked frame rewrites
                # its keyframe's points): the floor corrects it every frame it is seen.
                graph.add_scale(math.log(floor), FLOOR_SIGMA)
                graph.solve_scale()
        if oracle is not None:
            graph.nodes[node].pose = np.array(oracle, dtype=np.float64)
        else:
            graph.solve_frame()
        poses, dropped = {}, []
        if result.keyframe:
            dropped = graph.promote()
            if oracle is None:
                poses = graph.solve_window()
            else:
                poses = {k: graph.pose(k) for k in graph.keyframes}
        scale = graph.fold_scale()
        if poses or scale != 1.0:
            self.frontend.correct(poses, scale)
        if dropped:
            self.frontend.forget(dropped)
        if cloud is not None:
            # The frame's points are in the units the fold just replaced; its own floor,
            # where seen, places them better than the map's few seconds of floors.
            cloud = cloud._replace(points=cloud.points * (scale if floor is None else floor))
        pose = self._publish(node)
        if cloud is not None:
            self.place(cloud, pose, now + clock.now() - started)

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
