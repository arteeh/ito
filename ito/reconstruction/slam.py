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
from .sensors import HeadSensors
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
# The floor: points on rays at least this far below the horizon, the densest level among
# the lower half of them (low furniture stands above the floor, nothing below), holding at
# least this share of the frame's points, under a camera at least this high.
FLOOR_RAY = math.radians(10)
FLOOR_SHARE = 0.1
FLOOR_HEIGHT = 0.05
FLOOR_LEVEL = 0.05  # log half-width of the floor's level
FLOOR_SIGMA = 0.15  # one frame's log-scale from it, one sigma (MASt3R bends near depths)
# A ToF's zones measure depth itself, millimetres to centimetres; what is left is MASt3R
# bending between the zones and the rest of the frame.
RANGE_SIGMA = 0.05
SCALE_STEP = 0.01  # Between keyframes the front end takes a new scale past this log-change.


class SLAMBackend(RGBDBackend):
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
        super().__init__(capacity, intrinsics, **options)
        self.report = report
        self.frontend = frontend
        self.aligner = aligner
        self.intrinsics = intrinsics
        self.aligned = 0  # frames the map aligner placed
        # A keyframe needs the robot's motion back to its exposure; a minute outlasts any
        # look around the room.
        self.motion = MotionBuffer(motion_noise, seconds=60.0)
        self.sensors = HeadSensors(seconds=60.0)
        # The frame delay is estimated online, against the robot's motion: visual turns
        # only line up with the robot's at the right delay.
        bounds = (max(0.0, frame_delay + DELAY_RANGE[0]), frame_delay + DELAY_RANGE[1])
        self.graph = PoseGraph(
            self.motion, td=frame_delay, td_bounds=bounds, estimate_td=True, inertial=self.sensors
        )
        self.focal = float(intrinsics.fx)
        self.exposure = EXPOSURE if exposure is None else float(exposure)
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
        self.frontend.forget(graph.expire(self.motion.times[0]))
        node = graph.add_frame(captured)
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
        measured = None  # This frame's own metres per front-end unit, if it saw any.
        if cloud is not None:
            # MASt3R's scale wanders from frame to frame (each tracked frame rewrites its
            # keyframe's points): the floor, and the ToF wherever it looks, correct it every
            # frame they see, and place that frame's points by their own measure.
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
                weights = [sigma**-2 for _, sigma in seen]
                measured = math.exp(
                    sum(w * value for w, (value, _) in zip(weights, seen, strict=True))
                    / sum(weights)
                )
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
        if dropped:
            self.frontend.forget(dropped)
        if cloud is not None:
            # The frame's points are in the units the fold just replaced; its own floor and
            # ranges, where seen, place them better than the map's last few seconds of them.
            own = scale if measured is None else measured
            cloud = cloud._replace(points=cloud.points * own)
        pose = self._publish(node)
        if cloud is not None:
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
        self.report(
            f"Robot-first tracking | {self.tracked} frames, "
            f"{self.matched / self.tracked:.0%} matched, {self.blurred} blurred, "
            f"{self.sensors.ranged} ToF-scaled, "
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
