"""MASt3R-SLAM's tracker and local graph, with a bounded live keyframe window.

A robot that measures where its camera looks (body heading plus head pan) keeps the map
honest. On a plain wall or a cupboard door MASt3R matches almost anything to anything and
reports a camera standing still while the robot turns half a room. A tracked heading
that strays from the robot's, a few lost frames in a row or an unusable pose start a fresh
local map at the robot's heading, so the pilot keeps a 3D scene that faces the right way.
"""

import math

import numpy as np

from ito import clock

from .rgbd import RGBDBackend

# Larger than the robot's own error (joint/IMU latency during a brisk turn), smaller than
# the misplaced room a pilot notices when they look back.
HEADING_TOLERANCE = math.radians(12)
# Unmatched frames in a row a measured map rides out before it restarts at the robot's
# pose. A walking robot's gait sway or a brief blank view costs MASt3R a frame or two; a
# fresh map for each throws away the room seen a moment ago.
MEASURED_MISSES = 2
AXES = np.array([1.0, -1.0, -1.0])  # OpenCV camera/world axes (+Y down, +Z forward) to Ito.


def heading(rotation):
    """Yaw of a world-from-camera rotation's view direction (-Z), positive to the left."""
    return math.atan2(rotation[0, 2], rotation[2, 2])


def turn(angle):
    """Rotation about the world's vertical axis."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)))


class Keyframes(list):
    def last_keyframe(self):
        return self[-1]

    def update_T_WCs(self, poses, indices):
        for transform, index in zip(poses, indices, strict=True):
            self[int(index)].T_WC = transform


class SLAMBackend(RGBDBackend):
    def __init__(self, capacity, intrinsics, *, report, **options):
        from .mast3r_runtime import load_model, prepare

        source, weights = prepare(report)
        import lietorch
        import torch
        from mast3r_slam.config import config, load_config
        from mast3r_slam.mast3r_utils import resize_img
        from mast3r_slam.tracker import FrameTracker

        load_config(str(source / "config/base.yaml"))
        config["single_thread"] = True
        # Ito always knows the camera's intrinsics; calibrated tracking and points
        # on the true pixel rays keep straight edges straight.
        config["use_calib"] = True
        config["tracking"]["filtering_mode"] = "recent"
        config["local_opt"]["max_iters"] = 5
        self.model = load_model(weights, report)
        options["device"] = "cuda"
        super().__init__(capacity, intrinsics, **options)
        self.report = report
        self.lietorch = lietorch
        self.frames = Keyframes()
        self.tracker = FrameTracker(self.model, self.frames, "cuda")
        # MASt3R sees a resized, centre-cropped image; carry the driver's calibration there.
        blank = np.zeros((intrinsics.height, intrinsics.width, 3))
        _, (sx, sy, crop_x, crop_y) = resize_img(blank, 512, return_transformation=True)
        self.K = torch.tensor(
            [
                [intrinsics.fx / sx, 0, intrinsics.cx / sx - crop_x],
                [0, intrinsics.fy / sy, intrinsics.cy / sy - crop_y],
                [0, 0, 1],
            ],
            device="cuda",
        )
        self.transform = lietorch.Sim3.Identity(1, device="cuda")
        self.camera_pose = np.eye(4, dtype=np.float32)
        self.tracked = 0
        self.realigned = 0
        self.frame_id = 0
        self.lost = False
        self.failures = 0
        self.world = None  # Ito world from the (axis-flipped) map, fixed when the map starts.
        # Where the current local map started: its camera in the world, and the robot's
        # own idea of that camera. The robot's turns since then predict the heading.
        self.anchor = None
        self.axes = torch.tensor(AXES, device="cuda", dtype=torch.float32)
        report("MASt3R-SLAM ready; waiting for camera")

    def placed(self, transform):
        """The Ito world-from-camera pose of a map pose and its scale, or None if unusable."""
        matrix = transform.matrix()[0].double().cpu().numpy()
        scale = abs(np.linalg.det(matrix[:3, :3])) ** (1 / 3)
        if not np.isfinite(matrix).all() or not 0.01 < scale < 100:
            return None, scale
        # Sim3 rotations drift from orthonormal in float32; snap to the nearest one.
        u, _, vh = np.linalg.svd(matrix[:3, :3])
        rigid = np.eye(4)
        rigid[:3, :3] = (u @ vh) * AXES[:, None] * AXES[None, :]
        rigid[:3, 3] = matrix[:3, 3] * AXES
        return (rigid if self.world is None else self.world @ rigid), scale

    def unplaced(self, pose, scale):
        """The map's Sim3 for an Ito world-from-camera pose."""
        from ito.render.pose import quaternion

        rigid = np.linalg.inv(self.world) @ pose
        rotation = rigid[:3, :3] * AXES[:, None] * AXES[None, :]
        data = [*(rigid[:3, 3] * AXES), *quaternion(rotation), scale]
        return self.lietorch.Sim3(self.K.new_tensor(data)[None])

    def predicted(self, prior, fallback):
        """Where the robot's turns since the map started say the camera now looks."""
        start, robot = self.anchor
        rotation = prior[:3, :3] @ robot[:3, :3].T @ start[:3, :3]
        result = fallback.copy()
        result[:3, :3] = rotation
        return result

    def start_map(self, frame, pose, prior):
        self.frames.clear()
        self.frames.append(frame)
        self.tracker.reset_idx_f2k()
        self.failures = 0
        self.lost = False
        self.anchor = (pose, prior)

    def integrate(self, rgb, depth, camera, now, measured=False):
        """Camera is the robot's world-from-camera estimate; measured when it tracks the gaze."""
        import torch
        from mast3r_slam.frame import create_frame
        from mast3r_slam.geometry import constrain_points_to_ray
        from mast3r_slam.global_opt import FactorGraph
        from mast3r_slam.mast3r_utils import mast3r_inference_mono

        started = clock.now()
        prior = np.asarray(camera, dtype=np.float64)
        realigned = False
        with torch.inference_mode():
            frame = create_frame(self.frame_id, rgb.astype(np.float32) / 255, self.transform)
            frame.K = self.K
            self.frame_id += 1
            if self.failures >= max(len(self.frames), 1) + 2:
                # Every retained view failed: start a fresh local map where the camera
                # last was, so the pilot gets the room back instead of a fading memory.
                self.frames.clear()
                self.failures = 0
                self.report("SLAM tracking restarted from the current view")
            if not self.frames:
                points, confidence = mast3r_inference_mono(self.model, frame)
                frame.update_pointmap(points, confidence)
                pose, scale = self.placed(frame.T_WC)
                if self.world is None:
                    # The map's own frame is wherever the camera first looked. Turn it once
                    # to where the robot says that camera was, so the map has gravity down
                    # and the startup heading ahead, like the pilot's world.
                    self.world = prior @ np.linalg.inv(pose)
                    pose = prior.copy()
                self.start_map(frame, pose, prior)
            else:
                # Without the robot's heading, try one retained keyframe per observation
                # during loss. Never reset the world origin or fuse a failed pose.
                if self.lost and not measured:
                    self.tracker.reset_idx_f2k()
                    self.frames.insert(0, self.frames.pop())
                    frame.T_WC = self.frames[-1].T_WC
                new_keyframe, _, lost = self.tracker.track(frame)
                pose, scale = (None, 0) if lost else self.placed(frame.T_WC)
                if measured and pose is None and self.failures < MEASURED_MISSES:
                    # Keep the map and the last pose; the next frame tries the same keyframe.
                    self.failures += 1
                    self.tracker.reset_idx_f2k()
                    return
                if measured:
                    last, last_scale = self.placed(self.transform)
                    expected = self.predicted(prior, pose if pose is not None else last)
                    stray = (
                        math.remainder(heading(pose[:3, :3]) - heading(expected[:3, :3]), math.tau)
                        if pose is not None
                        else math.inf
                    )
                    if abs(stray) > HEADING_TOLERANCE:
                        # Keep SLAM's own tilt and position when it has them; the robot's
                        # heading is the one thing a featureless view cannot tell.
                        if pose is None:
                            pose, scale = expected, last_scale
                        else:
                            pose = pose.copy()
                            pose[:3, :3] = turn(-stray) @ pose[:3, :3]
                        frame.T_WC = self.unplaced(pose, scale)
                        self.start_map(frame, pose, prior)
                        new_keyframe = False
                        realigned = True
                        self.realigned += 1
                elif pose is None:
                    self.lost = True
                    self.failures += 1
                    self.report("SLAM tracking lost; move back toward the last view")
                    return
                self.failures = 0
                self.lost = False
                if new_keyframe:
                    self.frames.append(frame)
                    # Bounded graph includes adjacent edges and a local loop to the
                    # oldest retained view. Upstream rejects non-overlapping edges.
                    del self.frames[:-4]
                    graph = FactorGraph(self.model, self.frames, self.K, device="cuda")
                    last = len(self.frames) - 1
                    graph.add_factors(list(range(last)), list(range(1, last + 1)), 0.1)
                    if last > 1:
                        graph.add_factors([0], [last], 0.1)
                    graph.solve_GN_calib()
                    self.tracker.reset_idx_f2k()
                    pose, scale = self.placed(frame.T_WC)
                    if pose is None:
                        # The local solve diverged: drop the window, keep the world.
                        self.frames.clear()
                        self.report("SLAM tracking restarted from the current view")
                        return
            self.transform = frame.T_WC
            self.camera_pose = pose.astype(np.float32)
            local = constrain_points_to_ray(frame.img.shape[-2:], frame.X_canon[None], self.K)[0]
            world = torch.as_tensor(self.world, device="cuda", dtype=torch.float32)
            points = self.transform.act(local) * self.axes
            points = points @ world[:3, :3].T + world[:3, 3]
            # No confidence cut: MASt3R is least confident on plain walls and floors,
            # which it still places well, and a room without its walls is no room.
            valid = torch.isfinite(points).all(dim=1) & (local[:, 2] > 0)
            colors = (frame.uimg.to("cuda").reshape(-1, 3) * 255).to(torch.uint8)
            # Size splats by the pixel footprint in the world's own (arbitrary) Sim3 units.
            # Per-frame pose and depth jitter moves every point a little: at 1.5 pixels
            # almost every cell is new each frame, the budget starves and views arrive
            # as scattered dots. Three pixels refresh in place and hold the temporal
            # window in about 400k splats; smaller budgets get proportionally coarser.
            pixels = 3 * max(1, (400_000 / self.budget) ** 0.5)
            footprint = local[:, 2] * float(scale) * pixels / self.K[0, 0]
            # DLPack shares CUDA memory; only the fused slot records cross to the ring.
            self.integrate_points(
                self.xp.from_dlpack(points[valid].contiguous()),
                self.xp.from_dlpack(colors[valid].contiguous()),
                now + clock.now() - started,
                self.xp.from_dlpack(footprint[valid].contiguous()),
            )
            self.tracked += 1
            if realigned:
                self.report(f"SLAM realigned to the robot's heading | {self.realigned} times")
            else:
                self.report(f"MASt3R-SLAM tracking | {self.tracked} frames")
