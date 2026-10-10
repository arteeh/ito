"""MASt3R-SLAM's tracker and local graph, with a bounded live keyframe window.

A robot that measures where its camera looks (body heading plus head pan) keeps the map
honest. On a plain wall or a cupboard door MASt3R matches almost anything to anything and
reports a camera standing still while the robot turns half a room. Every fresh local map
leaves the old one's splats to fade beside it, doubled and tilted walls a pilot notices,
so a measured map restarts only when nothing else works: a tracked heading that strays
from the robot's turns the map's world back onto it over a few frames, frames MASt3R
cannot match are skipped while the robot's own pose carries the view, and after a turn
the tracker tries the keyframe that last faced the robot's way. A loss that outlasts
that, or a solve that diverges, starts a fresh map at the robot's heading; each restart
is counted by cause.
"""

import math
import os

import numpy as np

from ito import clock

from .rgbd import RGBDBackend

# A stray larger than the robot's own error (joint/IMU latency during a brisk turn) starts
# turning the world back onto the robot's heading, a third of the stray per frame, until
# it is within HEADING_SETTLED. Old splats stay put and fade; small steps keep the seam
# between them and new ones small.
HEADING_TOLERANCE = math.radians(12)
HEADING_SETTLED = math.radians(2)
HEADING_EASE = 1 / 3
# Further off than this MASt3R tracked the wrong room, not a drifting one.
HEADING_LOST = math.radians(60)
# How long a measured map skips unusable frames on the robot's own pose before it restarts.
# A gait sway, a step or a blank patch costs MASt3R a few frames; the pilot's view drops
# to the flat feed after TRACKING_LOST (2 s), so a lasting loss restarts well before that.
LOSS_SECONDS = 1.0
# A tracked frame this far in heading from the newest keyframe becomes one, whatever the
# tracker's overlap test says (#24). Frames match 92-99% within 10 degrees of the last
# tracked view and almost never past 20-30: during a head turn the map must grow keyframes
# along the way, or the turn outruns it and every later frame of the look goes unmatched.
# ITO_SLAM_KEYFRAME_YAW (degrees, 0 off) overrides, for live comparisons.
KEYFRAME_YAW = math.radians(float(os.environ.get("ITO_SLAM_KEYFRAME_YAW", 8)))
# Keyframes kept per heading sector, beyond the live window, for turning back; fine enough
# that the one nearest the robot's gaze is within the tracker's reach.
SPREAD = math.radians(float(os.environ.get("ITO_SLAM_SPREAD", 10)))
WINDOW = 4  # Recent keyframes in the local graph.
RESTART_CAUSES = ("unmatched", "heading", "failure")
REASONS = dict(unmatched="tracking lost", heading="heading lost", failure="solve failed")
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
        self.restarts = dict.fromkeys(RESTART_CAUSES, 0)
        self.corrections = 0  # Heading corrections started; each eases over a few frames.
        self.yaw_keyframes = 0  # Keyframes added because the camera turned (#24).
        self.correcting = False
        self.frame_id = 0
        self.lost = False
        self.failures = 0
        # When a measured map last stopped matching; None while it tracks.
        self.unmatched_since = None
        # The newest keyframe per heading sector of the current map, for turning back.
        self.spread = {}
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
        # Composed onto a map tilted unlike the robot's estimate (a measured camera rolls
        # 30 degrees at a wide look), that turn also swings the heading, and each correction
        # would carry the error on. The world shares the robot's startup heading, so the
        # robot's measured heading is the camera's.
        rotation = turn(heading(prior[:3, :3]) - heading(rotation)) @ rotation
        result = fallback.copy()
        result[:3, :3] = rotation
        return result

    def start_map(self, frame, pose, prior):
        self.frames.clear()
        self.frames.append(frame)
        self.spread.clear()
        self.remember(frame, pose)
        self.tracker.reset_idx_f2k()
        self.failures = 0
        self.lost = False
        self.correcting = False
        self.unmatched_since = None
        self.anchor = (pose, prior)

    def restart(self, cause):
        self.restarts[cause] += 1
        self.report(
            f"SLAM map restarted: {REASONS[cause]} | "
            + ", ".join(f"{name} {count}" for name, count in self.restarts.items())
        )

    def turned(self, pose):
        """The camera has turned far enough from the newest keyframe to need another."""
        if not KEYFRAME_YAW or pose is None or not self.frames:
            return False
        last, _ = self.placed(self.frames[-1].T_WC)
        if last is None:
            return False
        return abs(math.remainder(heading(pose[:3, :3]) - heading(last[:3, :3]), math.tau)) > (
            KEYFRAME_YAW
        )

    def remember(self, frame, pose):
        sectors = round(math.tau / SPREAD)
        self.spread[round(heading(pose[:3, :3]) / SPREAD) % sectors] = frame

    def relocalize(self, prior):
        """Track the next frame against the keyframe that faced where the robot now looks."""
        self.tracker.reset_idx_f2k()
        looking = heading(prior[:3, :3])

        def away(keyframe):
            pose, _ = self.placed(keyframe.T_WC)
            if pose is None:
                return math.inf
            return abs(math.remainder(heading(pose[:3, :3]) - looking, math.tau))

        best = min(self.spread.values(), key=away, default=None)
        if best is not None and away(best) + SPREAD / 2 < away(self.frames[-1]):
            # The local graph always joins neighbouring keyframes; one that faced
            # elsewhere starts its own chain rather than a bogus edge to the last.
            self.frames[:] = [best]

    def steer(self, pose, stray):
        """Turn the world about the camera toward the robot's heading, a part per frame."""
        if not self.correcting and abs(stray) <= HEADING_TOLERANCE:
            return pose
        if abs(stray) <= HEADING_SETTLED:
            self.correcting = False
            return pose
        if not self.correcting:
            self.correcting = True
            self.corrections += 1
        step = np.eye(4)
        step[:3, :3] = turn(-stray * HEADING_EASE)
        step[:3, 3] = pose[:3, 3] - step[:3, :3] @ pose[:3, 3]
        self.world = step @ self.world
        start, robot = self.anchor
        self.anchor = (step @ start, robot)
        return step @ pose

    def integrate(self, rgb, depth, camera, now, measured=False):
        """Camera is the robot's world-from-camera estimate; measured when it tracks the gaze."""
        import torch
        from mast3r_slam.frame import create_frame
        from mast3r_slam.geometry import constrain_points_to_ray
        from mast3r_slam.global_opt import FactorGraph
        from mast3r_slam.mast3r_utils import mast3r_inference_mono

        started = clock.now()
        prior = np.asarray(camera, dtype=np.float64)
        restarted = None
        with torch.inference_mode():
            frame = create_frame(self.frame_id, rgb.astype(np.float32) / 255, self.transform)
            frame.K = self.K
            self.frame_id += 1
            if self.failures >= max(len(self.frames), 1) + 2:
                # Every retained view failed: start a fresh local map where the camera
                # last was, so the pilot gets the room back instead of a fading memory.
                self.frames.clear()
                self.failures = 0
                restarted = "unmatched"
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
                elif measured and pose is not None:
                    # A map restarted from the last pose faces where the camera looked then;
                    # the robot says where it looks now.
                    pose = pose.copy()
                    pose[:3, :3] = (
                        turn(heading(prior[:3, :3]) - heading(pose[:3, :3])) @ pose[:3, :3]
                    )
                    frame.T_WC = self.unplaced(pose, scale)
                self.start_map(frame, pose, prior)
            else:
                # Without the robot's heading, try one retained keyframe per observation
                # during loss. Never reset the world origin or fuse a failed pose.
                if self.lost and not measured:
                    self.tracker.reset_idx_f2k()
                    self.frames.insert(0, self.frames.pop())
                    frame.T_WC = self.frames[-1].T_WC
                previous, previous_scale = self.placed(self.transform) if measured else (None, 0)
                if measured and self.unmatched_since is not None:
                    # Start the solve where the robot says the camera now looks.
                    frame.T_WC = self.unplaced(self.predicted(prior, previous), previous_scale)
                new_keyframe, _, lost = self.tracker.track(frame)
                pose, scale = (None, 0) if lost else self.placed(frame.T_WC)
                if measured:
                    cause = "unmatched" if lost else "failure" if pose is None else None
                    stray = 0.0
                    if pose is not None:
                        expected = self.predicted(prior, pose)
                        stray = math.remainder(
                            heading(pose[:3, :3]) - heading(expected[:3, :3]), math.tau
                        )
                        if abs(stray) > HEADING_LOST:
                            cause = "heading"
                    if cause and self.unmatched_since is None:
                        self.unmatched_since = now
                    if cause and now - self.unmatched_since < LOSS_SECONDS:
                        # Keep the map and the last pose; the robot's own pose carries
                        # the pilot's view until a frame matches again.
                        self.relocalize(prior)
                        return
                    if cause:
                        # Keep SLAM's own tilt and position when it has them; the robot's
                        # heading is the one thing a featureless view cannot tell.
                        if pose is None:
                            pose, scale = self.predicted(prior, previous), previous_scale
                        else:
                            pose = pose.copy()
                            pose[:3, :3] = turn(-stray) @ pose[:3, :3]
                        frame.T_WC = self.unplaced(pose, scale)
                        self.start_map(frame, pose, prior)
                        new_keyframe = False
                        restarted = cause
                    else:
                        pose = self.steer(pose, stray)
                    self.unmatched_since = None
                elif pose is None:
                    self.lost = True
                    self.failures += 1
                    self.report("SLAM tracking lost; move back toward the last view")
                    return
                self.failures = 0
                self.lost = False
                if not new_keyframe and not restarted and self.turned(pose):
                    new_keyframe = True
                    self.yaw_keyframes += 1
                if new_keyframe:
                    self.frames.append(frame)
                    # Bounded graph includes adjacent edges and a local loop to the
                    # oldest retained view. Upstream rejects non-overlapping edges.
                    del self.frames[:-WINDOW]
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
                        self.restart("failure")
                        return
                    self.remember(frame, pose)
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
            if restarted:
                self.restart(restarted)
            else:
                self.report(f"MASt3R-SLAM tracking | {self.tracked} frames")
