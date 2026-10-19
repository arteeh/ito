"""Vision-first tracking (ITO_SLAM=vision): MASt3R-SLAM's own tracker and keyframe graph place
the camera; the robot only predicts the next frame's turn, sets the metric scale and ties the
vision world to the robot's.

On recorded sim routes the images alone put co-visible frames within 0.07 degrees of each other
(original MASt3R-SLAM, calibrated), where robot-first fusion (fusion.py: visual edges at 0.5 degrees
against odometry with 1.5-3 degrees of gear-play bias, keyframes forgotten after 30) leaves 0.5-1.5
degrees with multi-degree glitches; a map is only sharp where every frame that saw a surface agrees
on it to a tenth of a degree.

Upstream's way, single-threaded as its WSL branch: every frame tracks against the newest keyframe
(Gauss-Newton on calibrated reprojection and log-depth), each keyframe's pointmap is the
confidence-weighted average of every frame tracked to it (weighted_pointmap), and each new keyframe
joins one Sim(3) graph of all keyframes (its predecessor plus older ones that face the same way
nearby, in place of upstream's retrieval) solved by upstream's calibrated Gauss-Newton with the
first keyframe pinned. Poses here are world-from-camera in Ito axes in the vision world (the first
keyframe's camera, MASt3R's units); slam.py maps them into the robot's world."""

import math
import os

import numpy as np

from .mast3r_frontend import AXES, Keyframes
from .tracking import Cloud

NEIGHBOURS = 3  # older keyframes linked to a new one, besides its predecessor
NEIGHBOUR_ANGLE = math.radians(35)
NEIGHBOUR_DISTANCE = 1.0  # vision units: metres at the start's scale
MIN_MATCH = 0.1  # upstream local_opt min_match_frac
# Each solve moves only the newest WINDOW keyframes; older ones linked to them hold still (pinned),
# and edges among old keyframes alone leave the graph. Upstream re-solves every keyframe on every
# new one: on the sim's loop route (62 keyframes) that took 2.5 s and 12 GB at its worst.
WINDOW = int(os.environ.get("ITO_VISION_WINDOW", "15"))


class VisionTracker:
    def __init__(
        self, intrinsics, report, *, precision="full", input_resolution=100, precision_report=None
    ):
        from .mast3r_runtime import load_model, prepare

        source, weights = prepare(report)
        import lietorch
        import torch
        from mast3r_slam.config import config, load_config
        from mast3r_slam.global_opt import FactorGraph
        from mast3r_slam.tracker import FrameTracker

        load_config(str(source / "config/base.yaml"))
        config["single_thread"] = True
        config["use_calib"] = True
        config["tracking"]["filtering_mode"] = "weighted_pointmap"
        self.config = config
        self.model = load_model(
            weights, report, precision=precision, precision_report=precision_report
        )
        self.lietorch, self.torch = lietorch, torch
        self.keyframes = Keyframes()
        self.tracker = FrameTracker(self.model, self.keyframes, "cuda")
        from .slam_image import SLAMImage

        self.image = SLAMImage(intrinsics, input_resolution)
        self.K = torch.as_tensor(self.image.calibration, device="cuda")
        self.graph = FactorGraph(self.model, self.keyframes, self.K, device="cuda")
        self.axes = torch.tensor(AXES, device="cuda", dtype=torch.float32)
        self.frame_id = 0
        self.previous = None  # last tracked frame's T_WC (lietorch Sim3)
        self.lost = 0
        self.solves = 0

    def bind_admission(self, admission):
        from .admission import bind_model, bind_native

        self.admission_hooks = bind_model(self.model, admission)
        bind_native(admission)

    def placed(self, transform):
        """World-from-camera (Ito axes, rigid) and scale of a lietorch Sim3, or (None, scale)."""
        matrix = transform.matrix()[0].double().cpu().numpy()
        scale = abs(np.linalg.det(matrix[:3, :3])) ** (1 / 3)
        if not np.isfinite(matrix).all() or not 1e-3 < scale < 1e3:
            return None, scale
        u, _, vh = np.linalg.svd(matrix[:3, :3])
        rigid = np.eye(4)
        rigid[:3, :3] = (u @ vh) * AXES[:, None] * AXES[None, :]
        rigid[:3, 3] = matrix[:3, 3] * AXES
        return rigid, scale

    def unplaced(self, pose, scale):
        from ito.render.pose import quaternion

        rotation = pose[:3, :3] * AXES[:, None] * AXES[None, :]
        data = [*(pose[:3, 3] * AXES), *quaternion(rotation), scale]
        return self.lietorch.Sim3(self.K.new_tensor(data)[None])

    def cloud(self, frame, scale):
        """The frame's pointmap in its camera frame (Ito axes), in vision-world units."""
        from mast3r_slam.geometry import constrain_points_to_ray

        torch = self.torch
        local = constrain_points_to_ray(frame.img.shape[-2:], frame.X_canon[None], self.K)[0]
        points = local * float(scale) * self.axes
        valid = torch.isfinite(points).all(dim=1) & (local[:, 2] > 0)
        colors = (frame.uimg.to("cuda").reshape(-1, 3) * 255).to(torch.uint8)
        fx, fy, cx, cy = (float(self.K[i, j]) for i, j in ((0, 0), (1, 1), (0, 2), (1, 2)))
        return Cloud(points, colors, valid, tuple(frame.img.shape[-2:]), (fx, fy, cx, cy))

    def track(self, rgb, turn):
        """Track one frame. turn: the robot's rotation of the camera since the previous tracked
        frame (3x3, previous-from-current, Ito camera axes) to seed the solve, or None.
        Returns (pose, scale, cloud, keyframe, tracked): pose world-from-camera (Ito axes, vision
        world), scale its Sim(3) scale; tracked False when the frame was dead-reckoned."""
        from mast3r_slam.mast3r_utils import mast3r_inference_mono

        torch = self.torch
        with torch.inference_mode():
            frame = self.image.frame(
                self.frame_id, rgb, self.lietorch.Sim3.Identity(1, device="cuda")
            )
            frame.K = self.K
            self.frame_id += 1
            if not self.keyframes:
                points, confidence = mast3r_inference_mono(self.model, frame)
                frame.update_pointmap(points, confidence)
                self.keyframes.append(frame)
                self.previous = frame.T_WC
                pose, scale = self.placed(frame.T_WC)
                return pose, scale, self.cloud(frame, scale), True, True
            last, scale = self.placed(self.previous)
            seed = last.copy()
            if turn is not None:
                seed[:3, :3] = last[:3, :3] @ turn
            frame.T_WC = self.unplaced(seed, scale)
            new_keyframe, _, lost = self.tracker.track(frame)
            if lost:
                # Dead reckoning on the robot's turn; after two in a row, start a new keyframe there
                # (its own mono inference) and link it into the graph like any other.
                self.lost += 1
                frame.T_WC = self.unplaced(seed, scale)
                if self.lost < 2:
                    self.previous = frame.T_WC
                    return seed, scale, None, False, False
                points, confidence = mast3r_inference_mono(self.model, frame)
                frame.update_pointmap(points, confidence)
                new_keyframe = True
            self.lost = 0
            if new_keyframe:
                self.keyframes.append(frame)
                self.link(len(self.keyframes) - 1)
                self.tracker.reset_idx_f2k()
            self.previous = frame.T_WC
            pose, scale = self.placed(frame.T_WC)
            return pose, scale, self.cloud(frame, scale), bool(new_keyframe), True

    def link(self, new):
        """Join keyframe new to the graph (its predecessor and older keyframes facing the same way
        nearby) and re-solve every keyframe."""
        if new == 0:
            return
        poses = [self.placed(kf.T_WC)[0] for kf in self.keyframes]
        mine = poses[new]
        scored = []
        for i, pose in enumerate(poses[: new - 1]):
            if pose is None or mine is None:
                continue
            angle = math.acos(float(np.clip(pose[:3, 2] @ mine[:3, 2], -1, 1)))
            distance = float(np.linalg.norm(pose[:3, 3] - mine[:3, 3]))
            if angle < NEIGHBOUR_ANGLE and distance < NEIGHBOUR_DISTANCE:
                scored.append((angle + distance, i))
        older = [i for _, i in sorted(scored)[:NEIGHBOURS]]
        ii = [new - 1] + older
        self.graph.add_factors(ii, [new] * len(ii), MIN_MATCH)
        self.prune(new)
        self.graph.solve_GN_calib()
        self.solves += 1

    def prune(self, new):
        """Keep the edges touching the newest WINDOW keyframes; pin the older ones they reach."""
        torch, graph = self.torch, self.graph
        first = max(0, new - WINDOW + 1)
        keep = (graph.ii >= first) | (graph.jj >= first)
        if not bool(keep.all()):
            for name in (
                "ii",
                "jj",
                "idx_ii2jj",
                "idx_jj2ii",
                "valid_match_j",
                "valid_match_i",
                "Q_ii2jj",
                "Q_jj2ii",
            ):
                setattr(graph, name, getattr(graph, name)[keep])
        unique = torch.unique(torch.cat([graph.ii, graph.jj]))
        graph.cfg["pin"] = max(1, int((unique < first).sum()))

    def forget(self, nodes):
        """Keyframes are kept for good (the worker calls this after an out-of-memory)."""

    def correct(self, keyframe_poses, scale):
        """The robot-first window's corrections do not apply here."""

    def keyframe_poses(self):
        return [self.placed(kf.T_WC) for kf in self.keyframes]
