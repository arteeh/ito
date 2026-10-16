import math
import os

import numpy as np

from ito import clock

from .rgbd import RGBDBackend

HEADING_TOLERANCE = math.radians(12)
HEADING_SETTLED = math.radians(2)
HEADING_EASE = 1 / 3
HEADING_LOST = math.radians(60)
LOSS_SECONDS = 1.0
SPREAD = math.radians(30)
WINDOW = 4
RECALL = os.environ.get("ITO_SLAM_RECALL", "1") != "0"
RESTART_CAUSES = ("unmatched", "heading", "failure")
REASONS = dict(unmatched="tracking lost", heading="heading lost", failure="solve failed")
AXES = np.array([1.0, -1.0, -1.0])  # OpenCV camera/world axes (+Y down, +Z forward) to Ito.


def heading(rotation):
    return math.atan2(rotation[0, 2], rotation[2, 2])


def turn(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)))


class Keyframes(list):
    def last_keyframe(self):
        return self[-1]

    def update_T_WCs(self, poses, indices):
        for transform, index in zip(poses, indices, strict=True):
            self[int(index)].T_WC = transform


class VisionSLAMBackend(RGBDBackend):
    def bind_admission(self, admission):
        from .admission import bind_model, bind_native

        super().bind_admission(admission)
        self.admission_hooks = bind_model(self.model, admission)
        bind_native(admission)

    def __init__(
        self,
        capacity,
        intrinsics,
        *,
        report,
        precision="full",
        input_resolution=100,
        precision_report=None,
        **options,
    ):
        from .mast3r_runtime import load_model, prepare

        source, weights = prepare(report)
        import lietorch
        import torch
        from mast3r_slam.config import config, load_config
        from mast3r_slam.tracker import FrameTracker

        load_config(str(source / "config/base.yaml"))
        config["single_thread"] = True
        config["use_calib"] = True
        config["tracking"]["filtering_mode"] = "recent"
        config["local_opt"]["max_iters"] = 5
        self.model = load_model(
            weights, report, precision=precision, precision_report=precision_report
        )
        options["device"] = "cuda"
        super().__init__(capacity, intrinsics, **options)
        self.report = report
        self.lietorch = lietorch
        self.frames = Keyframes()
        self.tracker = FrameTracker(self.model, self.frames, "cuda")
        from .slam_image import SLAMImage

        self.image = SLAMImage(intrinsics, input_resolution)
        self.K = torch.as_tensor(self.image.calibration, device="cuda")
        self.transform = lietorch.Sim3.Identity(1, device="cuda")
        self.camera_pose = np.eye(4, dtype=np.float32)
        self.tracked = 0
        self.restarts = dict.fromkeys(RESTART_CAUSES, 0)
        self.corrections = 0
        self.recalls = 0
        self.correcting = False
        self.frame_id = 0
        self.lost = False
        self.failures = 0
        self.unmatched_since = None
        self.spread = {}
        self.world = None
        self.anchor = None
        self.axes = torch.tensor(AXES, device="cuda", dtype=torch.float32)
        report("MASt3R-SLAM ready; waiting for camera")

    def placed(self, transform):
        matrix = transform.matrix()[0].double().cpu().numpy()
        scale = abs(np.linalg.det(matrix[:3, :3])) ** (1 / 3)
        if not np.isfinite(matrix).all() or not 0.01 < scale < 100:
            return None, scale
        u, _, vh = np.linalg.svd(matrix[:3, :3])
        rigid = np.eye(4)
        rigid[:3, :3] = (u @ vh) * AXES[:, None] * AXES[None, :]
        rigid[:3, 3] = matrix[:3, 3] * AXES
        return (rigid if self.world is None else self.world @ rigid), scale

    def unplaced(self, pose, scale):
        from ito.render.pose import quaternion

        rigid = np.linalg.inv(self.world) @ pose
        rotation = rigid[:3, :3] * AXES[:, None] * AXES[None, :]
        data = [*(rigid[:3, 3] * AXES), *quaternion(rotation), scale]
        return self.lietorch.Sim3(self.K.new_tensor(data)[None])

    def predicted(self, prior, fallback):
        start, robot = self.anchor
        rotation = prior[:3, :3] @ robot[:3, :3].T @ start[:3, :3]
        rotation = turn(heading(prior[:3, :3]) - heading(rotation)) @ rotation
        result = fallback.copy()
        result[:3, :3] = rotation
        return result

    def start_map(self, frame, pose, prior):
        self.frames.clear()
        self.frames.append(frame)
        if not RECALL:
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

    def remember(self, frame, pose):
        sectors = round(math.tau / SPREAD)
        self.spread[round(heading(pose[:3, :3]) / SPREAD) % sectors] = frame

    def relocalize(self, prior):
        self.tracker.reset_idx_f2k()
        looking = heading(prior[:3, :3])

        def away(keyframe):
            pose, _ = self.placed(keyframe.T_WC)
            if pose is None:
                return math.inf
            return abs(math.remainder(heading(pose[:3, :3]) - looking, math.tau))

        best = min(self.spread.values(), key=away, default=None)
        if best is not None and away(best) + SPREAD / 2 < away(self.frames[-1]):
            self.frames[:] = [best]
            return True
        return False

    def steer(self, pose, stray):
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
        import torch
        from mast3r_slam.geometry import constrain_points_to_ray
        from mast3r_slam.global_opt import FactorGraph
        from mast3r_slam.mast3r_utils import mast3r_inference_mono

        started = clock.now()
        prior = np.asarray(camera, dtype=np.float64)
        restarted = None
        with torch.inference_mode():
            frame = self.image.frame(self.frame_id, rgb, self.transform)
            frame.K = self.K
            self.frame_id += 1
            if self.failures >= max(len(self.frames), 1) + 2:
                self.frames.clear()
                self.failures = 0
                restarted = "unmatched"
            if not self.frames:
                points, confidence = mast3r_inference_mono(self.model, frame)
                frame.update_pointmap(points, confidence)
                pose, scale = self.placed(frame.T_WC)
                if self.world is None:
                    self.world = prior @ np.linalg.inv(pose)
                    pose = prior.copy()
                elif measured and pose is not None:
                    pose = pose.copy()
                    pose[:3, :3] = (
                        turn(heading(prior[:3, :3]) - heading(pose[:3, :3])) @ pose[:3, :3]
                    )
                    frame.T_WC = self.unplaced(pose, scale)
                self.start_map(frame, pose, prior)
            else:
                if self.lost and not measured:
                    self.tracker.reset_idx_f2k()
                    self.frames.insert(0, self.frames.pop())
                    frame.T_WC = self.frames[-1].T_WC
                previous, previous_scale = self.placed(self.transform) if measured else (None, 0)
                relocalized = False
                for attempt in range(2 if measured and RECALL else 1):
                    if measured and (self.unmatched_since is not None or attempt):
                        frame.T_WC = self.unplaced(self.predicted(prior, previous), previous_scale)
                    new_keyframe, _, lost = self.tracker.track(frame)
                    pose, scale = (None, 0) if lost else self.placed(frame.T_WC)
                    if not measured:
                        break
                    cause = "unmatched" if lost else "failure" if pose is None else None
                    stray = 0.0
                    if pose is not None:
                        expected = self.predicted(prior, pose)
                        stray = math.remainder(
                            heading(pose[:3, :3]) - heading(expected[:3, :3]), math.tau
                        )
                        if abs(stray) > HEADING_LOST:
                            cause = "heading"
                    if attempt and not cause:
                        self.recalls += 1
                    if cause != "unmatched" or attempt or not RECALL:
                        break
                    relocalized = True
                    if not self.relocalize(prior):
                        break
                if measured:
                    if cause and self.unmatched_since is None:
                        self.unmatched_since = now
                    if cause and now - self.unmatched_since < LOSS_SECONDS:
                        if not relocalized:
                            self.relocalize(prior)
                        return
                    if cause:
                        if pose is None:
                            pose, scale = self.predicted(prior, previous), previous_scale
                        else:
                            pose = pose.copy()
                            pose[:3, :3] = turn(-stray) @ pose[:3, :3]
                        frame.T_WC = self.unplaced(pose, scale)
                        if cause == "heading":
                            self.spread.clear()
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
                if new_keyframe:
                    self.frames.append(frame)
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
            valid = torch.isfinite(points).all(dim=1) & (local[:, 2] > 0)
            colors = (frame.uimg.to("cuda").reshape(-1, 3) * 255).to(torch.uint8)
            pixels = 3 * max(1, (400_000 / self.budget) ** 0.5)
            footprint = local[:, 2] * float(scale) * pixels / self.K[0, 0]
            when = now + clock.now() - started
            self.integrate_points(
                self.xp.from_dlpack(points[valid].contiguous()),
                self.xp.from_dlpack(colors[valid].contiguous()),
                when,
                self.xp.from_dlpack(footprint[valid].contiguous()),
            )
            height, width = frame.img.shape[-2:]
            depth = torch.where(valid, local[:, 2] * float(scale), 0).reshape(height, width)
            fx, fy, cx, cy = (float(self.K[i, j]) for i, j in ((0, 0), (1, 1), (0, 2), (1, 2)))
            self.carve(
                self.camera_pose,
                fx,
                fy,
                cx,
                cy,
                self.xp.from_dlpack(depth.float().contiguous()),
                when,
            )
            self.tracked += 1
            if restarted:
                self.restart(restarted)
            else:
                self.report(f"MASt3R-SLAM tracking | {self.tracked} frames")
