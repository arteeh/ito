import math

import numpy as np

from .tracking import Cloud, VisualResult

AXES = np.array([1.0, -1.0, -1.0])  # OpenCV camera/world axes (+Y down, +Z forward) to Ito.
LOCAL_WINDOW = 4
MIN_MATCH = 0.05
UNMATCHED_KEYFRAME = 3
NEWEST_PREFERENCE = math.radians(10)
RETRY_ANGLE = math.radians(30)
VISUAL_ROTATION = math.radians(0.5)
VISUAL_TRANSLATION = 0.02
SCALE_PRIOR = 0.1
UNKNOWN = np.full(6, np.inf)


def view_angle(a, b):
    cosine = float(np.clip(a[:3, 2] @ b[:3, 2], -1, 1))
    return math.acos(cosine)


class Keyframes(list):
    reference = None

    def last_keyframe(self):
        return self[-1 if self.reference is None else self.reference]

    def update_T_WCs(self, poses, indices):
        for transform, index in zip(poses, indices, strict=True):
            self[int(index)].T_WC = transform


class MASt3RFrontend:
    def __init__(
        self, intrinsics, report, *, precision="full", input_resolution=100, precision_report=None
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
        config["tracking"]["filtering_mode"] = "weighted_pointmap"
        config["local_opt"]["max_iters"] = 5
        self.config = config
        self.model = load_model(
            weights, report, precision=precision, precision_report=precision_report
        )
        self.lietorch = lietorch
        self.torch = torch
        self.frames = Keyframes()
        self.ids = []

        class DepthTracker(FrameTracker):
            def get_points_poses(self, frame, keyframe, indices, *args, **kwargs):
                self.depth_indices = indices
                return super().get_points_poses(frame, keyframe, indices, *args, **kwargs)

            def opt_pose_calib_sim3(self, *args, **kwargs):
                self.depth_valid = args[5]
                return super().opt_pose_calib_sim3(*args, **kwargs)

        self.tracker = DepthTracker(self.model, self.frames, "cuda")
        self.depth_match = None
        self.fused_pixels = 0
        if not hasattr(self.tracker, "prior"):
            raise RuntimeError("MASt3R bundle predates robot-first tracking; rebuild it")
        from .slam_image import SLAMImage

        self.image = SLAMImage(intrinsics, input_resolution)
        self.K = torch.as_tensor(self.image.calibration, device="cuda")
        self.axes = torch.tensor(AXES, device="cuda", dtype=torch.float32)
        self.frame_id = 0
        self.unmatched = 0
        self.tracked_against = None
        self.keyframe_count = 0

    def bind_admission(self, admission):
        from .admission import bind_model, bind_native

        self.admission_hooks = bind_model(self.model, admission)
        bind_native(admission)

    def placed(self, transform):
        matrix = transform.matrix()[0].double().cpu().numpy()
        scale = abs(np.linalg.det(matrix[:3, :3])) ** (1 / 3)
        if not np.isfinite(matrix).all() or not 0.01 < scale < 100:
            return None, scale
        # Sim3 rotations drift from orthonormal in float32; snap to the nearest one.
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

    def _angles(self, predicted):
        poses = [self.placed(kf.T_WC)[0] for kf in self.frames]
        return [math.inf if p is None else view_angle(p, predicted) for p in poses]

    def _reference(self, predicted):
        angles = self._angles(predicted)
        best = int(np.argmin(angles))
        if angles[-1] <= angles[best] + NEWEST_PREFERENCE:
            return len(self.frames) - 1
        return best

    def _retry(self, predicted, tried):
        angles = self._angles(predicted)
        angles[tried] = math.inf
        best = int(np.argmin(angles))
        return best if angles[best] <= RETRY_ANGLE else None

    def _attempt(self, frame, reference, predicted, prior_sigma):
        if self.ids[reference] != self.tracked_against:
            self.tracker.reset_idx_f2k()
            self.tracked_against = self.ids[reference]
        self.frames.reference = reference
        keyframe = self.frames[reference]
        keyframe_pose, keyframe_scale = self.placed(keyframe.T_WC)
        frame.T_WC = self.unplaced(predicted, keyframe_scale)
        self.tracker.prior = (frame.T_WC, self._prior_weight(prior_sigma, keyframe_scale))
        try:
            new_keyframe, _, lost = self.tracker.track(frame)
        finally:
            self.tracker.prior = None
            self.frames.reference = None
        match = 0.0 if lost else float(self.tracker.match_frac)
        pose, scale = (None, keyframe_scale) if lost else self.placed(frame.T_WC)
        usable = pose is not None and keyframe_pose is not None and match >= MIN_MATCH
        if usable:
            self.depth_match = (
                self.ids[reference],
                frame,
                self.tracker.depth_indices,
                self.tracker.depth_valid,
            )
        return new_keyframe, usable, match, keyframe_pose, keyframe_scale, pose, scale

    def _cloud(self, frame, scale):
        from mast3r_slam.geometry import constrain_points_to_ray

        torch = self.torch
        local = constrain_points_to_ray(frame.img.shape[-2:], frame.X_canon[None], self.K)[0]
        points = local * float(scale) * self.axes
        valid = torch.isfinite(points).all(dim=1) & (local[:, 2] > 0)
        colors = (frame.uimg.to("cuda").reshape(-1, 3) * 255).to(torch.uint8)
        fx, fy, cx, cy = (float(self.K[i, j]) for i, j in ((0, 0), (1, 1), (0, 2), (1, 2)))
        score = frame.get_average_conf().reshape(-1)
        confidence = torch.nan_to_num(1 - score.clamp_min(1).reciprocal()).clamp(0, 1)
        return Cloud(
            points,
            colors,
            valid,
            tuple(frame.img.shape[-2:]),
            (fx, fy, cx, cy),
            torch.where(valid, confidence, 0),
        )

    def _prior_weight(self, prior_sigma, scale):
        # Upstream's Sim3 tangent of the keyframe-from-frame pose: translation (in the
        # keyframe's units, metres over its scale), rotation, log-scale.
        sigma = np.concatenate(
            (
                np.maximum(prior_sigma[3:], 1e-3) / scale,
                np.maximum(prior_sigma[:3], 1e-4),
                [SCALE_PRIOR],
            )
        )
        return self.K.new_tensor(1 / sigma**2)

    def track(self, rgb, node, predicted, prior_sigma):
        from mast3r_slam.mast3r_utils import mast3r_inference_mono

        torch = self.torch
        self.depth_match = None
        with torch.inference_mode():
            frame = self.image.frame(
                self.frame_id,
                rgb,
                self.lietorch.Sim3.Identity(1, device="cuda"),
            )
            frame.K = self.K
            self.frame_id += 1
            if not self.frames:
                points, confidence = mast3r_inference_mono(self.model, frame)
                frame.update_pointmap(points, confidence)
                frame.T_WC = self.unplaced(predicted, 1.0)
                self._keep(frame, node)
                return VisualResult(None, None, UNKNOWN, 0.0, self._cloud(frame, 1.0), True)
            reference = self._reference(predicted)
            attempt = self._attempt(frame, reference, predicted, prior_sigma)
            if not attempt[1] and (other := self._retry(predicted, reference)) is not None:
                original = (frame.X_canon, frame.C, frame.N, frame.N_updates, frame.T_WC)
                retried = self._attempt(frame, other, predicted, prior_sigma)
                if retried[1]:
                    reference, attempt = other, retried
                else:
                    frame.X_canon, frame.C, frame.N, frame.N_updates, frame.T_WC = original
            new_keyframe, usable, match, keyframe_pose, keyframe_scale, pose, scale = attempt
            self.unmatched = 0 if usable else self.unmatched + 1
            if not usable:
                frame.T_WC = self.unplaced(predicted, keyframe_scale)
                scale = keyframe_scale
            becomes = bool(new_keyframe) and usable or self.unmatched >= UNMATCHED_KEYFRAME
            relative = np.linalg.inv(keyframe_pose) @ pose if usable else None
            if becomes:
                self._keep(frame, node)
                if usable:
                    relative = self._local_graph(reference)
                    _, scale = self.placed(frame.T_WC)
                self.unmatched = 0
            sigma = (
                np.repeat((VISUAL_ROTATION, VISUAL_TRANSLATION), 3) / math.sqrt(match)
                if relative is not None
                else UNKNOWN
            )
            return VisualResult(
                self.ids[reference], relative, sigma, match, self._cloud(frame, scale), becomes
            )

    def _keep(self, frame, node):
        self.frames.append(frame)
        self.ids.append(node)
        self.keyframe_count += 1
        self.tracker.reset_idx_f2k()
        self.tracked_against = None

    def _local_graph(self, reference):
        from mast3r_slam.global_opt import FactorGraph

        indices = list(range(max(0, len(self.frames) - LOCAL_WINDOW), len(self.frames)))
        if reference not in indices:
            indices.insert(0, reference)
        window = Keyframes(self.frames[i] for i in indices)
        last = len(window) - 1
        graph = FactorGraph(self.model, window, self.K, device="cuda")
        graph.add_factors(list(range(last)), list(range(1, last + 1)), 0.1)
        if last > 1:
            graph.add_factors([indices.index(reference)], [last], 0.1)
        graph.solve_GN_calib()
        self.tracker.reset_idx_f2k()
        reference_pose, _ = self.placed(self.frames[reference].T_WC)
        new_pose, _ = self.placed(self.frames[-1].T_WC)
        if reference_pose is None or new_pose is None:
            return None
        return np.linalg.inv(reference_pose) @ new_pose

    def correct(self, keyframe_poses, scale):
        with self.torch.inference_mode():
            for keyframe, node in zip(self.frames, self.ids, strict=True):
                _, own = self.placed(keyframe.T_WC)
                pose = keyframe_poses.get(node)
                if pose is None:
                    pose, _ = self.placed(keyframe.T_WC)
                if pose is not None:
                    keyframe.T_WC = self.unplaced(pose, own * scale)

    def refine(self, cloud, pose, reference):
        """Fuse accepted feature correspondences with the accumulated keyframe pointmap.

        A pointmap solve returns a new guess for the current image. Its reference has
        already seen many frames; carry that evidence onto the current pixel rays.
        This is in the shared visual gauge, before the independent metric calibration.
        """
        self.fused_pixels = 0
        if self.depth_match is None or reference not in self.ids:
            return cloud._replace(confidence=cloud.confidence * 0.25)
        node, frame, indices, valid = self.depth_match
        if node != reference:
            return cloud
        torch = self.torch
        with torch.inference_mode():
            keyframe = self.frames[self.ids.index(reference)]
            origin, scale = self.placed(keyframe.T_WC)
            points = self._cloud(keyframe, scale).points
            transform = torch.as_tensor(
                np.linalg.inv(pose) @ origin, device="cuda", dtype=torch.float32
            )
            projected = points @ transform[:3, :3].T + transform[:3, 3]
            z = -projected[:, 2]
            h, w = cloud.shape
            fx, fy, cx, cy = cloud.intrinsics
            u = fx * projected[:, 0] / z.clamp_min(1e-6) + cx
            v = cy - fy * projected[:, 1] / z.clamp_min(1e-6)
            indices = indices.reshape(-1).long()
            accepted = valid.reshape(-1) & torch.isfinite(projected).all(1) & (z > 0)
            accepted &= (indices >= 0) & (indices < h * w)
            indices = indices.clamp(0, h * w - 1)
            accepted &= cloud.valid[indices]
            candidates = torch.zeros(h * w, device="cuda")
            candidates.scatter_add_(0, indices, accepted.float())
            accepted &= (u - indices.remainder(w)).square() + (
                v - indices.div(w, rounding_mode="floor")
            ).square() < 9
            confidence = keyframe.get_average_conf().reshape(-1).clamp(0, 10)
            confidence = confidence * min(keyframe.N_updates, 4)
            confidence = torch.where(accepted, confidence, 0)
            weighted = torch.zeros(h * w, device="cuda")
            weight = torch.zeros_like(weighted)
            second = torch.zeros_like(weighted)
            weighted.scatter_add_(0, indices, torch.where(accepted, z * confidence, 0))
            second.scatter_add_(0, indices, torch.where(accepted, z.square() * confidence, 0))
            weight.scatter_add_(0, indices, confidence)
            current = frame.get_average_conf().reshape(-1).clamp(0.1, 10)
            depth = -cloud.points[:, 2]
            fused = (weighted + current * depth) / (weight + current)
            variance = (second + current * depth.square()) / (weight + current) - fused.square()
            variance = variance.clamp_min(0) / fused.square().clamp_min(1e-8)
            reliable = (weight > 0) & (variance < 0.15**2)
            certainty = torch.where(
                reliable,
                cloud.confidence * torch.exp(-variance / 0.15**2),
                torch.where(candidates > 0, 0, cloud.confidence * 0.25),
            )
            fused = torch.where(reliable, fused, depth)
            points = cloud.points * (fused / depth.clamp_min(1e-6))[:, None]
            self.fused_pixels = int(reliable.sum())
            # The retained frame now carries all contributing views, rather than
            # reverting to its first network prediction when it becomes a reference.
            _, own = self.placed(frame.T_WC)
            frame.X_canon = points * self.axes / own
            return cloud._replace(points=points, confidence=certainty)

    def forget(self, nodes):
        keep = [i for i, node in enumerate(self.ids) if node not in set(nodes)]
        self.frames[:] = [self.frames[i] for i in keep]
        self.ids[:] = [self.ids[i] for i in keep]
        if self.tracked_against in nodes:
            self.tracked_against = None
            self.tracker.reset_idx_f2k()
