import math
import os

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
# ITO_SLAM=full (mast3r_global.py): every keyframe kept in one global graph, upstream's
# weighted pointmap filtering, a reference chosen by place as well as direction, retrieval when
# tracking is lost. Unset: the frontend as before (local graph, forgetting keyframes).
FULL = os.environ.get("ITO_SLAM") == "full"
# Full: a keyframe a metre from the predicted camera counts like one turned this far away.
REACH = math.radians(30)
NEAR = 1.5  # metres: farther keyframes are only tried when retrieval names them


def view_angle(a, b):
    cosine = float(np.clip(a[:3, 2] @ b[:3, 2], -1, 1))
    return math.acos(cosine)


def _finite_pointmaps():
    """Weighted filtering keeps every update a keyframe's pointmap ever had: one non-finite
    prediction (a tracking step that blew up) would spoil it for good, and with it every
    global solve the keyframe is in. Points that are not finite count with zero confidence."""
    import torch
    from mast3r_slam.frame import Frame

    if getattr(Frame.update_pointmap, "finite", False):
        return
    original = Frame.update_pointmap

    def update_pointmap(self, X, C):
        ok = torch.isfinite(X).all(-1, keepdim=True) & torch.isfinite(C)
        original(self, torch.where(ok, X, 0), torch.where(ok, C, 0))
        self.X_canon = torch.where(self.C > 0, self.X_canon, 0)

    update_pointmap.finite = True
    Frame.update_pointmap = update_pointmap


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
        # ITO_SLAM_FILTER: how full SLAM keeps a keyframe's pointmap as frames track against it
        # (upstream's modes; an experiment: heavily used keyframes' edges have the worst tail).
        filtering = os.environ.get("ITO_SLAM_FILTER", "weighted_pointmap")
        config["tracking"]["filtering_mode"] = filtering if FULL else "recent"
        config["local_opt"]["max_iters"] = 5
        self.config = config
        if FULL:
            _finite_pointmaps()
        self.model = load_model(
            weights, report, precision=precision, precision_report=precision_report
        )
        self.lietorch = lietorch
        self.torch = torch
        self.frames = Keyframes()
        self.ids = []
        self.tracker = FrameTracker(self.model, self.frames, "cuda")
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
        self.global_graph = None  # set by SLAMBackend when ITO_SLAM=full
        self.reference = None  # full: index of the keyframe the last frame was tracked against
        self.last_scale = 1.0  # the scale the last returned cloud was made at
        self.lost = False  # full: the last keyframe was kept without a visual match
        self.relocalisations = 0
        self.last_match = None
        self.forced = 0  # keyframes kept because the pose graph's window was empty

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
        if not FULL:
            return [math.inf if p is None else view_angle(p, predicted) for p in poses]
        # Every keyframe is kept: one facing the same way from across the room shares nothing.
        scores = []
        for p in poses:
            distance = math.inf if p is None else float(np.linalg.norm(p[:3, 3] - predicted[:3, 3]))
            scores.append(
                math.inf if distance > NEAR else view_angle(p, predicted) + REACH * distance
            )
        return scores

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
        return new_keyframe, usable, match, keyframe_pose, keyframe_scale, pose, scale

    def _cloud(self, frame, scale):
        from mast3r_slam.geometry import constrain_points_to_ray

        torch = self.torch
        local = constrain_points_to_ray(frame.img.shape[-2:], frame.X_canon[None], self.K)[0]
        points = local * float(scale) * self.axes
        valid = torch.isfinite(points).all(dim=1) & (local[:, 2] > 0)
        colors = (frame.uimg.to("cuda").reshape(-1, 3) * 255).to(torch.uint8)
        fx, fy, cx, cy = (float(self.K[i, j]) for i, j in ((0, 0), (1, 1), (0, 2), (1, 2)))
        return Cloud(points, colors, valid, tuple(frame.img.shape[-2:]), (fx, fy, cx, cy))

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

    def track(self, rgb, node, predicted, prior_sigma, force=False):
        """force: keep this frame as a keyframe if it tracks (full SLAM, pose graph empty)."""
        from mast3r_slam.mast3r_utils import mast3r_inference_mono

        torch = self.torch
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
            tried = {reference}
            if not attempt[1] and (other := self._retry(predicted, reference)) is not None:
                tried.add(other)
                retried = self._attempt(frame, other, predicted, prior_sigma)
                if retried[1]:
                    reference, attempt = other, retried
            if not attempt[1] and self.global_graph is not None:
                # Relocalisation: the keyframes upstream's retrieval finds for this frame.
                for other in self.global_graph.query(frame):
                    if other in tried or other >= len(self.frames):
                        continue
                    tried.add(other)
                    retried = self._attempt(frame, other, predicted, prior_sigma)
                    if retried[1]:
                        reference, attempt = other, retried
                        self.relocalisations += 1
                        break
            new_keyframe, usable, match, keyframe_pose, keyframe_scale, pose, scale = attempt
            self.unmatched = 0 if usable else self.unmatched + 1
            self.last_match = match
            if not usable:
                frame.T_WC = self.unplaced(predicted, keyframe_scale)
                scale = keyframe_scale
            cloud = self._cloud(frame, scale)
            self.last_scale = float(scale)
            self.reference = reference
            becomes = (
                (bool(new_keyframe) or force) and usable or self.unmatched >= UNMATCHED_KEYFRAME
            )
            self.forced += bool(force and becomes)
            relative = np.linalg.inv(keyframe_pose) @ pose if usable else None
            if becomes:
                self._keep(frame, node)
                self.lost = not usable
                if usable and not FULL:
                    relative = self._local_graph(reference)
                self.unmatched = 0
            sigma = (
                np.repeat((VISUAL_ROTATION, VISUAL_TRANSLATION), 3) / math.sqrt(match)
                if relative is not None
                else UNKNOWN
            )
            return VisualResult(self.ids[reference], relative, sigma, match, cloud, becomes)

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
        owned = self.global_graph.owned if self.global_graph is not None else ()
        with self.torch.inference_mode():
            for index, (keyframe, node) in enumerate(zip(self.frames, self.ids, strict=True)):
                if index in owned:
                    continue  # the global graph poses it (and its scale) now
                _, own = self.placed(keyframe.T_WC)
                pose = keyframe_poses.get(node)
                if pose is None:
                    pose, _ = self.placed(keyframe.T_WC)
                if pose is not None:
                    keyframe.T_WC = self.unplaced(pose, own * scale)

    def forget(self, nodes):
        if FULL:
            return  # the global graph keeps every keyframe, by index
        keep = [i for i, node in enumerate(self.ids) if node not in set(nodes)]
        self.frames[:] = [self.frames[i] for i in keep]
        self.ids[:] = [self.ids[i] for i in keep]
        if self.tracked_against in nodes:
            self.tracked_against = None
            self.tracker.reset_idx_f2k()
