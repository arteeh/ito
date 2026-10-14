"""MASt3R-SLAM's tracker and local graph as a visual front end (tracking.Frontend).

Its map is fusion's world, in metres: every keyframe sits where fusion last put it. Each
frame is seeded at the pose fusion predicts and solved with that prediction as a prior
inside MASt3R's own Gauss-Newton step (Ito's adaptation of upstream's FrameTracker.solve),
so a plain wall or a blurred pan yields a pose leaning on the robot instead of "lost". A
frame that matches too little to say anything becomes a keyframe where fusion places it,
and later frames track against that: there is no map to restart. The reference keyframe
is the retained one that faced the predicted view most nearly, the newest when it is close.
"""

import math

import numpy as np

from .tracking import Cloud, VisualResult

AXES = np.array([1.0, -1.0, -1.0])  # OpenCV camera/world axes (+Y down, +Z forward) to Ito.
LOCAL_WINDOW = 4  # keyframes in MASt3R's own local graph
# Under this match fraction a solve only echoes its prior: no visual factor. Upstream
# declares such a frame lost.
MIN_MATCH = 0.05
# This many such frames in a row and the next becomes a keyframe at fusion's pose.
UNMATCHED_KEYFRAME = 3
# An older keyframe becomes the reference when it faces the predicted view this much closer
# than the newest one: coming back to a view ties the frame to where it was placed before.
NEWEST_PREFERENCE = math.radians(10)
# A frame the reference does not match is tried once more against the other keyframe that
# faces its predicted view most nearly, within this: after a turn too fast to follow, the
# view is often one an older keyframe in fusion's window saw.
RETRY_ANGLE = math.radians(30)
# A fully matched relative pose, one sigma; fewer matches widen it by 1/sqrt(match).
VISUAL_ROTATION = math.radians(0.5)
VISUAL_TRANSLATION = 0.02
SCALE_PRIOR = 0.1  # The prior's log-scale sigma: MASt3R keeps its own scale between frames.
UNKNOWN = np.full(6, np.inf)


def view_angle(a, b):
    """Angle between two world-from-camera poses' viewing directions (-Z)."""
    cosine = float(np.clip(a[:3, 2] @ b[:3, 2], -1, 1))
    return math.acos(cosine)


class Keyframes(list):
    """Upstream's keyframe store: the tracker reads last_keyframe(), the graph writes poses."""

    reference = None  # index tracked against; None: the newest

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
        # Ito always knows the camera's intrinsics; calibrated tracking and points
        # on the true pixel rays keep straight edges straight.
        config["use_calib"] = True
        config["tracking"]["filtering_mode"] = "recent"
        config["local_opt"]["max_iters"] = 5
        self.config = config
        self.model = load_model(
            weights, report, precision=precision, precision_report=precision_report
        )
        self.lietorch = lietorch
        self.torch = torch
        self.frames = Keyframes()
        self.ids = []  # node id of each keyframe, parallel to frames
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

    def bind_admission(self, admission):
        from .admission import bind_model, bind_native

        self.admission_hooks = bind_model(self.model, admission)
        bind_native(admission)

    # -- poses between MASt3R's Sim3 map and Ito ----------------------------------------

    def placed(self, transform):
        """The Ito world-from-camera pose of a map Sim3 and its scale, or None if unusable."""
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
        """The map's Sim3 for an Ito world-from-camera pose."""
        from ito.render.pose import quaternion

        rotation = pose[:3, :3] * AXES[:, None] * AXES[None, :]
        data = [*(pose[:3, 3] * AXES), *quaternion(rotation), scale]
        return self.lietorch.Sim3(self.K.new_tensor(data)[None])

    # -- tracking -----------------------------------------------------------------------

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
        """The keyframe besides `tried` facing the predicted view most nearly, if close."""
        angles = self._angles(predicted)
        angles[tried] = math.inf
        best = int(np.argmin(angles))
        return best if angles[best] <= RETRY_ANGLE else None

    def _attempt(self, frame, reference, predicted, prior_sigma):
        """Track the frame against one keyframe: (new keyframe, lost, keyframe pose and
        scale, the frame's pose and scale)."""
        if self.ids[reference] != self.tracked_against:
            # Upstream reuses the last frame's matches as the next search's start.
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

    def track(self, rgb, node, predicted, prior_sigma):
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
            if not attempt[1] and (other := self._retry(predicted, reference)) is not None:
                retried = self._attempt(frame, other, predicted, prior_sigma)
                if retried[1]:
                    reference, attempt = other, retried
            new_keyframe, usable, match, keyframe_pose, keyframe_scale, pose, scale = attempt
            self.unmatched = 0 if usable else self.unmatched + 1
            if not usable:
                frame.T_WC = self.unplaced(predicted, keyframe_scale)
                scale = keyframe_scale
            cloud = self._cloud(frame, scale)
            becomes = bool(new_keyframe) and usable or self.unmatched >= UNMATCHED_KEYFRAME
            relative = np.linalg.inv(keyframe_pose) @ pose if usable else None
            if becomes:
                self._keep(frame, node)
                if usable:
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
        """MASt3R's own bundle adjustment over the newest keyframes and the reference.

        Returns the new keyframe's pose relative to the reference after it.
        """
        from mast3r_slam.global_opt import FactorGraph

        indices = list(range(max(0, len(self.frames) - LOCAL_WINDOW), len(self.frames)))
        if reference not in indices:
            indices.insert(0, reference)
        window = Keyframes(self.frames[i] for i in indices)
        last = len(window) - 1
        graph = FactorGraph(self.model, window, self.K, device="cuda")
        # Neighbours in time, and the reference to the new keyframe; upstream drops edges
        # whose views do not overlap. The first keyframe stays pinned.
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
        """Fused poses back into MASt3R's keyframes, so its references agree with fusion."""
        with self.torch.inference_mode():
            for keyframe, node in zip(self.frames, self.ids, strict=True):
                _, own = self.placed(keyframe.T_WC)
                pose = keyframe_poses.get(node)
                if pose is None:
                    pose, _ = self.placed(keyframe.T_WC)
                if pose is not None:
                    keyframe.T_WC = self.unplaced(pose, own * scale)

    def forget(self, nodes):
        keep = [i for i, node in enumerate(self.ids) if node not in set(nodes)]
        self.frames[:] = [self.frames[i] for i in keep]
        self.ids[:] = [self.ids[i] for i in keep]
        if self.tracked_against in nodes:
            self.tracked_against = None
            self.tracker.reset_idx_f2k()
