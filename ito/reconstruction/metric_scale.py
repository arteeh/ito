"""One metric gauge from independent pixel parallax and robot motion, never floor height.

Pure rotation cannot observe metric scale: retain the visual gauge until triangulation
is sufficiently constrained. Only sparse depth samples cross to the CPU.
"""

import math
from collections import deque

import numpy as np


class MetricScale:
    def __init__(self, intrinsics):
        self.intrinsics = intrinsics
        self.log_scale = 0.0
        self.known = False
        self.observations = deque(maxlen=60)
        self.reference = None
        self.accepted = 0

    def apply(self, rgb, cloud, motion, exposure):
        import cv2

        h, w = cloud.shape
        fx, fy, cx, cy = cloud.intrinsics
        camera = self.intrinsics
        sx, sy = fx / camera.fx, fy / camera.fy
        affine = np.array([[sx, 0, cx - sx * camera.cx], [0, sy, cy - sy * camera.cy]])
        image = cv2.warpAffine(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), affine, (w, h))
        depth = -cloud.points[:, 2].reshape(h, w)
        self._observe(image, depth, motion.at(exposure), cloud.intrinsics, motion, exposure)
        if self.known:
            target = float(np.median(self.observations))
            self.log_scale += float(np.clip(target - self.log_scale, -0.005, 0.005))
        return cloud._replace(points=cloud.points * math.exp(self.log_scale))

    def _observe(self, image, depth, pose, intrinsics, motion, exposure):
        import cv2

        fx, fy, cx, cy = intrinsics
        if self.reference is None:
            corners = cv2.goodFeaturesToTrack(image, 1500, 0.01, 7)
            if corners is not None:
                self.reference = (image, corners, np.array(pose), exposure)
            return
        previous, corners, origin, reference_time = self.reference
        _, covariance = motion.between(reference_time, exposure)
        sigma = np.sqrt(np.diag(covariance))
        relative = np.linalg.inv(origin) @ pose
        baseline = np.linalg.norm(relative[:3, 3])
        rotation = np.arccos(np.clip((np.trace(relative[:3, :3]) - 1) / 2, -1, 1))
        if baseline < 0.015 and rotation < math.radians(20):
            return
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(previous, image, corners, None)
        if tracked is None:
            self.reference = None
            return
        back, reverse, _ = cv2.calcOpticalFlowPyrLK(image, previous, tracked, None)
        if back is None:
            self.reference = None
            return
        a, b = corners[:, 0], tracked[:, 0]
        good = status[:, 0].astype(bool) & reverse[:, 0].astype(bool)
        good &= np.linalg.norm(back[:, 0] - a, axis=1) < 0.5
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        # OpenCV camera axes; relative is reference-from-current in Ito axes.
        axes = np.diag([1.0, -1.0, -1.0])
        R, t = axes @ relative[:3, :3] @ axes, axes @ relative[:3, 3]
        angular_floor = 3 * np.max(sigma[:3]) * min(fx, fy)
        if good.sum() >= 30 and baseline > 3 * np.max(sigma[3:]):
            # Image geometry estimates the relative rotation more precisely than a
            # noisy camera prior. Odometry supplies the otherwise unobservable norm
            # of translation; its covariance still gates direction and rotation.
            essential, inliers = cv2.findEssentialMat(
                a[good], b[good], K, method=cv2.RANSAC, prob=0.999, threshold=0.5
            )
            if essential is not None and essential.shape == (3, 3):
                count, rotation, direction, _ = cv2.recoverPose(
                    essential, a[good], b[good], K, mask=inliers
                )
                candidate_R = rotation.T
                candidate_t = -rotation.T @ direction[:, 0] * baseline
                angle = math.acos(np.clip((np.trace(R.T @ candidate_R) - 1) / 2, -1, 1))
                moved = np.sum((candidate_t - t) ** 2 / np.maximum(sigma[3:] ** 2, 1e-12))
                if count >= 30 and angle < 3 * np.linalg.norm(sigma[:3]) and moved < 16.27:
                    R, t = candidate_R, candidate_t
                    angular_floor = 3.0
        P0 = K @ np.column_stack((np.eye(3), np.zeros(3)))
        P1 = K @ np.column_stack((R.T, -R.T @ t))
        xyz = cv2.triangulatePoints(P0, P1, a.T, b.T).T
        xyz = xyz[:, :3] / np.where(np.abs(xyz[:, 3:]) > 1e-9, xyz[:, 3:], np.nan)
        current = (xyz - t) @ R
        rays = np.column_stack(((b[:, 0] - cx) / fx, (b[:, 1] - cy) / fy, np.ones(len(b))))
        rotated = rays @ R.T
        measured = np.column_stack(((a[:, 0] - cx) / fx, (a[:, 1] - cy) / fy, np.ones(len(a))))
        parallax = np.linalg.norm(rotated[:, :2] / rotated[:, 2:] - measured[:, :2], axis=1)
        projection = xyz @ K.T
        error = np.linalg.norm(projection[:, :2] / projection[:, 2:] - a, axis=1)
        h, w = image.shape
        xy = np.rint(b).astype(int)
        good &= (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        xy = np.clip(xy, [0, 0], [w - 1, h - 1])
        # Sparse gather only; the dense depth image never leaves the device.
        z = depth[xy[:, 1], xy[:, 0]].cpu().numpy()
        good &= (xyz[:, 2] > 0) & (current[:, 2] > 0) & (z > 0) & (error < 0.75)
        good &= parallax * min(fx, fy) > max(3, angular_floor)
        good &= np.isfinite(current).all(axis=1)
        ratios = np.log(current[good, 2] / z[good])
        if len(ratios) >= 30 and baseline > 3 * np.max(sigma[3:]):
            center = float(np.median(ratios))
            mad = float(np.median(np.abs(ratios - center)))
            inliers = np.abs(ratios - center) < max(0.1, 2.5 * mad)
            if inliers.sum() >= 30 and mad < 0.25:
                value = float(np.median(ratios[inliers]))
                consensus = float(np.median(self.observations)) if self.observations else value
                if not self.known or abs(value - consensus) < 0.4:
                    self.observations.append(value)
                    target = float(np.median(self.observations))
                    if self.known or (
                        len(self.observations) >= 3
                        and np.median(np.abs(np.asarray(self.observations) - target)) <= 0.1
                    ):
                        self.known = True
                        self.accepted += 1
        # Keep the long baseline while it is trackable; replace on large turns/loss.
        if rotation > math.radians(20) or good.sum() < 30 or baseline > 0.15:
            self.reference = None
