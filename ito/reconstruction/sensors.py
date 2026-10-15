"""Optional, source-timed inertial orientation and coarse metric range evidence.

Only sparse depth gathers cross the device boundary. Range zones retain their angular
footprints; mixed-depth patches and stale/unregistered measurements constrain nothing.
"""

import math
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

UP = np.array([0.0, 1.0, 0.0])


class Sensors:
    def __init__(self):
        self.imu = deque(maxlen=6000)
        self.ranges = deque(maxlen=240)
        self.last = {}
        self.yaw_origin = None
        self.range_updates = 0
        self.range_used = None

    def extend(self, observations):
        for kind, samples, history in (
            ("imu", observations.imu, self.imu),
            ("range", observations.ranges, self.ranges),
        ):
            for sample in samples:
                key = kind, sample.sensor
                if sample.time <= self.last.get(key, -math.inf):
                    continue
                self.last[key] = sample.time
                if kind == "range":
                    history.append(sample)
                    continue
                mount = Rotation.from_quat(sample.mount.orientation)
                spin = mount.apply(sample.angular_velocity)
                force = mount.apply(sample.specific_force)
                if sample.orientation is not None:
                    rotation = Rotation.from_quat(sample.orientation) * mount.inv()
                    if self.yaw_origin is None:
                        matrix = rotation.as_matrix()
                        yaw = math.atan2(matrix[0, 2], matrix[2, 2])
                        self.yaw_origin = Rotation.from_rotvec(-UP * yaw)
                    rotation = self.yaw_origin * rotation
                elif history:
                    dt = sample.time - history[-1][0]
                    if dt > 0.1:
                        continue  # No integration across an unobserved sensor gap.
                    rotation = history[-1][1] * Rotation.from_rotvec(spin * dt)
                else:
                    # Gravity initializes tilt only near rest; yaw remains unobserved.
                    rotation = Rotation.identity()
                    if abs(np.linalg.norm(force) - 9.81) < 1:
                        rotation, _ = Rotation.align_vectors([UP], [force])
                history.append((sample.time, rotation, spin, force, sample))

    def covers(self, time, slack=0.05):
        return bool(self.imu) and self.imu[0][0] <= time <= self.imu[-1][0] + slack

    def at(self, time):
        """Camera-to-gravity-world rotation, camera gyro, and its source uncertainty."""
        times = np.array([s[0] for s in self.imu])
        index = min(int(np.searchsorted(times, time)), len(times) - 1)
        if index and abs(times[index - 1] - time) < abs(times[index] - time):
            index -= 1
        sample = self.imu[index]
        if time >= times[-1]:
            rotation = sample[1] * Rotation.from_rotvec(sample[2] * min(time - times[-1], 0.05))
        elif time <= times[0] or len(times) == 1:
            rotation = sample[1]
        else:
            upper = int(np.searchsorted(times, time))
            pair = [self.imu[upper - 1], self.imu[upper]]
            rotation = Slerp([p[0] for p in pair], Rotation.concatenate([p[1] for p in pair]))(
                [time]
            )[0]
        return rotation, sample[2], sample[4]

    def range_scale(self, cloud, time, motion):
        """A robust log metric scale, or None when the zones cannot constrain this view.

        Project each measured footprint at its actual range, including the mount's
        translation. Gather 5x5 samples per zone once; a depth discontinuity invalidates
        that zone rather than lending the sparse sensor invented object boundaries.
        """
        if not self.ranges:
            return None
        sample = min(self.ranges, key=lambda s: abs(s.time - time))
        if (sample.sensor, sample.time) == self.range_used:
            return None
        if abs(sample.time - time) > 0.04:
            return None
        mount = np.eye(4)
        mount[:3, :3] = Rotation.from_quat(sample.mount.orientation).as_matrix()
        mount[:3, 3] = sample.mount.position
        if len(motion.times) and motion.covers(time) and motion.covers(sample.time):
            mount = np.linalg.inv(motion.at(time)) @ motion.at(sample.time) @ mount
        elif self.covers(time) and self.covers(sample.time):
            relative = self.at(time)[0].inv() * self.at(sample.time)[0]
            mount[:3, :] = relative.as_matrix() @ mount[:3, :]
        elif abs(sample.time - time) > 0.01:
            return None
        fx, fy, cx, cy = cloud.intrinsics
        height, width = cloud.shape
        patches, targets = [], []
        for zone in sample.zones:
            if zone.quality != "valid" or zone.sigma / zone.distance > 0.1:
                continue
            ray = np.array(zone.direction)
            tangent = np.cross(ray, [0, 0, 1])
            if np.linalg.norm(tangent) < 1e-6:
                tangent = np.cross(ray, [0, 1, 0])
            tangent /= np.linalg.norm(tangent)
            vertical = np.cross(tangent, ray)
            footprint = np.array(
                [
                    ray
                    + x * math.tan(zone.angular_size[0] / 2) * tangent
                    + y * math.tan(zone.angular_size[1] / 2) * vertical
                    for x, y in ((-1, -1), (-1, 1), (1, -1), (1, 1))
                ]
            )
            corners = (footprint * zone.distance) @ mount[:3, :3].T + mount[:3, 3]
            point = mount[:3, :3] @ (ray * zone.distance) + mount[:3, 3]
            if point[2] >= -0.02 or (corners[:, 2] >= -0.02).any():
                continue
            u = fx * corners[:, 0] / -corners[:, 2] + cx
            v = cy - fy * corners[:, 1] / -corners[:, 2]
            # Require the whole footprint in view. Borders offer a biased subset.
            if u.min() < 0 or u.max() >= width or v.min() < 0 or v.max() >= height:
                continue
            xs = np.linspace(u.min(), u.max(), 7)[1:-1]
            ys = np.linspace(v.min(), v.max(), 7)[1:-1]
            xx, yy = np.meshgrid(xs, ys)
            patches.append(np.rint(yy).astype(int) * width + np.rint(xx).astype(int))
            targets.append(-point[2])
        if len(patches) < 6:
            return None
        indices = np.array(patches).reshape(len(patches), -1)
        # These are sparse gathers, not a dense device-to-host depth copy.
        depth = -cloud.points[indices, 2]
        valid = cloud.valid[indices]
        if hasattr(depth, "detach"):
            depth, valid = depth.detach().cpu().numpy(), valid.detach().cpu().numpy()
        depth = np.where(valid & np.isfinite(depth) & (depth > 0), depth, np.nan)
        with np.errstate(invalid="ignore"):
            middle = np.nanmedian(depth, axis=1)
            low, high = np.nanquantile(depth, [0.1, 0.9], axis=1)
        good = np.isfinite(middle) & (middle > 0) & ((high - low) < 0.15 * middle)
        ratios = np.log(np.array(targets)[good] / middle[good])
        if len(ratios) < 6:
            return None
        center = float(np.median(ratios))
        spread = float(np.median(np.abs(ratios - center)))
        if spread > 0.15:
            return None
        self.range_used = sample.sensor, sample.time
        self.range_updates += 1
        return center
