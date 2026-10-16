import math
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

IMU_MATCH = 0.03
IMU_GAP = 0.06
TILT_SIGMA = math.radians(1.0)
TURN_SIGMA = math.radians(0.1)
RANGE_MATCH = 0.05
ZONE_SUPPORT = 0.6
ZONE_SPREAD = 0.15
ZONES = 4
SAMPLES = 5
AGREEMENT = 0.1


def ray(across, up):
    """Unit ray(s) at these angles right of and above the sensor's axis (-Z)."""
    return (np.cos(up) * np.sin(across), np.sin(up), -np.cos(up) * np.cos(across))


class HeadSensors:
    def __init__(self, seconds=10.0):
        self.seconds = seconds
        self.gyro = deque()
        self.tilts = deque()
        self.gyro_sigma = None
        self.grids = deque(maxlen=32)
        self.ranged = 0

    def add(self, imu, ranges):
        for unit in imu[:1]:
            rotation = unit["mount"][:3, :3]
            self.gyro_sigma = unit["gyro_sigma"]
            for when, spin, _, orientation in unit["samples"]:
                if self.gyro and when <= self.gyro[-1][0]:
                    continue
                self.gyro.append((when, rotation @ np.asarray(spin, dtype=float)))
                if orientation is not None:
                    world = Rotation.from_quat(orientation).as_matrix() @ rotation.T
                    self.tilts.append((when, world.T @ np.array([0.0, 1.0, 0.0])))
        for history in (self.gyro, self.tilts):
            while history and history[-1][0] - history[0][0] > self.seconds:
                history.popleft()
        for grid in ranges:
            self.grids.append(grid)

    def angular_velocity(self, t):
        if not self.gyro:
            return None
        when, spin = min(self.gyro, key=lambda sample: abs(sample[0] - t))
        return spin if abs(when - t) <= IMU_MATCH else None

    def turn(self, t0, t1):
        """The camera's rotation from t0 to t1 (camera at t0 from camera at t1) by its gyro,
        and its one-sigma error in radians; None where the IMU does not cover both."""
        if not self.gyro or t1 <= t0:
            return None
        times = np.fromiter((g[0] for g in self.gyro), float, len(self.gyro))
        if times[0] > t0 or times[-1] < t1 - IMU_GAP:
            return None
        first, last = np.searchsorted(times, (t0, t1))
        if first == 0:
            return None
        edges = np.concatenate(([t0], times[first:last], [t1]))
        if np.diff(edges).max() > IMU_GAP:
            return None
        rotation = Rotation.identity()
        for start, end in zip(edges[:-1], edges[1:], strict=True):
            # Each interval turns at the rate of the sample that ends it (gyros report the
            # rate over the period just read), the last one at the newest rate.
            spin = self.gyro[min(int(np.searchsorted(times, end)), len(times) - 1)][1]
            rotation = rotation * Rotation.from_rotvec(spin * (end - start))
        sigma = math.hypot(self.gyro_sigma * (t1 - t0), TURN_SIGMA)
        return rotation, sigma

    def up(self, t):
        if not self.tilts:
            return None
        when, up = min(self.tilts, key=lambda sample: abs(sample[0] - t))
        return (up, TILT_SIGMA) if abs(when - t) <= IMU_MATCH else None

    def range_scale(self, cloud, t, motion):
        if not self.grids:
            return None
        grid = min(self.grids, key=lambda g: abs(g["time"] - t))
        if abs(grid["time"] - t) > RANGE_MATCH:
            return None
        mount = grid["mount"]
        if motion.covers(t) and motion.covers(grid["time"]):
            mount = np.linalg.inv(motion.at(t)) @ motion.at(grid["time"]) @ mount
        rows, cols = grid["shape"]
        width, height = grid["field_of_view"]
        fx, fy, cx, cy = cloud.intrinsics
        image_h, image_w = cloud.shape
        targets, pixels = [], []
        steps = (np.arange(SAMPLES) + 0.5) / SAMPLES - 0.5
        for zone, distance in enumerate(grid["distances"]):
            if not math.isfinite(distance):
                continue
            row, col = divmod(zone, cols)
            across = ((col + 0.5) / cols - 0.5) * width
            up = (0.5 - (row + 0.5) / rows) * height
            a, u = np.meshgrid(across + steps * width / cols, up + steps * height / rows)
            rays = np.stack(ray(a.ravel(), u.ravel()), axis=-1)
            point = mount[:3, :3] @ (np.array(ray(across, up)) * distance) + mount[:3, 3]
            spots = rays * distance @ mount[:3, :3].T + mount[:3, 3]
            if point[2] > -0.05 or (spots[:, 2] > -0.05).any():
                continue
            x = fx * spots[:, 0] / -spots[:, 2] + cx
            y = cy - fy * spots[:, 1] / -spots[:, 2]
            if x.min() < 0 or x.max() > image_w - 1 or y.min() < 0 or y.max() > image_h - 1:
                continue
            targets.append(-point[2])
            pixels.append(np.rint(y).astype(int) * image_w + np.rint(x).astype(int))
        if len(targets) < ZONES:
            return None
        index = np.stack(pixels)
        depth = -cloud.points[index.ravel(), 2]
        valid = cloud.valid[index.ravel()]
        if hasattr(depth, "cpu"):
            depth, valid = depth.float().cpu().numpy(), valid.cpu().numpy()
        depth = np.where(valid & (depth > 0), depth, np.nan).reshape(index.shape)
        supported = np.isfinite(depth).mean(axis=1) >= ZONE_SUPPORT
        with np.errstate(all="ignore"):
            middle = np.nanmedian(depth, axis=1)
            low, high = np.nanpercentile(depth, (10, 90), axis=1)
        flat = supported & (middle > 0) & ((high - low) < ZONE_SPREAD * middle)
        if flat.sum() < ZONES:
            return None
        ratios = np.log(np.asarray(targets)[flat] / middle[flat])
        centre = float(np.median(ratios))
        if float(np.median(np.abs(ratios - centre))) > AGREEMENT:
            return None
        self.ranged += 1
        return math.exp(centre)
