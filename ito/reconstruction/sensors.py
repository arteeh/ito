"""The robot's camera-mounted sensors, on the pilot clock: a head IMU and a multizone ToF.

Both are optional: a robot without them sends none, and tracking works from the camera and the
robot's motion alone, as before.

IMU: it rides on the camera's own link, so it measures the camera's turns directly, where the
robot's motion samples infer them through every joint between trunk and camera, gear play and
all. Its gyro, turned into the camera's axes by the mount and integrated between two frames'
exposures, is how the camera turned between them, good to its bias times the interval; its
own fusion's gravity says how the camera is tilted. Both become factors in fusion.py, tighter
than the robot's where the IMU covers the frames. The gyro at an exposure is also how fast the
camera turned then, which the blur gate (slam.py) believes before the motion samples.

ToF: each zone's range, along its ray, is a metric depth at a known spot in the frame. The
front end's points there, the zone's footprint of them, are in the front end's units; the
ratio is metres per unit, the same quantity the floor under the camera measures (slam.py),
but anywhere the sensor looks: a wall, furniture, a frame without floor. Zones whose
footprint straddles a depth edge (near and far points both) say nothing and are skipped.
"""

import math
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

# How far apart in time a sensor reading and a frame's exposure may be.
IMU_MATCH = 0.03
# An IMU gap longer than this is unobserved rotation: no factor spans it.
IMU_GAP = 0.06
# A chip's own gravity fusion is off by about this during a walk's accelerations; its gyro
# integration by this much on top of bias x time.
TILT_SIGMA = math.radians(1.0)
TURN_SIGMA = math.radians(0.1)
RANGE_MATCH = 0.05
# A zone counts when this share of its footprint's samples has a point, their depths spread
# less than this fraction (a flat patch, not an edge), and at least this many zones agree.
ZONE_SUPPORT = 0.6
ZONE_SPREAD = 0.15
ZONES = 4
SAMPLES = 5  # per side of a zone's footprint
# Zones' log-ratios may disagree by this much (median absolute deviation) and still be one
# scale; past it the front end's points are bent, or the sensor saw something else.
AGREEMENT = 0.1


def ray(across, up):
    """Unit ray(s) at these angles right of and above the sensor's axis (-Z)."""
    return (np.cos(up) * np.sin(across), np.sin(up), -np.cos(up) * np.cos(across))


class HeadSensors:
    def __init__(self, seconds=10.0):
        self.seconds = seconds
        self.gyro = deque()  # (time, rad/s in camera axes)
        self.tilts = deque()  # (time, up in camera axes) from the chip's fusion
        self.gyro_sigma = None
        self.grids = deque(maxlen=32)
        self.ranged = 0  # frames the ToF gave a scale

    def add(self, imu, ranges):
        """imu and ranges as Pilot._sensors sends them."""
        for unit in imu[:1]:  # One IMU per camera is all fusion takes.
            rotation = unit["mount"][:3, :3]
            self.gyro_sigma = unit["gyro_sigma"]
            for when, spin, _, orientation in unit["samples"]:
                if self.gyro and when <= self.gyro[-1][0]:
                    continue
                self.gyro.append((when, rotation @ np.asarray(spin, dtype=float)))
                if orientation is not None:
                    # World-from-sensor, yaw arbitrary: the world's up in the camera.
                    world = Rotation.from_quat(orientation).as_matrix() @ rotation.T
                    self.tilts.append((when, world.T @ np.array([0.0, 1.0, 0.0])))
        for history in (self.gyro, self.tilts):
            while history and history[-1][0] - history[0][0] > self.seconds:
                history.popleft()
        for grid in ranges:
            self.grids.append(grid)

    def angular_velocity(self, t):
        """Camera-frame rad/s at t from the IMU, or None if it read nothing near then."""
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
        """The world's up in the camera at t by the chip's own gravity fusion, and its sigma;
        None if it fused nothing near then."""
        if not self.tilts:
            return None
        when, up = min(self.tilts, key=lambda sample: abs(sample[0] - t))
        return (up, TILT_SIGMA) if abs(when - t) <= IMU_MATCH else None

    def range_scale(self, cloud, t, motion):
        """Metres per front-end unit in a frame's cloud, exposed at t, from the ToF reading
        nearest it; None if no reading is close enough in time or too few zones agree."""
        if not self.grids:
            return None
        grid = min(self.grids, key=lambda g: abs(g["time"] - t))
        if abs(grid["time"] - t) > RANGE_MATCH:
            return None
        mount = grid["mount"]
        if motion.covers(t) and motion.covers(grid["time"]):
            # Where the sensor was when it ranged, in the camera at the frame's exposure.
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
            # The zone's footprint as a grid of rays, the centre one the range is along.
            a, u = np.meshgrid(across + steps * width / cols, up + steps * height / rows)
            rays = np.stack(ray(a.ravel(), u.ravel()), axis=-1)
            point = mount[:3, :3] @ (np.array(ray(across, up)) * distance) + mount[:3, 3]
            spots = rays * distance @ mount[:3, :3].T + mount[:3, 3]
            if point[2] > -0.05 or (spots[:, 2] > -0.05).any():
                continue
            x = fx * spots[:, 0] / -spots[:, 2] + cx
            y = cy - fy * spots[:, 1] / -spots[:, 2]
            if x.min() < 0 or x.max() > image_w - 1 or y.min() < 0 or y.max() > image_h - 1:
                continue  # Only whole footprints: a clipped one is a biased sample.
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
