"""A multizone ToF frame as a range function over directions.

A RangeGrid's zones divide its field of view evenly in angle (as the VL53L5CX and the
simulated one do): zone (row, col) looks `across` right and `up` up of the sensor's -Z axis,
with row 0 at the top and column 0 at the left. Each distance is along the zone's centre ray.

Between zone centres the inverse range is interpolated bilinearly (exact for a plane seen by a
pinhole, near enough for these small angles), but only across zones that can be one surface:
neighbouring ranges of a plane seen at a grazing angle `GRAZING` or steeper differ by at most
sin(GRAZING) / sin(GRAZING - step). A larger jump is an edge (an object in front of a wall, a
doorway): there each point takes its nearest zone, so the mesh steps rather than hanging a
"curtain" between near and far. Floors seen at a shallow angle stay continuous this way, which a
fixed percentage threshold would tear into stairs.
"""

import math

import numpy as np

GRAZING = math.radians(16)


class Zones:
    def __init__(self, rows, cols, field_of_view, distances, sensor_to_camera):
        self.rows, self.cols = rows, cols
        self.fov = (float(field_of_view[0]), float(field_of_view[1]))
        self.step = (self.fov[0] / cols, self.fov[1] / rows)
        self.ranges = np.asarray(distances, dtype=np.float64).reshape(rows, cols)
        self.valid = np.isfinite(self.ranges) & (self.ranges > 0)
        self.mount = np.asarray(sensor_to_camera, dtype=np.float64)
        step = max(self.step)
        self.edge = math.sin(GRAZING) / math.sin(max(GRAZING - step, 0.02))

    @property
    def count(self):
        return int(self.valid.sum())

    @staticmethod
    def direction(across, up):
        """Unit rays (sensor axes: +X right, +Y up, looking down -Z)."""
        c = np.cos(up)
        return np.stack([c * np.sin(across), np.sin(up), -c * np.cos(across)], axis=-1)

    def centres(self):
        """(across, up) angles of every zone centre, shape (rows, cols)."""
        across = (np.arange(self.cols) + 0.5) * self.step[0] - self.fov[0] / 2
        up = self.fov[1] / 2 - (np.arange(self.rows) + 0.5) * self.step[1]
        return np.meshgrid(across, up)

    def samples(self, per_zone=3):
        """Points on the measured surface, `per_zone`² per valid zone spread over its footprint,
        in the sensor frame (N, 3). Used to find which volume blocks a frame touches."""
        offsets = ((np.arange(per_zone) + 0.5) / per_zone - 0.5) * 1.0
        rows, cols = np.nonzero(self.valid)
        if not len(rows):
            return np.zeros((0, 3))
        across = ((cols + 0.5)[:, None] + offsets[None, :]) * self.step[0] - self.fov[0] / 2
        up = self.fov[1] / 2 - ((rows + 0.5)[:, None] + offsets[None, :]) * self.step[1]
        a = np.repeat(across, per_zone, axis=1)
        u = np.tile(up, (1, per_zone))
        r = self.ranges[rows, cols][:, None]
        return (self.direction(a, u) * r[..., None]).reshape(-1, 3)

    def lookup(self, points):
        """Range along each point's ray and the point's own distance, for points in the sensor
        frame (N, 3). Range is NaN outside the field of view or where no zone measured."""
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        distance = np.sqrt(x * x + y * y + z * z)
        across = np.arctan2(x, -z)
        up = np.arcsin(np.clip(y / np.maximum(distance, 1e-9), -1, 1))
        # Continuous zone coordinates: centre of zone (r, c) at (r, c).
        u = (across + self.fov[0] / 2) / self.step[0] - 0.5
        v = (self.fov[1] / 2 - up) / self.step[1] - 0.5
        inside = (
            (z < 0)
            & (u >= -0.5)
            & (u <= self.cols - 0.5)
            & (v >= -0.5)
            & (v <= self.rows - 0.5)
        )
        out = np.full(len(points), np.nan)
        if not inside.any():
            return out, distance
        u, v = u[inside], v[inside]
        c0 = np.clip(np.floor(u).astype(int), 0, self.cols - 2)
        r0 = np.clip(np.floor(v).astype(int), 0, self.rows - 2)
        fu = np.clip(u - c0, -0.5, 1.5)
        fv = np.clip(v - r0, -0.5, 1.5)
        corners = np.stack(
            [
                self.ranges[r0, c0],
                self.ranges[r0, c0 + 1],
                self.ranges[r0 + 1, c0],
                self.ranges[r0 + 1, c0 + 1],
            ],
            axis=1,
        )
        ok = np.stack(
            [
                self.valid[r0, c0],
                self.valid[r0, c0 + 1],
                self.valid[r0 + 1, c0],
                self.valid[r0 + 1, c0 + 1],
            ],
            axis=1,
        )
        inverse = np.where(ok, 1.0 / np.where(ok, corners, 1.0), np.nan)
        # Edge test over every pair of measured corners (sides and diagonals); a missing zone
        # (ST reports some as unmeasured every frame) is filled from the measured ones.
        near = np.where(ok, inverse, -np.inf).max(axis=1)
        far = np.where(ok, inverse, np.inf).min(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = near / far
        bilinear_w = np.stack(
            [(1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv], axis=1
        ) * ok
        weight = bilinear_w.sum(axis=1)
        smooth = (ok.sum(axis=1) >= 2) & (ratio <= self.edge) & (weight >= 0.25)
        with np.errstate(divide="ignore", invalid="ignore"):
            bilinear = weight / (np.nan_to_num(inverse) * bilinear_w).sum(axis=1)
        # Nearest zone (by footprint) where the corners disagree or one is missing.
        rn = np.clip(np.rint(v).astype(int), 0, self.rows - 1)
        cn = np.clip(np.rint(u).astype(int), 0, self.cols - 1)
        nearest = np.where(self.valid[rn, cn], self.ranges[rn, cn], np.nan)
        out[inside] = np.where(smooth, bilinear, nearest)
        return out, distance


def zones_from_grid(grid, sensor_to_camera=None):
    """Zones of a pilot-side grid dict (time, mount 4x4, shape, field_of_view, distances)."""
    rows, cols = grid["shape"]
    mount = grid["mount"] if sensor_to_camera is None else sensor_to_camera
    return Zones(rows, cols, grid["field_of_view"], grid["distances"], mount)


def footprint(zones, distance):
    """Width (m) of one zone at `distance`."""
    return 2 * distance * math.tan(zones.step[0] / 2)
