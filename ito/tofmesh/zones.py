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
fixed percentage threshold would tear into stairs. Within that ratio, a big jump between two
neighbours must also be continuous with the zones beyond them (see `_joined`): a chair 1.5 m in
front of a wall 3 m away is an edge, a floor or a wall-floor crease is not.

Outside the outermost zone centres the range is extrapolated only where the three outermost zones
lie on one plane (collinear in inverse range), and only there may it carve free space; elsewhere
the border half-zone takes the value at the outer centres and counts as not interpolated (it
shapes the surface, never carves).
"""

import math

import numpy as np

GRAZING = math.radians(10)
# Neighbours closer than this ratio are one surface without further checks.
SMALL_JUMP = 1.15
# A bigger jump is one surface if lines through the zones on either side meet within it, or miss
# by at most this fraction of the inverse range (a plane is close to linear in inverse range).
CONTINUITY = 0.1


def _joined(inverse, axis):
    """Whether each pair of neighbouring zones along `axis` (0: rows, 1: columns) can be one
    surface, from the inverse ranges (NaN: unmeasured). Shape: one less along `axis`."""
    i = np.moveaxis(inverse, axis, -1)
    a, b = i[..., :-1], i[..., 1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        small = np.maximum(a, b) <= SMALL_JUMP * np.minimum(a, b)
    nan = np.full(i.shape[:-1] + (1,), np.nan)
    before = np.concatenate([nan, i[..., :-2]], -1)  # the zone before a
    after = np.concatenate([i[..., 2:], nan], -1)  # the zone after b
    # How far the line through (before, a) passes b, and the line through (after, b) passes a.
    # The two lines meet between a and b (a crease, or a plane) when these have the same sign;
    # at a step (near zones on one side, far on the other) they have opposite signs.
    left = a + (a - before) - b
    right = b + (b - after) - a
    tolerance = CONTINUITY * np.maximum(a, b)
    with np.errstate(invalid="ignore"):
        crosses = (left * right >= 0) | (np.abs(left) <= tolerance) | (np.abs(right) <= tolerance)
    # Lines on both sides are needed to tell a crease from a step; with one or none, the ratio
    # test alone decides. Pairs with an unmeasured zone are left to the missing-zone fill.
    unknown = ~np.isfinite(left) | ~np.isfinite(right)
    joined = small | crosses | unknown | ~np.isfinite(a) | ~np.isfinite(b)
    return np.moveaxis(joined, -1, axis)


def _flat(i0, i1, i2):
    """Whether three neighbouring zones (inverse ranges, outermost first) lie on one plane."""
    with np.errstate(invalid="ignore"):
        return np.abs(i0 - 2 * i1 + i2) <= CONTINUITY * i1


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
        inverse = np.where(self.valid, 1.0 / np.where(self.valid, self.ranges, 1.0), np.nan)
        self.joined_across = _joined(inverse, 1)  # (rows, cols - 1)
        self.joined_down = _joined(inverse, 0)  # (rows - 1, cols)
        # Border rows/columns that continue a plane outwards (extrapolated half a zone).
        no = np.zeros(max(rows, cols), bool)
        self.flat_left = _flat(*inverse[:, :3].T) if cols >= 3 else no[:rows]
        self.flat_right = _flat(*inverse[:, ::-1][:, :3].T) if cols >= 3 else no[:rows]
        self.flat_top = _flat(*inverse[:3]) if rows >= 3 else no[:cols]
        self.flat_bottom = _flat(*inverse[::-1][:3]) if rows >= 3 else no[:cols]

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
        """Range along each point's ray, the point's own distance, whether the range was
        interpolated between zones of one surface (False: nearest zone, at an edge or a gap),
        and how far the ray is known to be free, for points in the sensor frame (N, 3). Range
        and free are NaN outside the field of view or where no zone measured.

        Free is the range where interpolated; at an edge between the outer zone centres, the
        nearest of the surrounding measured zones (each reports the nearest target in its
        footprint, so nothing is nearer than all of them); past the outer centres, NaN."""
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        distance = np.sqrt(x * x + y * y + z * z)
        across = np.arctan2(x, -z)
        up = np.arcsin(np.clip(y / np.maximum(distance, 1e-9), -1, 1))
        # Continuous zone coordinates: centre of zone (r, c) at (r, c).
        u = (across + self.fov[0] / 2) / self.step[0] - 0.5
        v = (self.fov[1] / 2 - up) / self.step[1] - 0.5
        inside = (
            (z < 0) & (u >= -0.5) & (u <= self.cols - 0.5) & (v >= -0.5) & (v <= self.rows - 0.5)
        )
        out = np.full(len(points), np.nan)
        free = np.full(len(points), np.nan)
        interpolated = np.zeros(len(points), bool)
        if not inside.any():
            return out, distance, interpolated, free
        u, v = u[inside], v[inside]
        rn = np.clip(np.rint(v).astype(int), 0, self.rows - 1)
        cn = np.clip(np.rint(u).astype(int), 0, self.cols - 1)
        nearest = np.where(self.valid[rn, cn], self.ranges[rn, cn], np.nan)
        if self.rows < 2 or self.cols < 2:
            out[inside] = nearest
            return out, distance, interpolated, free
        c0 = np.clip(np.floor(u).astype(int), 0, self.cols - 2)
        r0 = np.clip(np.floor(v).astype(int), 0, self.rows - 2)
        # Past the outer zone centres: extrapolated where the border continues a plane, else the
        # value at the centres, which is not "interpolated" (so it never carves free space).
        left = (c0 == 0) & self.flat_left[r0] & self.flat_left[r0 + 1]
        right = (c0 == self.cols - 2) & self.flat_right[r0] & self.flat_right[r0 + 1]
        top = (r0 == 0) & self.flat_top[c0] & self.flat_top[c0 + 1]
        bottom = (r0 == self.rows - 2) & self.flat_bottom[c0] & self.flat_bottom[c0 + 1]
        fu = np.clip(u - c0, np.where(left, -0.5, 0.0), np.where(right, 1.5, 1.0))
        fv = np.clip(v - r0, np.where(top, -0.5, 0.0), np.where(bottom, 1.5, 1.0))
        within = ((u >= 0) | left) & ((u <= self.cols - 1) | right)
        within &= ((v >= 0) | top) & ((v <= self.rows - 1) | bottom)
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
        bilinear_w = (
            np.stack([(1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv], axis=1) * ok
        )
        weight = bilinear_w.sum(axis=1)
        joined = (
            self.joined_across[r0, c0]
            & self.joined_across[r0 + 1, c0]
            & self.joined_down[r0, c0]
            & self.joined_down[r0, c0 + 1]
        )
        smooth = (ok.sum(axis=1) >= 2) & (ratio <= self.edge) & (weight >= 0.25) & joined
        denominator = (np.nan_to_num(inverse) * bilinear_w).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            bilinear = np.where(denominator > 0, weight / denominator, np.nan)
        smooth &= np.isfinite(bilinear)
        # Nearest zone (by footprint) where the corners disagree or one is missing.
        out[inside] = np.where(smooth, bilinear, nearest)
        interpolated[inside] = smooth & within
        with np.errstate(invalid="ignore"):
            closest = np.where(ok, corners, np.inf).min(axis=1)
        inner = (u >= 0) & (u <= self.cols - 1) & (v >= 0) & (v <= self.rows - 1)
        free[inside] = np.where(
            smooth & within, bilinear, np.where(inner & np.isfinite(closest), closest, np.nan)
        )
        return out, distance, interpolated, free


def zones_from_grid(grid, sensor_to_camera=None):
    """Zones of a pilot-side grid dict (time, mount 4x4, shape, field_of_view, distances)."""
    rows, cols = grid["shape"]
    mount = grid["mount"] if sensor_to_camera is None else sensor_to_camera
    return Zones(rows, cols, grid["field_of_view"], grid["distances"], mount)


def footprint(zones, distance):
    """Width (m) of one zone at `distance`."""
    return 2 * distance * math.tan(zones.step[0] / 2)
