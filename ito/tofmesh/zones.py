"""A multizone ToF frame as a range function over directions.

A RangeGrid's zones divide its field of view evenly in angle (as the VL53L5CX and the
simulated one do): zone (row, col) looks `across` right and `up` up of the sensor's -Z axis,
with row 0 at the top and column 0 at the left. Each distance is along the zone's centre ray.

A zone reports every target in its footprint at least 600 mm apart (ST's limit; nearer ones merge
into one range), nearest first: a table edge and the wall behind it. One Zones holds one target per
zone: the nearest (what carves free space: nothing in the footprint is nearer), or the next ones
behind it (zones_from_grid).

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

# Image-guided edges (`Zones.guide`): where the range steps between zones (an edge, a gap), the
# step is placed along the camera frame's colour edges instead of half-way between zone centres.
# Fine cells per zone along each axis (odd, so a zone centre is the centre of a fine cell): 7
# gives ~0.8 deg cells, 1.4 cm at 1 m, 5.6 cm at 4 m (the volume's voxels are 4 cm).
GUIDE_SCALE = 7
# Cost of crossing a colour difference (RGB, 0..1, Euclidean) in fine cells of distance: a
# wall/table edge (~0.5) costs about three zones' width, wood grain (~0.05) under half a zone.
GUIDE_COLOUR = 40.0
# Zones whose image colours (at their centres) differ by at most this are taken as one surface
# where lookup interpolates between them (no edge to place).
GUIDE_ALIKE = 0.12
# The near side of an edge seeds from the part of its footprint that looks least like what is
# behind it (`Zones._foreground`).


def _joined(inverse, axis, doubt=True):
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
    if not doubt:
        # Only what the measured zones show: a small step, a line through two zones on one
        # side passing the other, or (with lines on both sides) a crease between them.
        with np.errstate(invalid="ignore"):
            line = (np.abs(left) <= tolerance) | (np.abs(right) <= tolerance)
        return np.moveaxis(small | line | (crosses & ~unknown), -1, axis)
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
        self.inverse = inverse
        # Joined by lines through measured zones on both sides (no benefit of the doubt).
        self.sure_across = _joined(inverse, 1, doubt=False)
        self.sure_down = _joined(inverse, 0, doubt=False)
        self.slope_u, self.slope_v = self._slopes()
        self._cell_table = None  # lookup's per-cell table (_cells), made on first use
        # Set by guide(): per fine cell, the zone whose plane it takes (-1: lookup decides).
        self.label = None

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
        nearest = np.where(self.valid, self.ranges, np.nan).reshape(-1)[rn * self.cols + cn]
        if self.rows < 2 or self.cols < 2:
            out[inside] = nearest
            return out, distance, interpolated, free
        c0 = np.clip(np.floor(u).astype(int), 0, self.cols - 2)
        r0 = np.clip(np.floor(v).astype(int), 0, self.rows - 2)
        # Everything that depends only on the four zones around a point comes from a table of
        # the (rows - 1) x (cols - 1) cells between zone centres (_cells), gathered per point.
        cell = r0 * (self.cols - 1) + c0
        t = self._cells()
        # Past the outer zone centres: extrapolated where the border continues a plane, else the
        # value at the centres, which is not "interpolated" (so it never carves free space).
        left, right = t["left"][cell], t["right"][cell]
        top, bottom = t["top"][cell], t["bottom"][cell]
        fu = np.clip(u - c0, np.where(left, -0.5, 0.0), np.where(right, 1.5, 1.0))
        fv = np.clip(v - r0, np.where(top, -0.5, 0.0), np.where(bottom, 1.5, 1.0))
        within = ((u >= 0) | left) & ((u <= self.cols - 1) | right)
        within &= ((v >= 0) | top) & ((v <= self.rows - 1) | bottom)
        # Bilinear weights of the measured corners (top left, top right, bottom left, bottom
        # right); a missing zone (ST reports some as unmeasured every frame) is filled from the
        # measured ones. Sums run corner by corner, in that order.
        gu, gv = 1 - fu, 1 - fv
        w0 = gu * gv * t["ok0"][cell]
        w1 = fu * gv * t["ok1"][cell]
        w2 = gu * fv * t["ok2"][cell]
        w3 = fu * fv * t["ok3"][cell]
        weight = w0 + w1 + w2 + w3
        smooth = t["smooth"][cell] & (weight >= 0.25)
        denominator = (
            t["inverse0"][cell] * w0
            + t["inverse1"][cell] * w1
            + t["inverse2"][cell] * w2
            + t["inverse3"][cell] * w3
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            bilinear = np.where(denominator > 0, weight / denominator, np.nan)
        smooth &= np.isfinite(bilinear)
        # Nearest zone (by footprint) where the corners disagree or one is missing.
        value = np.where(smooth, bilinear, nearest)
        interpolated[inside] = smooth & within
        closest = t["closest"][cell]
        inner = (u >= 0) & (u <= self.cols - 1) & (v >= 0) & (v <= self.rows - 1)
        known = np.where(
            smooth & within, bilinear, np.where(inner & np.isfinite(closest), closest, np.nan)
        )
        if self.label is not None:
            # Where zones disagree or one is missing, the camera image chose which measured
            # zone's surface each fine cell belongs to (guide): that zone's plane, extended to
            # this point, and free space up to it.
            g = GUIDE_SCALE
            fi = np.clip(((v + 0.5) * g).astype(int), 0, self.rows * g - 1)
            fj = np.clip(((u + 0.5) * g).astype(int), 0, self.cols * g - 1)
            zone = self.label[fi, fj]
            chosen = zone >= 0
            if chosen.any():
                picked = zone[chosen]
                zr, zc = picked // self.cols, picked % self.cols
                i0 = self.inverse[zr, zc]
                i = (
                    i0
                    + self.slope_u[zr, zc] * (u[chosen] - zc)
                    + self.slope_v[zr, zc] * (v[chosen] - zr)
                )
                # A plane extended until it runs away (to infinity or behind) keeps its range.
                i = np.where((i > 0.5 * i0) & (i < 2 * i0), i, i0)
                planar = np.full(len(u), np.nan)
                planar[chosen] = 1.0 / i
                value = np.where(chosen, planar, value)
                known = np.where(chosen & inner, planar, np.where(chosen, np.nan, known))
        out[inside] = value
        free[inside] = known
        return out, distance, interpolated, free

    def _foreground(self, cr, cc, sr, sc, colour, seen, zc):
        """Seed cells of the zones (rows cr, columns cc; centre cells sr, sc): a zone nearer than
        a neighbour (the near side of an edge) reports the nearest thing in its footprint, which
        may not be what its centre sees (a table leg off-centre, the wall at the centre). Its
        seed moves to the cell of its footprint that looks least like the farther neighbours
        (their centre colours), if that cell is unlike them all by more than GUIDE_ALIKE."""
        g = GUIDE_SCALE
        sr, sc = sr.copy(), sc.copy()
        for k, (r, c) in enumerate(zip(cr.tolist(), cc.tolist(), strict=True)):
            here = self.ranges[r, c]
            behind = [
                zc[q]
                for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                if 0 <= q[0] < self.rows and 0 <= q[1] < self.cols and self.valid[q]
                and self.ranges[q] > SMALL_JUMP * here and np.isfinite(zc[q]).all()
            ]  # fmt: skip
            if not behind:
                continue
            patch = colour[r * g : (r + 1) * g, c * g : (c + 1) * g]
            unlike = (
                np.min([np.linalg.norm(patch - b, axis=-1) for b in behind], axis=0)
                * seen[r * g : (r + 1) * g, c * g : (c + 1) * g]
            )
            best = int(np.argmax(unlike))
            if unlike.flat[best] > GUIDE_ALIKE:
                sr[k], sc[k] = r * g + best // g, c * g + best % g
        return sr, sc

    def _cells(self):
        """Per cell between four zone centres (row-major over (rows - 1) x (cols - 1)), what
        lookup needs that depends on the zones alone: which corners are measured (ok0..3: top
        left, top right, bottom left, bottom right) and their inverse ranges (0 where not), the
        nearest measured corner's range, whether the corners can be one surface, and whether
        the cell continues a plane past the outer zone centres. Computed once per frame."""
        if self._cell_table is not None:
            return self._cell_table
        r = np.repeat(np.arange(self.rows - 1), self.cols - 1)
        c = np.tile(np.arange(self.cols - 1), self.rows - 1)
        corners = np.stack(
            [self.ranges[r, c], self.ranges[r, c + 1], self.ranges[r + 1, c],
             self.ranges[r + 1, c + 1]],
            axis=1,
        )  # fmt: skip
        ok = np.stack(
            [self.valid[r, c], self.valid[r, c + 1], self.valid[r + 1, c],
             self.valid[r + 1, c + 1]],
            axis=1,
        )  # fmt: skip
        inverse = np.where(ok, 1.0 / np.where(ok, corners, 1.0), np.nan)
        # Edge test over every pair of measured corners (sides and diagonals).
        near = np.where(ok, inverse, -np.inf).max(axis=1)
        far = np.where(ok, inverse, np.inf).min(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = near / far
        joined = (
            self.joined_across[r, c]
            & self.joined_across[r + 1, c]
            & self.joined_down[r, c]
            & self.joined_down[r, c + 1]
        )
        with np.errstate(invalid="ignore"):
            closest = np.where(ok, corners, np.inf).min(axis=1)
        zero = np.nan_to_num(inverse)
        self._cell_table = dict(
            left=(c == 0) & self.flat_left[r] & self.flat_left[r + 1],
            right=(c == self.cols - 2) & self.flat_right[r] & self.flat_right[r + 1],
            top=(r == 0) & self.flat_top[c] & self.flat_top[c + 1],
            bottom=(r == self.rows - 2) & self.flat_bottom[c] & self.flat_bottom[c + 1],
            smooth=(ok.sum(axis=1) >= 2) & (ratio <= self.edge) & joined,
            closest=closest,
            corners=(r, c),
            **{f"ok{k}": ok[:, k] for k in range(4)},
            **{f"inverse{k}": zero[:, k] for k in range(4)},
        )
        return self._cell_table

    def _slopes(self):
        """Per zone, how inverse range changes per zone step along columns and rows, from the
        neighbours on the same surface: of the two one-sided differences the smaller when they
        agree in sign, none when they don't (a zone standing out, or into, its neighbours: a
        table leg); one side alone only if it continues a plane (three zones collinear in
        inverse range) or is a small step."""
        inverse = self.inverse

        def along(i):
            n = np.full((i.shape[0], 1), np.nan)
            left = i - np.hstack([n, i[:, :-1]])  # i[c] - i[c-1]
            right = np.hstack([i[:, 1:], n]) - i  # i[c+1] - i[c]
            left2 = np.hstack([n, i[:, :-1]]) - np.hstack([n, n, i[:, :-2]])
            right2 = np.hstack([i[:, 2:], n, n]) - np.hstack([i[:, 1:], n])
            with np.errstate(invalid="ignore"):
                both = np.where(
                    left * right > 0, np.where(np.abs(left) < np.abs(right), left, right), 0.0
                )
                small = (SMALL_JUMP - 1) * i
                lone_l = (np.abs(left) <= small) | (np.abs(left - left2) <= CONTINUITY * i)
                lone_r = (np.abs(right) <= small) | (np.abs(right - right2) <= CONTINUITY * i)
            ok_l, ok_r = np.isfinite(left), np.isfinite(right)
            out = np.where(ok_l & ok_r, both, 0.0)
            out = np.where(ok_l & ~ok_r & lone_l, left, out)
            out = np.where(ok_r & ~ok_l & lone_r, right, out)
            return np.where(np.isfinite(i), out, 0.0)

        # Rows run downwards in zone coordinates (v), as columns do rightwards (u).
        return along(inverse), along(inverse.T).T

    def guide(self, image, project):
        """Use a camera frame taken at about the same moment to decide, wherever neighbouring
        zones disagree (an edge: a table in front of a wall) or one is missing, which zone's
        surface each part of the view belongs to. image: (h, w, 3) uint8; project: sensor-frame
        points (N, 3) -> pixel (u, v) arrays of that image (NaN behind the camera).

        On a fine grid over the field of view (GUIDE_SCALE cells per zone), a cell whose four
        surrounding zones are all measured and one surface keeps lookup's interpolation. Every
        other cell is labelled with the measured zone nearest to it *along the image*: a
        geodesic distance over the grid in which crossing a colour difference costs as much as
        travelling GUIDE_COLOUR cells per unit of colour (edge-aware fill). The cell then takes
        that zone's plane (its range, tilted by its neighbours on the same surface: `_slopes`).
        So a table top's range reaches as far as the table's colour does and the wall's range
        the rest of the way, instead of the two meeting half-way between zone centres or being
        blended into a slope (a "curtain"). Returns the number of cells labelled."""
        self.label = None
        g = GUIDE_SCALE
        rows, cols = self.rows * g, self.cols * g
        if self.rows < 2 or self.cols < 2 or not self.count:
            return 0
        fu = (np.arange(cols) + 0.5) / g - 0.5  # zone coordinates of the fine cell centres
        fv = (np.arange(rows) + 0.5) / g - 0.5
        uu, vv = np.meshgrid(fu, fv)
        across = (uu + 0.5) * self.step[0] - self.fov[0] / 2
        up = self.fov[1] / 2 - (vv + 0.5) * self.step[1]
        rays = self.direction(across.ravel(), up.ravel())
        base, _, smooth, _ = self.lookup(rays)
        base, smooth = base.reshape(rows, cols), smooth.reshape(rows, cols)
        # Trusted as interpolated: at least three of the four surrounding zones measured (lookup
        # fills one missing corner from the others), within the grazing ratio, and every pair of
        # measured neighbours joined by what the measured zones show (no benefit of the doubt).
        c0 = np.clip(np.floor(uu).astype(int), 0, self.cols - 2)
        r0 = np.clip(np.floor(vv).astype(int), 0, self.rows - 2)
        tl, tr = self.valid[r0, c0], self.valid[r0, c0 + 1]
        bl, br = self.valid[r0 + 1, c0], self.valid[r0 + 1, c0 + 1]
        corners = np.stack(
            [self.ranges[r0, c0], self.ranges[r0, c0 + 1], self.ranges[r0 + 1, c0],
             self.ranges[r0 + 1, c0 + 1]]
        )  # fmt: skip
        counted = tl.astype(int) + tr + bl + br
        with np.errstate(invalid="ignore"):
            ratio = np.fmax.reduce(corners, 0) / np.fmin.reduce(corners, 0)
        sure = (
            (self.sure_across[r0, c0] | ~(tl & tr)) & (self.sure_across[r0 + 1, c0] | ~(bl & br))
            & (self.sure_down[r0, c0] | ~(tl & bl)) & (self.sure_down[r0, c0 + 1] | ~(tr & br))
        )  # fmt: skip
        inner = (uu >= 0) & (uu <= self.cols - 1) & (vv >= 0) & (vv <= self.rows - 1)
        trusted = (counted >= 3) & inner & ((ratio <= SMALL_JUMP) | ((ratio <= self.edge) & sure))
        # The camera's colour at each cell, seen at the cell's present range (camera and sensor
        # sit centimetres apart, so a wrong range moves it little): the mean of the 2x2 pixels
        # around the cell's point (cells are ~4 px apart in a 640-wide frame).
        h, w = image.shape[:2]
        depth = np.where(np.isfinite(base), base, np.nanmedian(self.ranges[self.valid]))
        px, py = project(rays * depth.reshape(-1, 1))
        x0 = np.floor(np.nan_to_num(np.asarray(px), nan=-9.0) - 0.5)
        y0 = np.floor(np.nan_to_num(np.asarray(py), nan=-9.0) - 0.5)
        seen = (x0 >= 0) & (x0 <= w - 2) & (y0 >= 0) & (y0 <= h - 2)
        if not seen.any():
            return 0
        x0 = np.where(seen, x0, 0).astype(np.intp)
        y0 = np.where(seen, y0, 0).astype(np.intp)
        flat = image.reshape(-1, 3)
        at = y0 * w + x0
        colour = (
            flat[at].astype(np.float32) + flat[at + 1] + flat[at + w] + flat[at + w + 1]
        ).reshape(rows, cols, 3) * (1 / (4 * 255.0))
        seen = seen.reshape(rows, cols)
        # Also trusted: what lookup interpolates between zones that all look alike in the image
        # (a floor seen at a shallow angle beside a missing zone): no edge to place there.
        middle = colour[g // 2 :: g, g // 2 :: g]
        known = self.valid & seen[g // 2 :: g, g // 2 :: g]
        zc = np.where(known[..., None], middle, np.nan)
        quads = np.stack([zc[:-1, :-1], zc[:-1, 1:], zc[1:, :-1], zc[1:, 1:]])
        with np.errstate(invalid="ignore"):
            spread = np.zeros((self.rows - 1, self.cols - 1))
            for i in range(4):
                for j in range(i + 1, 4):
                    d = np.linalg.norm(quads[i] - quads[j], axis=-1)
                    spread = np.fmax(spread, d)
        alike = spread[r0, c0] <= GUIDE_ALIKE
        trusted |= smooth & inner & alike
        # Seeds: the centre cell of every measured zone, labelled with its zone.
        label = np.full((rows, cols), -1, np.int64)
        dist = np.full((rows, cols), np.inf)
        cr, cc = np.nonzero(self.valid)
        sr, sc = cr * g + g // 2, cc * g + g // 2
        sr, sc = self._foreground(cr, cc, sr, sc, colour, seen, zc)
        label[sr, sc] = cr * self.cols + cc
        dist[sr, sc] = 0.0
        # Geodesic distance over the 8-neighbour grid by Jacobi relaxation; a cell to label is at
        # most about a zone from a measured zone centre (more past missing zones).
        steps = []
        padded = np.zeros((rows + 2, cols + 2, 3))
        padded[1:-1, 1:-1] = colour
        open_ = np.zeros((rows + 2, cols + 2), bool)
        open_[1:-1, 1:-1] = seen
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    sl = (slice(1 + dy, 1 + dy + rows), slice(1 + dx, 1 + dx + cols))
                    difference = np.linalg.norm(colour - padded[sl], axis=-1)
                    cost = np.where(
                        open_[sl] & seen, math.hypot(dy, dx) + GUIDE_COLOUR * difference, np.inf
                    )
                    steps.append((sl, cost))
        big_d = np.full((rows + 2, cols + 2), np.inf)
        big_l = np.full((rows + 2, cols + 2), -1, np.int64)
        for _ in range(3 * g):
            big_d[1:-1, 1:-1] = dist
            big_l[1:-1, 1:-1] = label
            new_d, new_l = dist.copy(), label.copy()
            for sl, cost in steps:
                d = big_d[sl] + cost
                better = d < new_d
                np.copyto(new_d, d, where=better)
                np.copyto(new_l, big_l[sl], where=better)
            if np.array_equal(new_l, label) and np.allclose(new_d, dist, equal_nan=True):
                break
            dist, label = new_d, new_l
        todo = ~trusted & (label >= 0)
        if not todo.any():
            return 0
        self.label = np.where(todo, label, -1)
        return int(todo.sum())


def zones_from_grid(grid, sensor_to_camera=None):
    """Zones of a pilot-side grid dict (time, mount 4x4, shape, field_of_view, distances: per zone
    its targets, nearest first, NaN-padded): one per target, the zones' nearest targets first,
    then their second nearest, and so on."""
    rows, cols = grid["shape"]
    mount = grid["mount"] if sensor_to_camera is None else sensor_to_camera
    targets = np.asarray(grid["distances"], dtype=np.float64).reshape(rows * cols, -1)
    return [
        Zones(rows, cols, grid["field_of_view"], targets[:, k], mount)
        for k in range(targets.shape[1])
    ]
