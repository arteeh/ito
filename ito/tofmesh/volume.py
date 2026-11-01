"""A sparse, bounded signed-distance volume fused from ToF frames, coloured from camera frames.

Why a volume and not stitched patches: every ToF frame re-measures surfaces already seen (the
head scans back and forth), so patches would pile up without end and disagree by the pose
error. A truncated signed distance field (TSDF) averages every measurement of a surface into
one value per voxel: revisits refine instead of adding, noise averages out, and free space seen
through a stale surface erases it. Its size grows with the area explored, never with time.

Voxels live in blocks of BLOCK³, allocated only where a ToF frame saw a surface, in a pool of
fixed capacity: RAM is allocated once, at start. A full pool reuses the block touched longest
ago that is farthest from the robot (that part of the map is forgotten).

Units: metres, world axes as the robot's odometry (+Y up). Signed distance is positive in front
of a surface (towards the sensor), negative behind it.
"""

import numpy as np

BLOCK = 8
# Weight of a voxel saturates here, so a changed scene (a moved chair) relearns in seconds.
MAX_WEIGHT = 40.0
MAX_COLOUR_WEIGHT = 12.0
# Free space seen through a voxel (well in front of the measured surface, along a ray whose range
# is interpolated) also forgets this fraction of its weight per frame at full range trust: a
# saturated surface that is gone (a moved object, a wall doubled by drift) fades in a few seconds,
# while one stray ray only dents it.
CARVE = 0.2
MESH_WEIGHT = 0.1  # a voxel takes part in the mesh from this weight (surface.extract)
# Block keys as one integer each (_code): each axis offset by KEY_OFFSET into 21 bits, so codes
# sort as the keys do (x, then y, then z). 2^20 blocks of 32 cm is 300 km each way.
KEY_OFFSET = 2**20


def _code(keys):
    """(N, 3) int64 block keys -> (N,) int64, ordered like the keys (lexicographically)."""
    k = keys + KEY_OFFSET
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


def _unique_keys(keys):
    """np.unique(keys, axis=0), the same keys in the same order, without its slow row sort."""
    _, first = np.unique(_code(keys), return_index=True)
    return keys[first]


# Offsets (dy, dx) of the ring of cells at Chebyshev distance k, for k up to 6 (paint).
_RINGS = [
    np.array([(dy, dx) for dy in range(-k, k + 1) for dx in range(-k, k + 1)
              if max(abs(dx), abs(dy)) == k], np.int64).reshape(-1, 2)
    for k in range(7)
]  # fmt: skip
# The offsets of a block's 27-neighbourhood (itself included), in (dx, dy, dz) order.
_AROUND = np.stack(np.meshgrid(*[np.arange(-1, 2)] * 3, indexing="ij"), -1).reshape(-1, 3)


class Volume:
    def __init__(self, voxel=0.04, truncation=None, capacity=6000, max_range=4.0):
        self.voxel = float(voxel)
        self.truncation = float(truncation if truncation is not None else 3.5 * voxel)
        self.capacity = int(capacity)
        self.max_range = float(max_range)
        self.block_size = self.voxel * BLOCK
        n = self.capacity + 1  # the last row stays empty: a missing neighbour reads it
        self.sdf = np.zeros((n, BLOCK, BLOCK, BLOCK), np.float32)
        self.weight = np.zeros((n, BLOCK, BLOCK, BLOCK), np.float32)
        self.colour = np.zeros((n, BLOCK, BLOCK, BLOCK, 3), np.float32)
        self.colour_weight = np.zeros((n, BLOCK, BLOCK, BLOCK), np.float32)
        self.keys = np.zeros((n, 3), np.int64)
        self.touched = np.zeros(n, np.float64)
        self.slots = {}
        self.free = list(range(self.capacity - 1, -1, -1))
        self.dirty = set()  # geometry changed
        self.painted = set()  # only colours changed
        self.evicted = set()
        self.fresh = set()  # slots given to a new block (the renderer clears their texture)
        self.evictions = 0
        self._lookup = None  # (sorted codes, their slots) of the allocated blocks, see _slots_of
        local = np.stack(np.meshgrid(*[np.arange(BLOCK)] * 3, indexing="ij"), -1).reshape(-1, 3)
        self.local = (local + 0.5) * self.voxel  # voxel centres inside a block (512, 3)

    @property
    def used(self):
        return len(self.slots)

    @property
    def nbytes(self):
        return sum(a.nbytes for a in (self.sdf, self.weight, self.colour, self.colour_weight))

    def _key(self, points):
        return np.floor(points / self.block_size).astype(np.int64)

    def allocate(self, points, now, position):
        """Slots of the blocks containing these world points, allocating missing ones."""
        if not len(points):
            return np.zeros(0, np.int64)
        keys = _unique_keys(self._key(points))
        out = []
        missing = []
        for key in map(tuple, keys.tolist()):
            slot = self.slots.get(key)
            if slot is None:
                missing.append(key)
            else:
                out.append(slot)
        taken = self._take(len(missing), now, position, set(out))
        if taken:
            self._lookup = None
        for key, slot in zip(missing, taken, strict=False):
            self.slots[key] = slot
            self.keys[slot] = key
            self.sdf[slot] = 0
            self.weight[slot] = 0
            self.colour[slot] = 0
            self.colour_weight[slot] = 0
            self.fresh.add(slot)
            out.append(slot)
        slots = np.array(out, np.int64)
        self.touched[slots] = now
        return slots

    def _take(self, count, now, position, protected):
        """Up to `count` free slots; a full pool forgets its stalest blocks, staleness growing
        with distance from the robot (scored once per call)."""
        taken = [self.free.pop() for _ in range(min(count, len(self.free)))]
        short = count - len(taken)
        if short <= 0:
            return taken
        candidates = np.array([s for s in self.slots.values() if s not in protected], np.int64)
        if not len(candidates):
            return taken
        centres = (self.keys[candidates] + 0.5) * self.block_size
        far = np.linalg.norm(centres - position, axis=1)
        score = (now - self.touched[candidates]) * (1 + far)
        short = min(short, len(candidates))
        stalest = candidates[np.argpartition(-score, short - 1)[:short]]
        self._lookup = None
        for slot in stalest.tolist():
            key = tuple(self.keys[slot].tolist())
            del self.slots[key]
            self.dirty.discard(slot)
            self.painted.discard(slot)
            self.evicted.add(key)
            self.evictions += 1
            taken.append(slot)
        return taken

    def existing(self, points):
        """Slots of the already allocated blocks containing these world points (no allocation,
        no `touched` refresh)."""
        if not len(points):
            return np.zeros(0, np.int64)
        slots = self._slots_of(_unique_keys(self._key(points)))
        return slots[slots != self.capacity]

    def _slots_of(self, keys):
        """Slot of each block key (N, 3), or the empty row (capacity) where none is allocated."""
        if self._lookup is None:
            slots = np.fromiter(self.slots.values(), np.int64, len(self.slots))
            codes = _code(self.keys[slots])
            order = np.argsort(codes)
            self._lookup = codes[order], slots[order]
        codes, slots = self._lookup
        wanted = _code(keys)
        at = np.minimum(np.searchsorted(codes, wanted), max(len(codes) - 1, 0))
        if not len(codes):
            return np.full(len(wanted), self.capacity, np.int64)
        return np.where(codes[at] == wanted, slots[at], self.capacity)

    def centres(self, slots):
        """World centres of every voxel of these blocks, (len(slots), 512, 3)."""
        origin = self.keys[slots] * self.block_size
        return origin[:, None, :] + self.local[None, :, :]

    def integrate(self, zones, world_from_sensor, now, position):
        """Fuse one ToF frame (zones.Zones) seen from world_from_sensor (4x4)."""
        if not zones.count:
            return 0
        samples = zones.samples(3)
        if not len(samples):
            return 0
        rays = samples / np.linalg.norm(samples, axis=1, keepdims=True)
        # Blocks around the surface as interpolated along each sample's ray (a zone's one range
        # misses a floor seen at a shallow angle by tens of centimetres across its footprint).
        measured = zones.lookup(samples)[0]
        r = np.where(np.isfinite(measured), measured, np.linalg.norm(samples, axis=1))
        keep = r <= self.max_range
        rays, r = rays[keep], r[keep]
        if not len(r):
            return 0
        band = np.concatenate(
            [rays * (r + d)[:, None] for d in (-self.truncation, 0.0, self.truncation)]
        )
        rotate, origin = world_from_sensor[:3, :3].T, world_from_sensor[:3, 3]
        slots = self.allocate(band @ rotate + origin, now, position)
        if not len(slots):
            return 0
        # Blocks already mapped in front of the surface, along the same rays: free space seen
        # through them erases what is no longer there (a moved object, a wall doubled by drift).
        step = self.block_size / 2
        reach = r - self.truncation
        t = np.arange(step, max(float(reach.max()), step), step)
        along = (rays[:, None, :] * t[None, :, None])[t[None, :] < reach[:, None]]
        front = self.existing(along @ rotate + origin)
        front = front[~np.isin(front, slots)]
        if len(front):
            # Only blocks holding a surface can change; empty space needs no more carving.
            surface = ((self.sdf[front] < 0) & (self.weight[front] > 0)).reshape(len(front), -1)
            front = front[surface.any(axis=1)]
        banded = len(slots)
        slots = np.concatenate([slots, front]) if len(front) else slots
        sensor_from_world = np.linalg.inv(world_from_sensor)
        points = self.centres(slots).reshape(-1, 3)
        local = points @ sensor_from_world[:3, :3].T + sensor_from_world[:3, 3]
        measured, distance, interpolated, free = zones.lookup(local)
        sdf = measured - distance
        ok = np.isfinite(sdf) & (sdf > -self.truncation) & (distance > 0.03)
        ok &= measured <= self.max_range
        # Free space is carved only as far as the ray is known to be free: a zone's one range
        # stands for its whole footprint, and on a floor seen at a shallow angle that footprint
        # spans tens of centimetres of range, so carving with it would punch holes in the floor
        # in front of the measured point (zones.lookup: `free`).
        with np.errstate(invalid="ignore"):
            ok &= (sdf <= self.truncation) | (distance < free - self.truncation)
        if not ok.any():
            return 0
        # Farther ranges are noisier (sim and datasheet: ~3 mm near, ~2 cm at 4 m), and the
        # zone's footprint grows with distance: trust them less.
        trust = np.clip(1.3 - measured / 4.0, 0.3, 1.0)
        w = trust
        # Behind the surface, confidence falls off: the far side is unseen. Free space far in
        # front only erodes slowly, so one stray ray does not erase a surface.
        # (Not to zero: a floor seen at a shallow angle has its underside voxels far "behind"
        # along the ray though barely below it, and they must count for the surface to close.)
        w = np.where(sdf < 0, w * (1 + 0.7 * sdf / self.truncation), w)
        w = np.where(sdf > self.truncation, w * 0.3, w)
        keep = np.where(ok & (sdf > self.truncation), 1 - CARVE * trust, 1).astype(np.float32)
        w = np.where(ok, w, 0).astype(np.float32)
        s = np.clip(np.nan_to_num(sdf), -self.truncation, self.truncation).astype(np.float32)
        shape = (len(slots), BLOCK, BLOCK, BLOCK)
        w, s = w.reshape(shape), s.reshape(shape)
        old_w = self.weight[slots] * keep.reshape(shape)
        total = old_w + w
        with np.errstate(invalid="ignore", divide="ignore"):
            fused = np.where(total > 0, (self.sdf[slots] * old_w + s * w) / total, 0)
        weight = np.minimum(total, MAX_WEIGHT)
        touched = w.reshape(len(slots), -1).any(axis=1)
        # Blocks only seen through re-mesh when their surface can have changed: a voxel changed
        # side or crossed the weight a mesh needs (surface.extract's min_weight).
        old_sdf, old_weight = self.sdf[slots[banded:]], self.weight[slots[banded:]]
        flipped = ((old_sdf < 0) != (fused[banded:] < 0)) | (
            (old_weight >= MESH_WEIGHT) != (weight[banded:] >= MESH_WEIGHT)
        )
        if len(slots) > banded:
            touched[banded:] &= flipped.reshape(len(slots) - banded, -1).any(axis=1)
        self.sdf[slots] = fused
        self.weight[slots] = weight
        self.dirty.update(slots[touched].tolist())
        return int(ok.sum())

    def visible(self, camera_from_world, intrinsics, size, distance):
        """Slots whose block may be seen by this camera within `distance`."""
        if not self.slots:
            return np.zeros(0, np.int64)
        slots = np.fromiter(self.slots.values(), np.int64, len(self.slots))
        centres = (self.keys[slots] + 0.5) * self.block_size
        local = centres @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
        z = -local[:, 2]
        radius = self.block_size * 0.87
        fx, fy, cx, cy = intrinsics
        w, h = size
        with np.errstate(divide="ignore", invalid="ignore"):
            u = cx + fx * local[:, 0] / z
            v = cy - fy * local[:, 1] / z
            margin_u = fx * radius / np.maximum(z, 0.05)
            margin_v = fy * radius / np.maximum(z, 0.05)
        ok = (z > -radius) & (z < distance + radius)
        ok &= (u > -margin_u) & (u < w + margin_u) & (v > -margin_v) & (v < h + margin_v)
        ok |= np.linalg.norm(local, axis=1) < radius * 1.5
        return slots[ok]

    def paint(self, image, world_from_camera, intrinsics, now, distance=4.5):
        """Colour the surface voxels this rectified pinhole camera frame sees, with occlusion
        from the volume's own surface. image: (h, w, 3) uint8."""
        h, w = image.shape[:2]
        camera_from_world = np.linalg.inv(world_from_camera)
        slots = self.visible(camera_from_world, intrinsics, (w, h), distance)
        if not len(slots):
            return 0
        grid = self.sdf[slots]
        sdf = grid.reshape(len(slots), -1)
        near = (self.weight[slots].reshape(len(slots), -1) > 0) & (np.abs(sdf) < self.voxel)
        block, voxel = np.nonzero(near)
        if not len(block):
            return 0
        points = self.keys[slots][block] * self.block_size + self.local[voxel]
        local = points @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
        z = -local[:, 2]
        fx, fy, cx, cy = intrinsics
        with np.errstate(divide="ignore", invalid="ignore"):
            u = cx + fx * local[:, 0] / z
            v = cy - fy * local[:, 1] / z
        ok = (z > 0.05) & (z < distance) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
        block, voxel, u, v, z, local = block[ok], voxel[ok], u[ok], v[ok], z[ok], local[ok]
        if not len(z):
            return 0
        # Occlusion: a depth buffer of the surface voxels themselves, each drawn as a square of
        # its own size on screen (so a surface leaves no gaps between its voxels at any range),
        # in 4-pixel cells. Each voxel writes the farthest depth its own surface reaches across
        # that square, which grows with the grazing angle (the signed distance's slope against
        # the ray); a voxel behind that depth is hidden. So a box hides the floor behind it, but
        # nearer floor does not hide farther floor: the low camera sees the floor nearly edge-on,
        # where a fixed margin over a 3x3 minimum of 8-pixel cells hid every floor voxel beyond
        # ~1 m, so it was never painted (grey in the mesh).
        scale = 4
        gw, gh = w // scale + 1, h // scale + 1
        gu = (u / scale).astype(np.int64) + 6  # 6 cells of margin on every side
        gv = (v / scale).astype(np.int64) + 6
        f = min(fx, fy)
        reach = np.clip(np.ceil(0.75 * f * self.voxel / z / scale), 1, 6).astype(np.int64)
        normal = np.stack(np.gradient(grid, axis=(1, 2, 3)), -1).reshape(len(slots), -1, 3)
        normal = normal[block, voxel] @ camera_from_world[:3, :3].T
        ray = local / np.linalg.norm(local, axis=1, keepdims=True)
        cos = np.abs((normal * ray).sum(1)) / np.maximum(np.linalg.norm(normal, axis=1), 1e-9)
        cos = np.clip(cos, 0.02, 1.0)  # roundoff can put a parallel pair past 1 (sqrt of < 0)
        slope = np.sqrt(1 - cos * cos) / cos  # depth change per lateral distance
        across = (reach + 0.71) * scale * z / f  # from this voxel to the far side of its square
        # A plane's inverse depth is linear across the image: the depth it reaches there.
        with np.errstate(divide="ignore"):
            far = z / np.maximum(1 - slope * across / z, 0)
        # Every voxel's square at once: the cells of each ring k around it (for the voxels whose
        # reach is at least k), the nearest depth per cell kept (a minimum: order-free).
        width = gw + 12
        depth = np.full((gh + 12) * width, np.inf)
        centre = gv * width + gu
        cells, values = [], []
        for k in range(int(reach.max()) + 1):
            at = reach >= k
            ring = _RINGS[k][:, 0] * width + _RINGS[k][:, 1]
            cells.append((centre[at][:, None] + ring[None, :]).reshape(-1))
            values.append(np.repeat(far[at], len(ring)))
        np.minimum.at(depth, np.concatenate(cells), np.concatenate(values))
        seen = z <= depth[centre] + 1.5 * self.voxel + 0.02 * z
        block, voxel, u, v, z = block[seen], voxel[seen], u[seen], v[seen], z[seen]
        x0 = np.floor(u).astype(np.int64)
        y0 = np.floor(v).astype(np.int64)
        x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
        fx_, fy_ = (u - x0)[:, None], (v - y0)[:, None]
        img = image.astype(np.float32)
        rgb = (
            img[y0, x0] * (1 - fx_) * (1 - fy_)
            + img[y0, x1] * fx_ * (1 - fy_)
            + img[y1, x0] * (1 - fx_) * fy_
            + img[y1, x1] * fx_ * fy_
        )
        # Nearer and more central views see more detail.
        radial = np.hypot((u - cx) / fx, (v - cy) / fy)
        wc = (1.0 / (1.0 + 0.4 * z * z)) * np.clip(1.3 - 0.5 * radial, 0.2, 1.0)
        target = slots[block]
        flat_colour = self.colour.reshape(-1, 3)
        flat_weight = self.colour_weight.reshape(-1)
        index = target * BLOCK**3 + voxel
        old = flat_weight[index]
        total = old + wc
        flat_colour[index] = (flat_colour[index] * old[:, None] + rgb * wc[:, None]) / total[
            :, None
        ]
        flat_weight[index] = np.minimum(total, MAX_COLOUR_WEIGHT)
        self.painted.update(np.unique(target).tolist())
        return int(len(index))

    def take_dirty(self, colours=False):
        """Blocks whose geometry changed (and, with colours, those only repainted) since the
        last call, and the keys of blocks forgotten meanwhile."""
        dirty, evicted = self.dirty, self.evicted
        self.dirty, self.evicted = set(), set()
        painted = set()
        if colours:
            painted, self.painted = self.painted - dirty, set()
        return np.array(sorted(dirty), np.int64), np.array(sorted(painted), np.int64), evicted

    def neighbourhood(self, slots):
        """(len(slots), 3, 3, 3) slot of each block and its 26 neighbours (missing: the empty
        row)."""
        slots = np.asarray(slots, np.int64)
        keys = self.keys[slots][:, None, :] + _AROUND[None, :, :]  # (M, 27, 3)
        return self._slots_of(keys.reshape(-1, 3)).reshape(len(slots), 3, 3, 3)
