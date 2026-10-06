"""Vectorized RGB-D fusion into stable voxel slots; all distances are metres."""

import numpy as np


class RGBDBackend:
    def __init__(
        self,
        capacity,
        intrinsics,
        *,
        voxel_size=0.04,
        window_seconds=4.0,
        fade_seconds=0.5,
        device="auto",
    ):
        self.xp = np
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("Reconstruction device must be auto, cpu or cuda")
        if device != "cpu":
            try:
                import cupy

                if cupy.cuda.runtime.getDeviceCount():
                    # Compile and run one kernel now: a missing toolkit must surface here,
                    # where "auto" can still fall back, not mid-session.
                    float((cupy.arange(4, dtype=cupy.float32) * 2).sum())
                    self.xp = cupy
            except Exception as exc:
                if device == "cuda":
                    raise RuntimeError(
                        "CUDA reconstruction needs CuPy and an NVIDIA CUDA device"
                    ) from exc
            if device == "cuda" and self.xp is np:
                raise RuntimeError("CUDA reconstruction needs an NVIDIA CUDA device")
        self.intrinsics = intrinsics
        self.voxel_size, self.window, self.fade = voxel_size, window_seconds, fade_seconds
        self.records = np.zeros((capacity, 4, 4), np.float32)
        self.keys = np.full(capacity, -1, np.int64)
        self.seen = np.full(capacity, -np.inf)
        self.retiring = np.zeros(capacity, bool)
        self.dirty = np.zeros(capacity, bool)
        self.retire_revision = np.full(capacity, np.inf)
        self.release_after = np.full(capacity, np.inf)
        self.budget = capacity
        self.refreshed = 0
        xp = self.xp
        y, x = xp.mgrid[: intrinsics.height, : intrinsics.width]
        self.rays = xp.stack(
            (
                (x - intrinsics.cx) / intrinsics.fx,
                -(y - intrinsics.cy) / intrinsics.fy,
                -xp.ones_like(x),
            ),
            axis=-1,
        )

    @property
    def count(self):
        return int(np.count_nonzero(self.keys >= 0))

    def resize(self, budget, now):
        if budget > len(self.keys):
            extra = budget - len(self.keys)
            self.records = np.pad(self.records, ((0, extra), (0, 0), (0, 0)))
            self.keys = np.pad(self.keys, (0, extra), constant_values=-1)
            self.seen = np.pad(self.seen, (0, extra), constant_values=-np.inf)
            self.retiring = np.pad(self.retiring, (0, extra))
            self.dirty = np.pad(self.dirty, (0, extra))
            self.retire_revision = np.pad(self.retire_revision, (0, extra), constant_values=np.inf)
            self.release_after = np.pad(self.release_after, (0, extra), constant_values=np.inf)
        self.budget = budget
        excess = max(0, self.count - budget - int(self.retiring.sum()))
        candidates = np.flatnonzero((self.keys >= 0) & ~self.retiring)
        if excess:
            self.retire(candidates[np.argsort(self.seen[candidates])[:excess]])

    def retire(self, indices):
        # Preserve the natural deadline so delayed eviction packets cannot revive stale geometry.
        self.records[indices, 3, 3] = 1
        self.retiring[indices] = True
        self.dirty[indices] = True

    def expire(self, now, acknowledged):
        accepted = self.retiring & (self.retire_revision <= acknowledged)
        self.release_after[accepted] = np.minimum(
            self.release_after[accepted], now + self.fade + 0.05
        )
        expired = (self.keys >= 0) & np.where(
            self.retiring, self.release_after <= now, self.records[:, 1, 3] <= now
        )
        self.keys[expired] = -1
        self.records[expired] = 0
        self.retiring[expired] = False
        self.retire_revision[expired] = np.inf
        self.release_after[expired] = np.inf
        self.dirty[expired] = True

    def integrate(self, rgb, depth, camera, now):
        xp = self.xp
        z = xp.asarray(depth)
        valid = xp.isfinite(z) & (z > 0) & (z < 1000)
        points = self.rays[valid] * z[valid, None]
        transform = xp.asarray(camera)
        points = points @ transform[:3, :3].T + transform[:3, 3]
        self.integrate_points(points, xp.asarray(rgb)[valid], now)

    def integrate_points(self, points, colors, now):
        """Fuse world-space dense points through the same budget, fade and eviction policy."""
        xp = self.xp
        valid = xp.all(xp.isfinite(points), axis=1)
        points, colors = points[valid], colors[valid]
        cells = xp.floor(points / self.voxel_size).astype(xp.int64)
        bounded = xp.all((cells >= -(1 << 20)) & (cells < (1 << 20)), axis=1)
        cells, points = cells[bounded] + (1 << 20), points[bounded]
        colors = colors[bounded]
        keys = (cells[:, 0] << 42) | (cells[:, 1] << 21) | cells[:, 2]
        keys, unique = xp.unique(keys, return_index=True)
        points, colors = points[unique], colors[unique]
        if xp is not np:
            keys, points, colors = map(xp.asnumpy, (keys, points, colors))
        occupied = np.flatnonzero(self.keys >= 0)
        order = occupied[np.argsort(self.keys[occupied])]
        locations = np.searchsorted(self.keys[order], keys)
        matched = locations < len(order)
        matched[matched] &= self.keys[order[locations[matched]]] == keys[matched]
        slots = order[locations[matched]]
        # A new observation rescues a pressure victim unless the pilot is shrinking the budget.
        refresh = ~self.retiring[slots] | (self.count <= self.budget)
        slots = slots[refresh]
        observed = np.flatnonzero(matched)[refresh]
        self.retiring[slots] = False
        self.retire_revision[slots] = np.inf
        self.release_after[slots] = np.inf
        self.records[slots, 3, 3] = 0
        self.refreshed += len(slots)
        free = np.flatnonzero(self.keys < 0)
        room = max(0, self.budget - self.count)
        new = np.flatnonzero(~matched)
        admitted = min(len(new), len(free), room)
        # Spread admissions over the image instead of favoring one sorted spatial edge.
        selected = new[np.linspace(0, len(new) - 1, admitted, dtype=int)] if admitted else new[:0]
        slots = np.concatenate((slots, free[:admitted]))
        observed = np.concatenate((observed, selected))
        self.keys[slots] = keys[observed]
        self.seen[slots] = now
        self.records[slots, 0, :3] = points[observed]
        self.records[slots, 0, 3] = 0.85
        self.records[slots, 1, :3] = self.voxel_size * 0.65
        self.records[slots, 1, 3] = now + self.window + self.fade
        self.records[slots, 2, 0] = 1
        self.records[slots, 3, :3] = (colors[observed] / 255 - 0.5) / 0.2820947918
        self.dirty[slots] = True
        pressure = min(len(new) - admitted, max(1, self.budget // 4))
        pressure = max(0, pressure - int(self.retiring.sum()))
        candidates = np.flatnonzero((self.keys >= 0) & ~self.retiring & (self.seen < now))
        if pressure:
            self.retire(candidates[np.argsort(self.seen[candidates])[:pressure]])
