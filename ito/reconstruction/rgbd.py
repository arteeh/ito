"""Projective surface fusion; RGB-D distances are metres, SLAM's its own units.

Splats stay until the budget needs their slots: the scene builds up into one room the pilot
can look back at. Under pressure, old low-confidence surfaces fade out together.
"""

import numpy as np

from .surface import ellipse, orientation, patches, surfaces

LEVELS = 32  # Power-of-two cell sizes from 2^-16 to 2^15 world units.
KEEP = 1e30  # Fade deadline of a splat nothing has evicted: never.


class RGBDBackend:
    def __init__(
        self,
        capacity,
        intrinsics,
        *,
        voxel_size=0.08,
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
        self.voxel_size, self.fade = voxel_size, fade_seconds
        self.records = np.zeros((capacity, 4, 4), np.float32)
        self.keys = np.full(capacity, -1, np.int64)
        self.seen = np.full(capacity, -np.inf)
        self.confidence = np.zeros(capacity, np.float32)
        self.normals = np.zeros((capacity, 3), np.float32)
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
        ).astype(xp.float32)

    @property
    def count(self):
        return int(np.count_nonzero(self.keys >= 0))

    def resize(self, budget, now):
        if budget > len(self.keys):
            extra = budget - len(self.keys)
            self.records = np.pad(self.records, ((0, extra), (0, 0), (0, 0)))
            self.keys = np.pad(self.keys, (0, extra), constant_values=-1)
            self.seen = np.pad(self.seen, (0, extra), constant_values=-np.inf)
            self.confidence = np.pad(self.confidence, (0, extra))
            self.normals = np.pad(self.normals, ((0, extra), (0, 0)))
            self.retiring = np.pad(self.retiring, (0, extra))
            self.dirty = np.pad(self.dirty, (0, extra))
            self.retire_revision = np.pad(self.retire_revision, (0, extra), constant_values=np.inf)
            self.release_after = np.pad(self.release_after, (0, extra), constant_values=np.inf)
        self.budget = budget
        excess = max(0, self.count - budget - int(self.retiring.sum()))
        candidates = np.flatnonzero((self.keys >= 0) & ~self.retiring)
        if excess:
            self.retire(self.oldest(candidates, excess))

    def oldest(self, candidates, n):
        # A partial selection: a full budget is too many slots to sort every frame.
        if n >= len(candidates):
            return candidates
        priority = self.seen[candidates] + np.log1p(self.confidence[candidates])
        return candidates[np.argpartition(priority, n - 1)[:n]]

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
        expired = (self.keys >= 0) & self.retiring & (self.release_after <= now)
        self.keys[expired] = -1
        self.records[expired] = 0
        self.retiring[expired] = False
        self.retire_revision[expired] = np.inf
        self.release_after[expired] = np.inf
        self.dirty[expired] = True

    def carve(self, camera, fx, fy, cx, cy, depth, now):
        """Retire older splats this view sees through or replaces with its own.

        A surface in view is what the camera sees now: earlier splats on it that this
        frame did not refresh, or in front of it, fade out together, while surfaces out
        of view or behind it stay. Otherwise kept splats pile up wherever the scene moved
        or the map was misplaced, as a second ghost room. Run it after integrating the
        frame. Depth is axial along the camera's -Z, in world units, at pixels of these
        intrinsics; zero marks no measurement.
        """
        xp = self.xp
        live = np.flatnonzero((self.keys >= 0) & ~self.retiring & (self.seen < now))
        if not len(live):
            return
        centers = xp.asarray(self.records[live, 0, :3])
        size = xp.asarray(self.records[live, 1, 0])
        camera = xp.asarray(camera, dtype=xp.float32)
        local = (centers - camera[:3, 3]) @ camera[:3, :3]
        z = -local[:, 2]
        ahead = z > 1e-6
        safe = xp.where(ahead, z, 1)
        u = xp.rint(cx + fx * local[:, 0] / safe).astype(xp.int64)
        v = xp.rint(cy - fy * local[:, 1] / safe).astype(xp.int64)
        height, width = depth.shape
        seen = ahead & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        observed = xp.zeros(len(live), xp.float32)
        observed[seen] = depth[v[seen], u[seen]]
        # On or in front of the seen surface, within depth noise and the splat's size.
        stale = (observed > 0) & (z < observed * 1.1 + 2 * size)
        if xp is not np:
            stale = xp.asnumpy(stale)
        if stale.any():
            self.retire(live[stale])

    def integrate(self, rgb, depth, camera, now):
        xp = self.xp
        z = xp.asarray(depth, dtype=xp.float32)
        valid = xp.isfinite(z) & (z > 0) & (z < 1000)
        local = self.rays * xp.where(valid, z, 0)[..., None]
        normals, sizes, valid, axes = surfaces(local, valid, xp)
        transform = xp.asarray(camera, dtype=xp.float32)
        points = local @ transform[:3, :3].T + transform[:3, 3]
        normals = normals @ transform[:3, :3].T
        axes = axes @ transform[:3, :3].T
        colors = xp.asarray(rgb)
        pixel_sizes = sizes
        # Uniform patches may coalesce up to this world-space footprint; texture
        # and normal discontinuities retain their original one-pixel samples.
        samples, sample_colors, sample_normals, sample_sizes, sample_axes, sizes = patches(
            points, colors, normals, sizes, valid, self.voxel_size, xp, axes
        )
        claimed = xp.zeros(z.shape, bool)
        live = np.flatnonzero((self.keys >= 0) & ~self.retiring)
        if len(live):
            old = xp.asarray(self.records[live, 0, :3])
            projected = (old - transform[:3, 3]) @ transform[:3, :3]
            distance = -projected[:, 2]
            i = self.intrinsics
            safe = xp.maximum(distance, 1e-6)
            u = xp.rint(i.cx + i.fx * projected[:, 0] / safe).astype(xp.int64)
            v = xp.rint(i.cy - i.fy * projected[:, 1] / safe).astype(xp.int64)
            inside = (distance > 0) & (u >= 0) & (u < i.width) & (v >= 0) & (v < i.height)
            u, v = xp.clip(u, 0, i.width - 1), xp.clip(v, 0, i.height - 1)
            target = points[v, u]
            normal = normals[v, u]
            delta = target - old
            footprint = sizes[v, u]
            old_size = xp.asarray(np.sqrt(np.prod(self.records[live, 1, :2], axis=1))) * 2
            old_normal = xp.asarray(self.normals[live])
            old_color = xp.asarray(self.records[live, 3, :3]) * 0.2820947918 + 0.5
            # Depth noise is independent of footprint. The tangent gate prevents
            # nearby samples sliding along the surface and smearing image detail.
            plane = xp.abs(xp.sum(delta * normal, axis=1))
            tangent2 = xp.maximum(0, xp.sum(delta * delta, axis=1) - plane**2)
            match = (
                inside
                & valid[v, u]
                & (plane < xp.maximum(footprint, z[v, u] * 0.015))
                & (tangent2 < (1.5 * xp.maximum(footprint, old_size)) ** 2)
                & (xp.abs(xp.sum(old_normal * normal, axis=1)) > 0.8)
                & (xp.max(xp.abs(old_color - colors[v, u] / 255), axis=1) < 0.3)
                & (old_size < 2 * footprint)
            )
            matched = xp.flatnonzero(match)
            # Claim the observed area, not just its centre pixel. Otherwise a
            # subpixel camera shift seeds new neighbours beside every coarse patch.
            w, x, y, zq = xp.asarray(self.records[live, 2])[matched].T
            major = xp.stack(
                (1 - 2 * (y * y + zq * zq), 2 * (x * y + w * zq), 2 * (x * zq - w * y)), axis=1
            )
            minor = xp.stack(
                (2 * (x * y - w * zq), 1 - 2 * (x * x + zq * zq), 2 * (y * zq + w * x)), axis=1
            )
            radii = xp.asarray(self.records[live, 1, :2])[matched]
            major = (major * radii[:, :1]) @ transform[:3, :3]
            minor = (minor * radii[:, 1:]) @ transform[:3, :3]
            local = projected[matched]
            distance = safe[matched]

            def project(axis):
                return xp.stack(
                    (
                        i.fx * (axis[:, 0] + local[:, 0] * axis[:, 2] / distance) / distance,
                        i.fy * (axis[:, 1] + local[:, 1] * axis[:, 2] / distance) / distance,
                    ),
                    axis=1,
                )

            extent = xp.sqrt(project(major) ** 2 + project(minor) ** 2)
            radius_x, radius_y = extent[:, 0], extent[:, 1]
            center_u = i.cx + i.fx * projected[matched, 0] / safe[matched]
            center_v = i.cy - i.fy * projected[matched, 1] / safe[matched]
            x0 = xp.clip(xp.floor(center_u - radius_x + 0.5).astype(xp.int64), 0, i.width)
            y0 = xp.clip(xp.floor(center_v - radius_y + 0.5).astype(xp.int64), 0, i.height)
            x1 = xp.clip(
                xp.maximum(x0 + 1, xp.floor(center_u + radius_x + 0.5)).astype(xp.int64), 0, i.width
            )
            y1 = xp.clip(
                xp.maximum(y0 + 1, xp.floor(center_v + radius_y + 0.5)).astype(xp.int64),
                0,
                i.height,
            )
            coverage = xp.zeros((i.height + 1, i.width + 1), xp.int32)
            for yy, xx, sign in ((y0, x0, 1), (y1, x1, 1), (y0, x1, -1), (y1, x0, -1)):
                xp.add.at(coverage, (yy, xx), sign)
            claimed = coverage.cumsum(0).cumsum(1)[:-1, :-1] > 0
            observed = (
                target[matched],
                colors[v[matched], u[matched]],
                normal[matched],
                footprint[matched],
                axes[v[matched], u[matched]]
                * (footprint[matched] / xp.maximum(pixel_sizes[v[matched], u[matched]], 1e-8))[
                    :, None, None
                ],
            )
            if xp is not np:
                matched = xp.asnumpy(matched)
                observed = tuple(xp.asnumpy(value) for value in observed)
            self.fuse(live[matched], *observed, now)
        # A merged patch's representative projects to its centre, not its corner.
        sample_local = (samples - transform[:3, 3]) @ transform[:3, :3]
        i = self.intrinsics
        safe = xp.maximum(-sample_local[:, 2], 1e-6)
        u = xp.clip(
            xp.rint(i.cx + i.fx * sample_local[:, 0] / safe).astype(xp.int64), 0, i.width - 1
        )
        v = xp.clip(
            xp.rint(i.cy - i.fy * sample_local[:, 1] / safe).astype(xp.int64), 0, i.height - 1
        )
        new = ~claimed[v, u]
        self.integrate_points(
            samples[new],
            sample_colors[new],
            now,
            sample_sizes[new],
            sample_normals[new],
            sample_axes[new],
        )
        i = self.intrinsics
        self.carve(camera, i.fx, i.fy, i.cx, i.cy, xp.where(valid, z, 0), now)

    def fuse(self, slots, points, colors, normals, sizes, axes, now):
        # Cap history so a long-observed surface still adapts to a changed room.
        weight = np.minimum(self.confidence[slots], 31)
        old = self.records[slots]
        tangent, scales = ellipse(axes, np)
        old_size = np.sqrt(np.prod(old[:, 1, :2], axis=1)) * 2
        # A distant pixel must not erase colour detail measured up close.
        observation = np.where(weight > 0, np.minimum(1, (old_size / sizes) ** 2), 1)
        alpha = observation / (weight + observation)
        old[:, 0, :3] += alpha[:, None] * (points - old[:, 0, :3])
        rgb = (colors / 255 - 0.5) / 0.2820947918
        old[:, 3, :3] += alpha[:, None] * (rgb - old[:, 3, :3])
        previous = self.normals[slots]
        normals = normals * np.where(np.sum(previous * normals, axis=1) < 0, -1, 1)[:, None]
        normal = previous + alpha[:, None] * (normals - previous)
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        self.normals[slots] = normal
        finer = (weight == 0) | (sizes < old_size)
        w, x, y, z = old[:, 2].T
        previous_tangent = np.stack(
            (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)), axis=1
        )
        tangent = np.where(finer[:, None], tangent, previous_tangent)
        tangent -= np.sum(tangent * normal, axis=1, keepdims=True) * normal
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
        old[:, 1, :2] = np.where(finer[:, None], scales, old[:, 1, :2])
        old[:, 1, 2] = old[:, 1, 1] * 0.05
        old[:, 2] = orientation(normal, np, tangent)
        old[:, 0, 3] = 0.98
        old[:, 1, 3] = KEEP
        old[:, 3, 3] = 0
        self.records[slots] = old
        self.confidence[slots] = weight + observation
        self.seen[slots] = now
        self.dirty[slots] = True
        self.refreshed += len(slots)

    def integrate_points(self, points, colors, now, sizes=None, normals=None, axes=None):
        """Fuse world-space dense points through the same budget, fade and eviction policy.

        Sizes are measured pixel footprints in world units. Only the lookup cells
        snap to powers of two; render scales retain the measured surface footprint.
        """
        xp = self.xp
        valid = xp.all(xp.isfinite(points), axis=1)
        points, colors = points[valid], colors[valid]
        if not len(points):
            return
        normals = (
            normals[valid]
            if normals is not None
            else xp.tile(xp.asarray([0.0, 0.0, 1.0]), (len(points), 1))
        )
        footprint = sizes[valid] if sizes is not None else xp.full(len(points), self.voxel_size)
        if axes is None:
            q = orientation(normals, xp)
            w, x, y, z = q.T
            tangent = xp.stack(
                (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)), axis=1
            )
            axes = (
                xp.stack((tangent, xp.cross(normals, tangent)), axis=1) * footprint[:, None, None]
            )
        else:
            axes = axes[valid]
        # Cell edges stay float64: a float32 4 cm voxel is not 4 cm, and surfaces on
        # exact voxel multiples would then flip between cells every frame.
        if sizes is None:
            levels = xp.zeros(len(points), xp.int64)
            cell = xp.full(len(points), self.voxel_size)
        else:
            levels = xp.clip(xp.floor(xp.log2(sizes[valid])), -LEVELS // 2, LEVELS // 2 - 1)
            levels = levels.astype(xp.int64)
            cell = 2.0**levels
            levels += LEVELS // 2
        cells = xp.floor(points / cell[:, None]).astype(xp.int64)
        bounded = xp.all((cells >= -(1 << 18)) & (cells < (1 << 18)), axis=1)
        cells, points, cell = cells[bounded] + (1 << 18), points[bounded], cell[bounded]
        colors, levels = colors[bounded], levels[bounded]
        normals, footprint, axes = normals[bounded], footprint[bounded], axes[bounded]
        keys = (levels << 57) | (cells[:, 0] << 38) | (cells[:, 1] << 19) | cells[:, 2]
        keys, unique = xp.unique(keys, return_index=True)
        points, colors, cell = points[unique], colors[unique], cell[unique]
        normals, footprint, axes = normals[unique], footprint[unique], axes[unique]
        # Match against the live slots where the keys are: sorting a full budget of
        # keys on the CPU costs more than a dense frame's whole GPU pass.
        occupied = np.flatnonzero(self.keys >= 0)
        stored = xp.asarray(self.keys[occupied])
        order = xp.argsort(stored)
        stored = stored[order]
        locations = xp.minimum(xp.searchsorted(stored, keys), max(len(stored) - 1, 0))
        matched = stored[locations] == keys if len(stored) else xp.zeros(len(keys), bool)
        if xp is not np:
            keys, points, colors, cell, locations, matched, order, normals, footprint, axes = map(
                xp.asnumpy,
                (keys, points, colors, cell, locations, matched, order, normals, footprint, axes),
            )
        slots = occupied[order[locations[matched]]]
        # A new observation rescues a pressure victim unless the pilot is shrinking the budget.
        refresh = (~self.retiring[slots] | (self.count <= self.budget)) & (self.seen[slots] < now)
        incoming = np.flatnonzero(matched)
        refresh &= np.abs(np.sum(self.normals[slots] * normals[incoming], axis=1)) > 0.8
        refresh &= (
            np.linalg.norm(self.records[slots, 0, :3] - points[incoming], axis=1)
            < 2 * footprint[incoming]
        )
        old_rgb = self.records[slots, 3, :3] * 0.2820947918 + 0.5
        refresh &= np.max(np.abs(old_rgb - colors[incoming] / 255), axis=1) < 0.3
        slots = slots[refresh]
        observed = np.flatnonzero(matched)[refresh]
        self.retiring[slots] = False
        self.retire_revision[slots] = np.inf
        self.release_after[slots] = np.inf
        self.records[slots, 3, 3] = 0
        free = np.flatnonzero(self.keys < 0)
        room = max(0, self.budget - self.count)
        new = np.flatnonzero(~matched)
        admitted = min(len(new), len(free), room)
        # Spread admissions over the image instead of favoring one sorted spatial edge.
        selected = new[np.linspace(0, len(new) - 1, admitted, dtype=int)] if admitted else new[:0]
        self.fuse(
            slots,
            points[observed],
            colors[observed],
            normals[observed],
            footprint[observed],
            axes[observed],
            now,
        )
        slots, observed = free[:admitted], selected
        self.keys[slots] = keys[observed]
        self.confidence[slots] = 0
        self.normals[slots] = normals[observed]
        self.records[slots] = 0
        self.fuse(
            slots,
            points[observed],
            colors[observed],
            normals[observed],
            footprint[observed],
            axes[observed],
            now,
        )
        # Keep a little room free: splats the view replaces hold their slots while they
        # fade, and the next frame's own splats must not wait for them.
        spare = max(0, self.budget - self.count) + int(self.retiring.sum())
        wanted = len(new) - admitted + self.budget // 16
        pressure = min(max(0, wanted - spare), max(1, self.budget // 4))
        candidates = np.flatnonzero((self.keys >= 0) & ~self.retiring & (self.seen < now))
        if pressure:
            self.retire(self.oldest(candidates, pressure))
