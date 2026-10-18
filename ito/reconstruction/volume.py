"""One surface per place: every mapped frame's depth, fused.

A depth network guesses each frame afresh. The same pot lands a few centimetres nearer
seen from the shelf than seen from the door, a plain wall a little in front of or behind
itself from frame to frame; each guess seeded on its own leaves a second pot and a wall of
blotches. Fused into a truncated signed distance field in sparse voxel blocks, every frame
both places surface near its depth and clears the space in front of it, weighted, so a
surface many views agree on outweighs one view's guess and space seen empty stays empty.
A frame is mapped by the depth the field gives along its rays, not by its own.
"""

import os

import numpy as np

VOXEL = float(os.environ.get("ITO_FUSE_VOXEL", "0.012"))  # metres
BLOCK = 8  # voxels along a block's edge
# Truncation: share of the observed depth, within metres (floor, ceiling). Learned depth
# is a few percent off; a band narrower than that would carve one view's surface with
# the next view's.
TRUNCATION = float(os.environ.get("ITO_FUSE_TRUNCATION", "0.05"))
TRUNCATION_RANGE = (0.025, 0.15)
MAX_WEIGHT = float(os.environ.get("ITO_FUSE_WMAX", "64"))
NEEDED = float(os.environ.get("ITO_FUSE_WMIN", "2"))  # weight both sides of a surface needs
# A pixel's weight falls off with the cosine of its ray to the optical axis to this power:
# learned depth at the rim of the wide rectified view is about twice as far off as in its
# middle (4% against 8-11% with the ToF's correction), a quarter to a sixth of the weight.
RADIAL = float(os.environ.get("ITO_FUSE_RADIAL", "8"))
# And with distance: (1 m / depth) to this power. Learned depth is off by a share of the
# distance, so a look from 0.5 m pins a surface 4 times as tightly as one from 2 m and must
# outweigh it 16 times, or close-ups take on the far view's centimetres.
DISTANCE = float(os.environ.get("ITO_FUSE_DISTANCE", "2"))
# A pixel takes the field's surface nearest its own depth within this share of it (wider, and a
# pixel of the wall beside a bookcase takes the bookcase's side); with none
# there it keeps its own (a surface the field does not hold yet, or one it holds wrongly:
# mapped, it contradicts what lies in front of it).
SNAP = float(os.environ.get("ITO_FUSE_SNAP", "0.08"))
# Space the field holds empty this surely (normalised distance, weight) refutes a pixel that
# would put a surface there, and a mapped splat sitting there.
FREE = (0.5, float(os.environ.get("ITO_FUSE_FREE_WEIGHT", "3")))
# ITO_FUSE_CONFIRM=T: a pixel trusted less than T (rim, no ToF zone near) that finds no field
# surface near its depth waits for the field to confirm it instead of mapping its own guess.
CONFIRM = float(os.environ.get("ITO_FUSE_CONFIRM", "0"))
FILL = os.environ.get("ITO_FUSE_FILL", "1") != "0"  # rays without depth search FILL_RANGE
FILL_RANGE = (0.2, 4.0)  # metres
STEPS = 24  # along a snap window
FILL_STEPS = 64
RAYS = 32768  # rays searched at once
BLOCKS = 4096  # visible blocks updated at once
OFFSET = 1 << 20  # block coordinates are packed in 21 bits each


class Volume:
    def __init__(self, device="cuda"):
        import torch

        self.torch = torch
        self.device = device
        i64 = dict(dtype=torch.int64, device=device)
        self.keys = torch.empty(0, **i64)  # sorted packed block coordinates
        self.slots = torch.empty(0, **i64)  # pool slot of each sorted key
        self.origins = torch.empty((0, 3), **i64)  # block coordinate of each slot
        self.sdf = torch.empty((0, BLOCK**3), dtype=torch.float16, device=device)
        self.weight = torch.empty((0, BLOCK**3), dtype=torch.float16, device=device)
        self.count = 0
        b = torch.arange(BLOCK, **i64)
        z, y, x = torch.meshgrid(b, b, b, indexing="ij")
        # Linear voxel index within a block: x + BLOCK * (y + BLOCK * z).
        self.local = torch.stack((x, y, z), -1).reshape(-1, 3)
        self.frames = 0
        self.filled = 0
        self.agreed = 0
        self.refuted = 0
        self.carved = 0

    @property
    def nbytes(self):
        return int(self.sdf.numel() * 2 + self.weight.numel() * 2 + self.origins.numel() * 8)

    def tensor(self, value):
        return self.torch.as_tensor(np.asarray(value), device=self.device, dtype=self.torch.float32)

    @staticmethod
    def pack(blocks):
        b = blocks + OFFSET
        return (b[..., 0] << 42) | (b[..., 1] << 21) | b[..., 2]

    def find(self, keys):
        """Pool slot of each packed block key, -1 where none is allocated."""
        torch = self.torch
        if not len(self.keys):
            return torch.full_like(keys, -1)
        position = torch.searchsorted(self.keys, keys).clamp_max(len(self.keys) - 1)
        return torch.where(self.keys[position] == keys, self.slots[position], -1)

    def allocate(self, blocks):
        torch = self.torch
        keys = torch.unique(self.pack(blocks))
        keys = keys[self.find(keys) < 0]
        if not len(keys):
            return
        needed = self.count + len(keys)
        if needed > len(self.sdf):
            capacity = max(needed, 2 * len(self.sdf), 4096)
            grown = capacity - len(self.sdf)
            self.sdf = torch.cat(
                (self.sdf, torch.ones((grown, BLOCK**3), dtype=self.sdf.dtype, device=self.device))
            )
            self.weight = torch.cat(
                (
                    self.weight,
                    torch.zeros((grown, BLOCK**3), dtype=self.weight.dtype, device=self.device),
                )
            )
            self.origins = torch.cat(
                (self.origins, torch.zeros((grown, 3), dtype=torch.int64, device=self.device))
            )
        slots = torch.arange(self.count, needed, dtype=torch.int64, device=self.device)
        b = keys
        self.origins[slots] = torch.stack(
            (
                ((b >> 42) & 0x1FFFFF) - OFFSET,
                ((b >> 21) & 0x1FFFFF) - OFFSET,
                (b & 0x1FFFFF) - OFFSET,
            ),
            -1,
        )
        self.count = needed
        keys = torch.cat((self.keys, keys))
        slots = torch.cat((self.slots, slots))
        order = torch.argsort(keys)
        self.keys, self.slots = keys[order], slots[order]

    def lookup(self, points):
        """(slot or -1, linear voxel index) of the voxel holding each world point."""
        torch = self.torch
        voxel = torch.floor(points / VOXEL).to(torch.int64)
        block = torch.div(voxel, BLOCK, rounding_mode="floor")
        local = voxel - block * BLOCK
        slot = self.find(self.pack(block))
        return slot, local[..., 0] + BLOCK * (local[..., 1] + BLOCK * local[..., 2])

    def rays(self, intrinsics, shape):
        torch = self.torch
        height, width = shape
        fx, fy, cx, cy = intrinsics
        v, u = torch.meshgrid(
            torch.arange(height, device=self.device, dtype=torch.float32),
            torch.arange(width, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        return torch.stack(((u - cx) / fx, -(v - cy) / fy, -torch.ones_like(u)), -1)

    @staticmethod
    def truncation(depth):
        return (depth * TRUNCATION).clamp(*TRUNCATION_RANGE)

    def integrate(self, depth, pose, intrinsics, trust=None):
        """depth: H x W axial metres (0: none) of a camera at pose (map-from-camera, Ito axes:
        x right, y up, looking down -z); trust: H x W factor on each pixel's weight."""
        torch = self.torch
        height, width = depth.shape
        fx, fy, cx, cy = intrinsics
        d = self.tensor(depth)
        rotation, origin = self.tensor(pose[:3, :3]), self.tensor(pose[:3, 3])
        rays = self.rays(intrinsics, depth.shape)
        cosine = 1 / torch.linalg.norm(rays, dim=-1)
        weights = cosine**RADIAL * d.clamp_min(0.3).reciprocal() ** DISTANCE
        if trust is not None:
            weights = weights * self.tensor(trust)
        # Allocate the blocks around every second pixel's surface, through its band.
        sample = d[::2, ::2]
        ray = rays[::2, ::2][sample > 0]
        observed = sample[sample > 0]
        if not len(observed):
            return
        band = self.truncation(observed)
        offsets = torch.tensor((-1.0, -0.5, 0.0, 0.5, 1.0), device=self.device)
        along = observed[:, None] + band[:, None] * offsets
        world = (ray[:, None, :] * along[..., None]) @ rotation.T + origin
        self.allocate(
            torch.div(
                torch.floor(world / VOXEL).to(torch.int64), BLOCK, rounding_mode="floor"
            ).reshape(-1, 3)
        )
        # Every allocated block in view: surface near the depth, free space before it.
        size = BLOCK * VOXEL
        centres = (self.origins[: self.count].float() + 0.5) * size
        seen = (centres - origin) @ rotation
        z = -seen[:, 2]
        reach = 0.87 * size
        far = float(observed.max()) + float(self.truncation(observed.max()))
        safe = z.clamp_min(0.05)
        u = cx + fx * seen[:, 0] / safe
        v = cy - fy * seen[:, 1] / safe
        margin = fx * reach / safe
        visible = (z > 0.05 - reach) & (z < far + reach)
        visible &= (u > -margin) & (u < width + margin) & (v > -margin) & (v < height + margin)
        slots = torch.nonzero(visible)[:, 0]
        local = (self.local.float() + 0.5) * VOXEL
        for chunk in torch.split(slots, BLOCKS):
            points = (self.origins[chunk].float() * size)[:, None, :] + local
            seen = (points - origin) @ rotation
            z = -seen[..., 2]
            safe = z.clamp_min(0.05)
            pu = torch.round(cx + fx * seen[..., 0] / safe).long()
            pv = torch.round(cy - fy * seen[..., 1] / safe).long()
            inside = (z > 0.05) & (pu >= 0) & (pu < width) & (pv >= 0) & (pv < height)
            pu, pv = pu.clamp(0, width - 1), pv.clamp(0, height - 1)
            surface = d[pv, pu]
            mu = self.truncation(surface)
            distance = surface - z
            update = inside & (surface > 0) & (distance > -mu)
            if not update.any():
                continue
            value = (distance / mu).clamp(-1, 1)
            w = torch.where(update, weights[pv, pu], 0)
            old_sdf = self.sdf[chunk].float()
            old_weight = self.weight[chunk].float()
            total = old_weight + w
            fused = torch.where(
                update, (old_sdf * old_weight + value * w) / total.clamp_min(1e-6), old_sdf
            )
            self.sdf[chunk] = fused.half()
            self.weight[chunk] = total.clamp_max(MAX_WEIGHT).half()
        self.frames += 1

    def free(self, points):
        """Whether the field holds each world point (N x 3 tensor) surely empty."""
        torch = self.torch
        if not self.count:
            return torch.zeros(len(points), dtype=torch.bool, device=self.device)
        slot, linear = self.lookup(points)
        index = slot.clamp_min(0) * BLOCK**3 + linear
        s = self.sdf.reshape(-1)[index].float()
        w = self.weight.reshape(-1)[index].float()
        return (slot >= 0) & (s > FREE[0]) & (w >= FREE[1])

    def search(self, rays, near, far, steps, rotation, origin, prefer=None):
        """Depth of a surface of the field between near and far along each ray (0: none):
        the one nearest prefer, or the first."""
        torch = self.torch
        out = torch.zeros_like(near)
        fraction = torch.linspace(0, 1, steps, device=self.device)
        flat_sdf, flat_weight = self.sdf.reshape(-1), self.weight.reshape(-1)
        for chunk in torch.split(torch.arange(len(near), device=self.device), RAYS):
            along = near[chunk, None] + (far[chunk, None] - near[chunk, None]) * fraction
            points = (rays[chunk, None, :] * along[..., None]) @ rotation.T + origin
            slot, linear = self.lookup(points)
            index = slot.clamp_min(0) * BLOCK**3 + linear
            s = torch.where(slot >= 0, flat_sdf[index].float(), 1.0)
            w = torch.where(slot >= 0, flat_weight[index].float(), 0.0)
            known = w >= NEEDED
            cross = (s[:, :-1] > 0) & (s[:, 1:] <= 0) & known[:, :-1] & known[:, 1:]
            s0, s1 = s[:, :-1], s[:, 1:]
            d0, d1 = along[:, :-1], along[:, 1:]
            found = d0 + (d1 - d0) * s0 / (s0 - s1).clamp_min(1e-6)
            if prefer is None:
                score = torch.arange(steps - 1, device=self.device, dtype=torch.float32)
                score = score.expand_as(found)
            else:
                score = (found - prefer[chunk, None]).abs()
            score = torch.where(cross, score, float("inf"))
            best = torch.argmin(score, 1)[:, None]
            hit = cross.any(1)
            out[chunk] = torch.where(hit, found.gather(1, best)[:, 0], 0)
        return out

    def raycast(self, pose, intrinsics, depth, trust=None):
        """Axial depth (H x W, 0: none) for a frame: the field's surface nearest each pixel's
        own depth within SNAP of it, else its own; where it has none, (FILL) the field's
        first surface over FILL_RANGE."""
        torch = self.torch
        own = self.tensor(depth)
        d = own.reshape(-1)
        out = d.clone()
        if not self.count:
            return depth
        rotation, origin = self.tensor(pose[:3, :3]), self.tensor(pose[:3, 3])
        rays = self.rays(intrinsics, depth.shape).reshape(-1, 3)
        seen = torch.nonzero(d > 0)[:, 0]
        if len(seen):
            target = d[seen]
            found = self.search(
                rays[seen],
                target / (1 + SNAP),
                target * (1 + SNAP),
                STEPS,
                rotation,
                origin,
                target,
            )
            free = self.free((rays[seen] * target[:, None]) @ rotation.T + origin)
            if CONFIRM and trust is not None:
                free |= self.tensor(trust).reshape(-1)[seen] < CONFIRM
            out[seen] = torch.where(found > 0, found, torch.where(free, 0, target))
            self.agreed += int((found > 0).sum())
            self.refuted += int((free & (found <= 0)).sum())
        if FILL:
            empty = torch.nonzero(d <= 0)[:, 0]
            if len(empty):
                found = self.search(
                    rays[empty],
                    torch.full((len(empty),), FILL_RANGE[0], device=self.device),
                    torch.full((len(empty),), FILL_RANGE[1], device=self.device),
                    FILL_STEPS,
                    rotation,
                    origin,
                )
                out[empty] = found
                self.filled += int((found > 0).sum())
        return out.reshape(depth.shape).cpu().numpy()

    def fuse(self, depth, pose, intrinsics, trust=None):
        """Integrate a frame's depth and give back the depth the field has along its rays."""
        self.integrate(depth, pose, intrinsics, trust)
        return self.raycast(pose, intrinsics, depth, trust)
