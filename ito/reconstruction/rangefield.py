"""A frame's depth bent to the ToF's zones where they read it.

A learned depth guess is right in shape far more often than in distance: from the floor, a
pot across the room comes out a third nearer than it stands while the wall behind it does not.
The ToF reads 64 distances in the middle of the view. Each is a log ratio between the reading
and the frame's depth where the zone's ray lands; those ratios spread over the frame through
pixels whose depth and colour resemble the zone's (a joint bilateral filter), fading to no
correction away from them. So the pot moves back to where the ToF found it, the wall stays.
"""

import math
import os

import numpy as np

# Spread of a zone's ratio: log depth, colour (0..1 per channel) and pixels (at 512 across).
SIGMA_DEPTH = float(os.environ.get("ITO_TOF_FIELD_DEPTH", "0.15"))
SIGMA_COLOUR = 0.15
SIGMA_PIXELS = 40.0
PRIOR = 0.05  # weight of "no correction" against the zones' summed weights
FLAT = 0.1  # depth spread around a zone's pixel, of depth, beyond which it straddles an edge
# Fusion's trust in a pixel: depth no zone read is about twice as far off as depth beside one.
TRUST_FLOOR = 0.3
TRUST_HALF = 0.3  # summed zone weight at which trust is half way up
LIMIT = math.log(2.0)  # a ratio past this is a zone hitting something the frame does not show


def colour_distance(colour, at):
    """Squared colour distance (K x H x W) of each pixel of colour (H x W x 3) to each of at
    (K x 3), without a K x H x W x 3 intermediate."""
    import torch

    dot = torch.einsum("hwc,kc->khw", colour, at)
    return ((colour**2).sum(-1)[None] - 2 * dot + (at**2).sum(-1)[:, None, None]).clamp_min(0)


class RangeField:
    def __init__(self, device="cuda"):
        import torch

        self.torch = torch
        self.device = device
        self.used = 0
        self.frames = 0

    def nearest(self, depth, footprints, intrinsics):
        """Zone samples (row, column, axial metres) from footprints (sensors.zone_footprints):
        a VL53 reads the nearest target anywhere in a zone, so its reading belongs to the
        footprint's pixel whose surface lies nearest the sensor, at the depth along that
        pixel's ray where the distance from the sensor is the reading."""
        fx, fy, cx, cy = intrinsics
        samples = []
        for pixels, reading, origin in footprints:
            rows, columns = pixels[:, 0], pixels[:, 1]
            d = depth[rows, columns]
            keep = d > 0
            if keep.sum() < 3:
                continue
            rows, columns, d = rows[keep], columns[keep], d[keep]
            rays = np.stack(((columns - cx) / fx, -(rows - cy) / fy, -np.ones_like(d)), -1)
            k = int(np.argmin(np.linalg.norm(rays * d[:, None] - origin, axis=1)))
            u = rays[k]
            uo, uu = float(u @ origin), float(u @ u)
            disc = uo * uo - uu * (float(origin @ origin) - reading * reading)
            if disc <= 0:
                continue
            samples.append((int(rows[k]), int(columns[k]), (uo + math.sqrt(disc)) / uu))
        return samples

    def correct(self, depth, image, hits):
        """depth: H x W metres (0: none), image: H x W x 3 uint8, hits: (row, column, axial
        metres) -> the corrected depth, the zones used and each pixel's trust: from TRUST_FLOOR
        far from any zone that read it up to 1 beside one."""
        torch = self.torch
        height, width = depth.shape
        samples = []
        for row, column, target in hits:
            r0, r1 = max(row - 2, 0), min(row + 3, height)
            c0, c1 = max(column - 2, 0), min(column + 3, width)
            around = depth[r0:r1, c0:c1]
            here = depth[row, column]
            if here <= 0 or (around <= 0).any():
                continue
            if around.max() - around.min() > FLAT * here:
                continue
            ratio = math.log(target / here)
            if abs(ratio) < LIMIT:
                samples.append((row, column, ratio))
        if not samples:
            return depth, 0, None
        scale = width / 512
        d = torch.as_tensor(depth, device=self.device, dtype=torch.float32)
        log_depth = torch.log(d.clamp_min(1e-3))
        colour = torch.as_tensor(image, device=self.device, dtype=torch.float32) / 255
        rows = torch.tensor([s[0] for s in samples], device=self.device)
        columns = torch.tensor([s[1] for s in samples], device=self.device)
        ratios = torch.tensor([s[2] for s in samples], device=self.device, dtype=torch.float32)
        v, u = torch.meshgrid(
            torch.arange(height, device=self.device, dtype=torch.float32),
            torch.arange(width, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        # All zones at once: K x H x W weights (64 zones of a 512 x 288 frame, 38 MB).
        w = torch.exp(
            -((v[None] - rows[:, None, None]) ** 2 + (u[None] - columns[:, None, None]) ** 2)
            / (2 * (SIGMA_PIXELS * scale) ** 2)
            - (log_depth[None] - log_depth[rows, columns][:, None, None]) ** 2
            / (2 * SIGMA_DEPTH**2)
            - colour_distance(colour, colour[rows, columns]) / (2 * SIGMA_COLOUR**2)
        )
        numerator = (w * ratios[:, None, None]).sum(0)
        denominator = w.sum(0)
        field = numerator / (denominator + PRIOR)
        corrected = torch.where(d > 0, d * torch.exp(field), 0)
        trust = TRUST_FLOOR + (1 - TRUST_FLOOR) * denominator / (denominator + TRUST_HALF)
        self.used += len(samples)
        self.frames += 1
        return corrected.cpu().numpy(), len(samples), trust.cpu().numpy()
