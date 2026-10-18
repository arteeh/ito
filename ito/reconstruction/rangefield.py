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

# Spread of a zone's ratio: log depth, colour (0..1 per channel) and pixels (at 512 across).
SIGMA_DEPTH = float(os.environ.get("ITO_TOF_FIELD_DEPTH", "0.15"))
SIGMA_COLOUR = 0.15
SIGMA_PIXELS = 40.0
PRIOR = 0.05  # weight of "no correction" against the zones' summed weights
FLAT = 0.1  # depth spread around a zone's pixel, of depth, beyond which it straddles an edge
LIMIT = math.log(2.0)  # a ratio past this is a zone hitting something the frame does not show


class RangeField:
    def __init__(self, device="cuda"):
        import torch

        self.torch = torch
        self.device = device
        self.used = 0
        self.frames = 0

    def correct(self, depth, image, hits):
        """depth: H x W metres (0: none), image: H x W x 3 uint8, hits: (row, column, axial
        metres) -> the corrected depth and the zones used."""
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
            return depth, 0
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
        numerator = torch.zeros_like(d)
        denominator = torch.zeros_like(d)
        for k in range(len(samples)):
            r, c = rows[k], columns[k]
            w = torch.exp(
                -((v - r) ** 2 + (u - c) ** 2) / (2 * (SIGMA_PIXELS * scale) ** 2)
                - (log_depth - log_depth[r, c]) ** 2 / (2 * SIGMA_DEPTH**2)
                - ((colour - colour[r, c]) ** 2).sum(-1) / (2 * SIGMA_COLOUR**2)
            )
            numerator += w * ratios[k]
            denominator += w
        field = numerator / (denominator + PRIOR)
        corrected = torch.where(d > 0, d * torch.exp(field), 0)
        self.used += len(samples)
        self.frames += 1
        return corrected.cpu().numpy(), len(samples)
