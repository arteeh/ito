"""A neural depth guess may add provisional geometry; agreement permits erasing history.

Carving from an unsupported, suddenly farther surface makes holes in an already seen
floor. Keep that older surface until another view confirms the new depth. All dense
work stays on the device, with only three observations retained.
"""

from collections import deque

import numpy as np


class DepthSupport:
    def __init__(self):
        self.history = deque(maxlen=3)

    def observe(self, cloud, pose):
        import torch

        height, width = cloud.shape
        fx, fy, cx, cy = cloud.intrinsics
        device = cloud.points.device
        depth = -cloud.points[:, 2]
        supported = torch.zeros_like(cloud.valid)
        for points, valid, origin in self.history:
            transform = torch.as_tensor(
                np.linalg.inv(pose) @ origin, device=device, dtype=torch.float32
            )
            local = points @ transform[:3, :3].T + transform[:3, 3]
            z = -local[:, 2]
            u = torch.round(cx + fx * local[:, 0] / z.clamp_min(1e-6)).long()
            v = torch.round(cy - fy * local[:, 1] / z.clamp_min(1e-6)).long()
            visible = valid & torch.isfinite(local).all(1) & (z > 0)
            visible &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
            nearest = torch.full_like(depth, torch.inf)
            nearest.scatter_reduce_(
                0,
                (v[visible] * width + u[visible]),
                z[visible],
                reduce="amin",
                include_self=True,
            )
            supported |= torch.isfinite(nearest) & ((nearest - depth).abs() < 0.08 * depth)
        self.history.append((cloud.points.detach(), cloud.valid.detach(), np.array(pose)))
        return supported & cloud.valid
