"""Depth more than one view agrees on, before it becomes map.

A pairwise network guesses each frame's depth afresh: a surface lands a few percent nearer or
farther each time, and silhouettes ramp into the space between objects. Laid onto the map
one guess after another, those guesses become doubled edges, seams and streaks.

So a frame's pixel is mapped only where another view saw the same surface: its point, put into
each reference view (the last few mapped frames and the retained views facing most like it),
lands within AGREE of that view's own depth. Its depth becomes the mean of the agreeing views'.
A point a reference sees straight through (that view's surface lies well behind it) is a
guess in free space; it needs more views for it than against it. A first glimpse of new
surface waits for the next frame to confirm it.
"""

import math
from collections import deque

import numpy as np

AGREE = 0.05  # of depth: two views this close see one surface
RECENT = 4  # last mapped frames to check a new one against
RETAINED = 4  # and retained views facing most like it
NEEDED = 1  # other views that must agree (the frame itself is one more)
FACING = math.cos(math.radians(60))
NEAREST = 1.5  # metres: retained views further than this from the camera are not asked


class Consensus:
    def __init__(self, device="cuda"):
        import torch

        self.torch = torch
        self.device = device
        self.recent = deque(maxlen=RECENT)
        self.kept = 0
        self.dropped = 0

    def references(self, pose, views):
        chosen = list(self.recent)
        seen = {round(r[3], 4) for r in chosen}
        scored = []
        for view in views:
            if round(view.captured, 4) in seen or view.depth is None:
                continue
            cosine = float(view.pose[:3, 2] @ pose[:3, 2])
            distance = float(np.linalg.norm(view.pose[:3, 3] - pose[:3, 3]))
            if cosine > FACING and distance < NEAREST:
                scored.append((1 - cosine + 0.2 * distance, view))
        scored.sort(key=lambda item: item[0])
        for _, view in scored[:RETAINED]:
            chosen.append((view.depth, view.pose, view.intrinsics, view.captured))
        return chosen

    def fuse(self, depth, pose, intrinsics, captured, views):
        """depth (H x W metres, 0: none) of a camera at pose (map-from-camera, Ito axes) ->
        (depth agreed by NEEDED other views, 0 elsewhere; agreeing views per pixel)."""
        references = self.references(pose, views)
        self.recent.append((depth.copy(), pose.copy(), intrinsics, captured))
        fused, agree = agreed(depth, pose, intrinsics, references, self.torch, self.device)
        kept = int((fused > 0).sum())
        self.kept += kept
        self.dropped += int((depth > 0).sum()) - kept
        return fused, agree


def agreed(depth, pose, intrinsics, references, torch, device="cuda"):
    """depth where NEEDED of references (depth, pose, intrinsics, ...) see the same surface,
    as the mean of the agreeing views' depths (0 elsewhere), and the agreeing count."""

    def tensor(value):
        return torch.as_tensor(np.asarray(value), device=device, dtype=torch.float32)

    height, width = depth.shape
    fx, fy, cx, cy = intrinsics
    d = tensor(depth)
    v, u = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    local = torch.stack(((u - cx) / fx * d, -(v - cy) / fy * d, -d), -1)
    world = local @ tensor(pose[:3, :3]).T + tensor(pose[:3, 3])
    total = d.clone()
    agree = torch.zeros_like(d)
    against = torch.zeros_like(d)
    for ref_depth, ref_pose, ref_intrinsics, *_ in references:
        rh, rw = ref_depth.shape
        rfx, rfy, rcx, rcy = ref_intrinsics
        seen = (world - tensor(ref_pose[:3, 3])) @ tensor(ref_pose[:3, :3])
        z = -seen[..., 2]
        safe = z.clamp_min(1e-3)
        ru = torch.round(rcx + rfx * seen[..., 0] / safe).long()
        rv = torch.round(rcy - rfy * seen[..., 1] / safe).long()
        inside = (z > 0.01) & (ru >= 0) & (ru < rw) & (rv >= 0) & (rv < rh) & (d > 0)
        found = torch.zeros_like(d)
        found[inside] = tensor(ref_depth)[rv[inside], ru[inside]]
        known = inside & (found > 0)
        relative = (found - z) / safe
        ok = known & (relative.abs() < AGREE)
        # The reference's surface, moved along this camera's ray by the same share.
        total += torch.where(ok, d * found / safe, 0)
        agree += ok
        against += known & (relative > AGREE)
    keep = (agree >= NEEDED) & (against < agree) & (d > 0)
    fused = torch.where(keep, total / (agree + 1), 0)
    return fused.cpu().numpy(), agree.cpu().numpy()
