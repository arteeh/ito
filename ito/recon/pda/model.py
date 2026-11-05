"""Prompt Depth Anything on the GPU, behind the Model the depth process (ito.tofmesh.depth)
runs: a camera frame and a coarse metric depth prompt (the ToF's ranges, seen from the camera)
in, a metric depth map out (restored from feat/pda, c484cfe)."""

import numpy as np
import torch
from transformers import PromptDepthAnythingForDepthEstimation

from ito.recon import models

from . import SIZES

INPUT = (364, 644)  # the frame as the model sees it (height, width; multiples of 14)
PROMPT = (36, 64)  # the depth prompt (height, width)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
FINE = 4  # prompt: points per ToF zone along each axis
NEAREST = 0.05  # metres: nothing nearer is believed in the prompt


class Model:
    """predict(): a camera frame's depth (h, w) in metres along -Z. Where the ToF gives no prompt
    (too little of the frame, or one distance only: a wall or shelf right in front), the model
    runs without one; that map is not metric (`metric` False for it), and the depth process
    scales it by the ToF's real readings, or skips it when they cannot scale it."""

    metric = True

    def __init__(self, size, precision):
        # Half: the whole network in float16, as its Transformers port runs it; Full: float32.
        self.dtype = torch.float16 if precision == "half" else torch.float32
        net = PromptDepthAnythingForDepthEstimation.from_pretrained(
            models() / SIZES[size], local_files_only=True, dtype=self.dtype
        )
        self.net = net.to("cuda").eval()
        self.mean = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)

    def predict(self, image, intrinsics, support, zones, sensor_from_camera):
        h, w = image.shape[:2]
        camera_from_sensor = np.linalg.inv(sensor_from_camera)
        depth = prompt(zones, camera_from_sensor, intrinsics, (w, h))
        self.metric = depth is not None
        resize = torch.nn.functional.interpolate
        x = torch.from_numpy(image).to("cuda").permute(2, 0, 1)[None].float() / 255
        x = resize(x, INPUT, mode="bilinear", antialias=True, align_corners=False)
        x = ((x - self.mean) / self.std).to(self.dtype)
        p = None if depth is None else torch.from_numpy(depth).to("cuda")[None, None].to(self.dtype)
        out = self.net(pixel_values=x, prompt_depth=p).predicted_depth
        out = resize(out[:, None].float(), (h, w), mode="bilinear", align_corners=False)
        return out[0, 0].cpu().numpy()


def prompt(zones, camera_from_sensor, intrinsics, size):
    """The depth prompt (PROMPT cells, metres along the camera's -Z) of a camera frame of `size`
    (w, h) at the pinhole `intrinsics` (fx, fy, cx, cy): the ToF frame's surface (zones.Zones,
    the sensor placed by camera_from_sensor, 4x4), nearest first per cell, the cells it does
    not reach filled from the nearest ones it does. None where the ToF sees too little of the
    frame or only one distance (the model scales its answer between the prompt's nearest and
    farthest, so they must differ)."""
    g = FINE
    across = (np.arange(zones.cols * g) + 0.5) / g * zones.step[0] - zones.fov[0] / 2
    up = zones.fov[1] / 2 - (np.arange(zones.rows * g) + 0.5) / g * zones.step[1]
    a, u = np.meshgrid(across, up)
    rays = zones.direction(a.ravel(), u.ravel())
    ranges = zones.lookup(rays)[0]
    ok = np.isfinite(ranges)
    p = (rays[ok] * ranges[ok, None]) @ camera_from_sensor[:3, :3].T + camera_from_sensor[:3, 3]
    z = -p[:, 2]
    p, z = p[z > NEAREST], z[z > NEAREST]
    fx, fy, cx, cy = intrinsics
    (w, h), (ph, pw) = size, PROMPT
    col = np.floor((cx + fx * p[:, 0] / z + 0.5) * pw / w).astype(np.int64)
    row = np.floor((cy - fy * p[:, 1] / z + 0.5) * ph / h).astype(np.int64)
    seen = (col >= 0) & (col < pw) & (row >= 0) & (row < ph)
    depth = np.full((ph, pw), np.inf)
    np.minimum.at(depth, (row[seen], col[seen]), z[seen])
    known = np.isfinite(depth)
    if known.sum() < 8 or np.ptp(depth[known]) < 0.05:
        return None
    while not known.all():  # each hole cell: the mean of its measured (or filled) neighbours
        padded = np.pad(np.where(known, depth, 0.0), 1)
        counted = np.pad(known.astype(np.float64), 1)
        total = sum(padded[1 + dy : 1 + dy + ph, 1 + dx : 1 + dx + pw]
                    for dy in (-1, 0, 1) for dx in (-1, 0, 1))  # fmt: skip
        count = sum(counted[1 + dy : 1 + dy + ph, 1 + dx : 1 + dx + pw]
                    for dy in (-1, 0, 1) for dx in (-1, 0, 1))  # fmt: skip
        grow = ~known & (count > 0)
        depth[grow] = total[grow] / count[grow]
        known |= grow
    return depth.astype(np.float32)
