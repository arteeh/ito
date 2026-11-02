"""Prompt Depth Anything (PromptDA, CVPR 2025) in its own process: a camera frame and a coarse
metric depth prompt (the ToF's ranges, seen from the camera) in, a metric depth map out.

It runs at whatever rate it manages, one request at a time (the mesh worker sends the next when
the last answer is in), at below-normal process priority, so it never holds the mesh worker, the
link or the windows: a slow PDA only updates the mesh less often. The models are bundled in the
release's models folder (release/windows.py), never downloaded: a missing one is an error.
"""

import ctypes
import queue
import sys
import time
from pathlib import Path

import numpy as np

MODELS = {
    "large": "prompt-depth-anything-vitl-hf",
    "small": "prompt-depth-anything-vits-hf",
}
INPUT = (364, 644)  # the frame as the model sees it (height, width; multiples of 14)
OUTPUT = (90, 160)  # the depth map sent back (height, width)
PROMPT = (36, 64)  # the depth prompt (height, width)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def folder(model):
    """Where a model lives: the release's models folder, beside its runtime."""
    return Path(sys.prefix).parent / "models" / MODELS[model]


def missing(model):
    path = folder(model) / "model.safetensors"
    return None if path.is_file() else f"Prompt Depth Anything model not found: {path}"


def _lower_priority():
    if sys.platform == "win32":
        below_normal = 0x4000
        kernel = ctypes.windll.kernel32
        kernel.SetPriorityClass(kernel.GetCurrentProcess(), below_normal)
    else:
        import os

        os.nice(5)


def run(requests, results, model):
    """The process: requests (key, image (h, w, 3) uint8, prompt PROMPT metres) -> results
    ("depth", key, OUTPUT map in metres along -Z, ms); ("ready", device) once loaded, ("error",
    text) on failure (the process then ends)."""
    try:
        _lower_priority()
        import torch
        from transformers import PromptDepthAnythingForDepthEstimation

        if not torch.cuda.is_available():
            raise RuntimeError("Prompt Depth Anything needs an NVIDIA GPU with CUDA")
        torch.set_num_threads(2)
        net = PromptDepthAnythingForDepthEstimation.from_pretrained(
            folder(model), local_files_only=True, dtype=torch.float16
        )
        net = net.to("cuda").eval()
        mean = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)
        std = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)
        resize = torch.nn.functional.interpolate
        results.put(("ready", torch.cuda.get_device_name()))
        while True:
            item = requests.get()
            if item is None:
                return
            key, image, prompt = item
            started = time.perf_counter()
            with torch.inference_mode():
                x = torch.from_numpy(image).to("cuda").permute(2, 0, 1)[None].float() / 255
                x = resize(x, INPUT, mode="bilinear", antialias=True, align_corners=False)
                x = ((x - mean) / std).half()
                p = torch.from_numpy(prompt).to("cuda")[None, None].half()
                depth = net(pixel_values=x, prompt_depth=p).predicted_depth
                depth = resize(depth[:, None].float(), OUTPUT, mode="area")[0, 0]
                depth = depth.cpu().numpy()
            results.put(("depth", key, depth, (time.perf_counter() - started) * 1000))
    except Exception as exc:  # reported to the mesh worker, which goes back to guessed depth
        try:
            results.put(("error", f"Prompt Depth Anything failed: {exc}"), timeout=1)
        except queue.Full:
            pass


# -- the mesh worker's side -------------------------------------------------------------------
FINE = 4  # prompt: points per ToF zone along each axis
# A depth map pixel whose 3x3 neighbourhood spans a larger depth ratio than this is on an edge,
# which the model draws as a slope between the near and the far surface (a "curtain"): left out.
EDGE = 1.2
NEAREST = 0.05  # metres: nothing nearer is believed (prompt or map)


def prompt(zones, world_from_sensor, camera, intrinsics, size):
    """The depth prompt (PROMPT cells, metres along the camera's -Z) of a camera frame of `size`
    (w, h) at the pinhole `intrinsics` (fx, fy, cx, cy) seen from `camera` (4x4 world from
    camera): the ToF frame's surface (zones.Zones, seen from world_from_sensor), nearest first
    per cell, the cells it does not reach filled from the nearest ones it does. None where the
    ToF sees too little of the frame or only one distance (the model scales its answer between
    the prompt's nearest and farthest, so they must differ)."""
    g = FINE
    across = (np.arange(zones.cols * g) + 0.5) / g * zones.step[0] - zones.fov[0] / 2
    up = zones.fov[1] / 2 - (np.arange(zones.rows * g) + 0.5) / g * zones.step[1]
    a, u = np.meshgrid(across, up)
    rays = zones.direction(a.ravel(), u.ravel())
    ranges = zones.lookup(rays)[0]
    ok = np.isfinite(ranges)
    to_camera = np.linalg.inv(camera) @ world_from_sensor
    p = (rays[ok] * ranges[ok, None]) @ to_camera[:3, :3].T + to_camera[:3, 3]
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


class DepthMap:
    """A PDA depth map as a range source for Volume.integrate, in place of a ToF frame's
    zones.Zones: the camera is its sensor (sensor frame = camera frame, looking down -Z).
    `inside` and `outside` are how much a pixel counts, as a share of a ToF reading, within
    the ToF's field of view (zones.Zones `fov`, seen through `camera_to_sensor`, 3x3) and
    beyond it."""

    def __init__(self, depth, intrinsics, size, fov, camera_to_sensor, inside, outside):
        h, w = depth.shape
        fx, fy, cx, cy = intrinsics
        sx, sy = w / size[0], h / size[1]
        self.k = (fx * sx, fy * sy, (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5)
        padded = np.pad(depth, 1, mode="edge")
        around = np.stack([padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
                           for dy in (-1, 0, 1) for dx in (-1, 0, 1)])  # fmt: skip
        with np.errstate(divide="ignore", invalid="ignore"):
            flat = around.max(0) <= EDGE * around.min(0)
        self.depth = np.where(flat & (depth > NEAREST), depth, np.nan)
        self.count = int(np.isfinite(self.depth).sum())
        # Per pixel: inside the ToF's field of view or not, from its ray in the sensor's axes.
        rays = self._rays(*np.meshgrid(np.arange(w), np.arange(h)))
        rays = rays @ np.asarray(camera_to_sensor).T
        across = np.arctan2(rays[..., 0], -rays[..., 2])
        up = np.arcsin(np.clip(rays[..., 1] / np.linalg.norm(rays, axis=-1), -1, 1))
        seen = (rays[..., 2] < 0) & (np.abs(across) <= fov[0] / 2) & (np.abs(up) <= fov[1] / 2)
        self.weight = np.where(seen, inside, outside).astype(np.float64)

    def _rays(self, u, v):
        fx, fy, cx, cy = self.k
        return np.stack([(u - cx) / fx, -(v - cy) / fy, -np.ones_like(u, np.float64)], -1)

    def samples(self, per_zone=3):
        """Points on the map's surface (every second pixel), camera frame (N, 3)."""
        d = self.depth[::2, ::2]
        v, u = np.nonzero(np.isfinite(d))
        return self._rays(u * 2, v * 2) * d[v, u][:, None]

    def _pixels(self, points):
        fx, fy, cx, cy = self.k
        h, w = self.depth.shape
        z = -points[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.rint(cx + fx * points[:, 0] / z)
            v = np.rint(cy - fy * points[:, 1] / z)
        ok = (z > NEAREST) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        return np.where(ok, v, 0).astype(np.intp), np.where(ok, u, 0).astype(np.intp), ok, z

    def lookup(self, points):
        """As zones.Zones.lookup: range along each point's ray, its distance, whether the range
        may carve (True where the map has one), and how far the ray is free (the range)."""
        v, u, ok, z = self._pixels(points)
        distance = np.linalg.norm(points, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(ok, self.depth[v, u] * distance / z, np.nan)
        return out, distance, np.isfinite(out), out

    def weights(self, points):
        v, u, ok, _ = self._pixels(points)
        return np.where(ok, self.weight[v, u], 0.0)
