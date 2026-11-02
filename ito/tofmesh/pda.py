"""Dense monocular metric depth in a separate below-normal-priority process.

DA3Metric-Large replaces PromptDA: a sparse, narrow ToF sensor is not a dense LiDAR
prompt. Only real zone centres validate metric scale; no ToF min/max bounds or invented
prompt pixels. One outstanding frame, at the model's own rate, never a queue of stale maps.
"""

import ctypes
import queue
import sys
import time
from pathlib import Path

import numpy as np

MODEL = "DA3METRIC-LARGE"
INPUT = (364, 644)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
NEAREST = 0.05
EDGE = 1.2


def folder(model):
    return Path(sys.prefix).parent / "models" / MODEL


def missing(model):
    path = folder(model) / "model.safetensors"
    return None if path.is_file() else f"Depth Anything 3 model not found: {path}"


def _lower_priority():
    if sys.platform == "win32":
        kernel = ctypes.windll.kernel32
        kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000)
    else:
        import os

        os.nice(5)


def run(requests, results, model):
    try:
        _lower_priority()
        import torch

        from . import dense
        from .da3 import MetricDepth

        if not torch.cuda.is_available():
            raise RuntimeError("Depth Anything 3 needs an NVIDIA GPU with CUDA")
        torch.set_num_threads(2)
        net = MetricDepth(folder(model)).to("cuda").eval()
        mean = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)
        std = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)
        resize = torch.nn.functional.interpolate
        stream = torch.cuda.Stream(priority=0)  # CUDA's least-urgent priority, not high priority
        scale = None
        results.put(("ready", torch.cuda.get_device_name()))
        while True:
            item = requests.get()
            if item is None:
                return
            key, image, support, intrinsics, camera, zones, sensor, shares = item
            started = time.perf_counter()
            with torch.inference_mode(), torch.cuda.stream(stream):
                h, w = image.shape[:2]
                x = torch.from_numpy(image).to("cuda").permute(2, 0, 1)[None].float() / 255
                x = resize(x, INPUT, mode="bilinear", antialias=True, align_corners=False)
                output = net((x - mean) / std)
                # DA3Metric's training focal is 300 pixels. Use the *resized* rectified K,
                # not the original fisheye lens or the output depth map's focal length.
                focal = (intrinsics[0] * INPUT[1] / w + intrinsics[1] * INPUT[0] / h) / 2
                depth = resize(
                    output[:, None].float(), (h, w), mode="bilinear", align_corners=False
                )[0, 0] * (focal / 300.0)
                depth = depth.cpu().numpy()
            # A coherent robust scalar correction is legitimate; copying sparse ranges into
            # unobserved pixels is not. Never clamp the rest of the room to the ToF's range.
            candidate = metric_scale(depth, support)
            if candidate is not None:
                scale = candidate if scale is None else 0.9 * scale + 0.1 * candidate
            depth *= 1.0 if scale is None else scale
            depth[~np.isfinite(depth) | (depth <= NEAREST)] = np.nan
            inference_ms = (time.perf_counter() - started) * 1000
            size = (w, h)
            depth = anchor(depth, zones, sensor, intrinsics, size, shares["inside"])
            mapped = DepthMap(depth, intrinsics, size, zones.fov, sensor[:3, :3], **shares)
            direct = dense.mesh(mapped, camera, image)
            results.put(("depth", key, mapped, direct, inference_ms))
    except Exception as exc:
        try:
            results.put(("error", f"Depth Anything 3 failed: {exc}"), timeout=1)
        except queue.Full:
            pass


def metric_scale(depth, support):
    """Median scale only with >=8 agreeing real ToF centres, away from depth edges."""
    if len(support) < 8:
        return None
    h, w = depth.shape
    u, v = np.rint(support[:, :2]).astype(int).T
    ok = (u >= 1) & (u < w - 1) & (v >= 1) & (v < h - 1)
    u, v, z = u[ok], v[ok], support[ok, 2]
    patch = np.stack([depth[v + dy, u + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    good = np.isfinite(patch).all(0) & (patch.min(0) > NEAREST)
    good &= patch.max(0) <= 1.1 * patch.min(0)
    ratio = z[good] / np.median(patch[:, good], axis=0)
    if len(ratio) < 8:
        return None
    med = np.median(ratio)
    agreeing = ratio[np.abs(np.log(ratio / med)) < 0.15]
    if len(agreeing) < max(8, len(ratio) * 0.6) or not 0.5 < med < 2.0:
        return None
    return float(np.median(agreeing))


def prompt(zones, world_from_sensor, camera, intrinsics, size):
    """Only the sensor's actual valid zone centres: projected camera (u, v, z metres)."""
    rays = zones.direction(*zones.centres())[zones.valid]
    points = rays * zones.ranges[zones.valid, None]
    transform = np.linalg.inv(camera) @ world_from_sensor
    points = points @ transform[:3, :3].T + transform[:3, 3]
    z = -points[:, 2]
    points, z = points[z > NEAREST], z[z > NEAREST]
    fx, fy, cx, cy = intrinsics
    u, v = cx + fx * points[:, 0] / z, cy - fy * points[:, 1] / z
    ok = (u >= 0) & (u < size[0]) & (v >= 0) & (v < size[1])
    return np.stack([u[ok], v[ok], z[ok]], -1).astype(np.float32)


def anchor(depth, zones, sensor_from_camera, intrinsics, size, share):
    """Within real ToF support, blend the model with the measured surface.

    Solve ray/sphere intersection using the actual translated ToF mount. No ToF reading
    means no correction. This never stretches the ToF image over the camera's field of view.
    """
    h, w = depth.shape
    fx, fy, cx, cy = intrinsics
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    rays = np.stack([(u - cx) / fx, -(v - cy) / fy, -np.ones_like(u)], -1)
    a = rays.reshape(-1, 3) @ sensor_from_camera[:3, :3].astype(np.float32).T
    b = sensor_from_camera[:3, 3].astype(np.float32)
    points = a * depth.reshape(-1, 1) + b
    measured, _, _, _ = zones.lookup(points)
    aa = np.sum(a * a, axis=1)
    ab = a @ b
    disc = ab * ab - aa * (b @ b - measured * measured)
    z = (-ab + np.sqrt(np.maximum(disc, 0))) / aa
    ok = np.isfinite(measured) & (disc > 0) & (z > NEAREST)
    out = depth.copy().ravel()
    out[ok] = (z[ok] + share * out[ok]) / (1 + share)
    return out.reshape(h, w)


class DepthMap:
    """Dense depth with edge-aware one-sided surface normals. Never erase edge pixels.

    Inside=0 protects measured geometry, but still fills unmeasured holes. The volume passes
    its measured-support mask to weights(), rather than confusing an angular footprint with
    actual knowledge. Outside=0 really disables neural geometry outside the ToF footprint.
    """

    def __init__(self, depth, intrinsics, size, fov, camera_to_sensor, inside, outside):
        h, w = depth.shape
        fx, fy, cx, cy = intrinsics
        sx, sy = w / size[0], h / size[1]
        self.k = (fx * sx, fy * sy, (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5)
        self.depth = np.where(depth > NEAREST, depth, np.nan)
        self.count = int(np.isfinite(self.depth).sum())
        rays = self._rays(
            *np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        )
        self.xyz = rays * self.depth[..., None]
        tangents = []
        for axis in (1, 0):
            before, after = np.roll(self.xyz, 1, axis), np.roll(self.xyz, -1, axis)
            db, da = (
                np.abs(before[..., 2] - self.xyz[..., 2]),
                np.abs(after[..., 2] - self.xyz[..., 2]),
            )
            edge = [slice(None), slice(None)]
            edge[axis] = 0
            db[tuple(edge)] = np.inf
            edge[axis] = -1
            da[tuple(edge)] = np.inf
            tangents.append(np.where((db < da)[..., None], self.xyz - before, after - self.xyz))
        normals = np.cross(tangents[1], tangents[0])
        norm = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = normals / np.maximum(norm, 1e-9)
        # Consistent sign: normals face the camera (towards positive signed distance).
        normals *= np.where(np.sum(normals * rays, -1, keepdims=True) > 0, -1, 1)
        self.normals = normals
        sensor_rays = rays @ np.asarray(camera_to_sensor).T
        across = np.arctan2(sensor_rays[..., 0], -sensor_rays[..., 2])
        up = np.arcsin(np.clip(sensor_rays[..., 1] / np.linalg.norm(sensor_rays, axis=-1), -1, 1))
        seen = (
            (sensor_rays[..., 2] < 0) & (np.abs(across) <= fov[0] / 2) & (np.abs(up) <= fov[1] / 2)
        )
        self.inside = seen
        self.share, self.outside = inside, outside

    def _rays(self, u, v):
        fx, fy, cx, cy = self.k
        return np.stack([(u - cx) / fx, -(v - cy) / fy, -np.ones_like(u, np.float32)], -1)

    def samples(self, per_zone=3):
        return self.xyz[np.isfinite(self.depth)]

    def band(self, truncation, max_range):
        ok = np.isfinite(self.depth) & (np.linalg.norm(self.xyz, axis=-1) <= max_range)
        p, n = self.xyz[ok], self.normals[ok]
        return np.concatenate([p + t * n for t in (-truncation, 0.0, truncation)])

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
        v, u, ok, z = self._pixels(points)
        distance = np.linalg.norm(points, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(ok, self.depth[v, u] * distance / z, np.nan)
        return out, distance, np.zeros(len(points), bool), np.full(len(points), np.nan)

    def signed_distance(self, points):
        v, u, ok, _ = self._pixels(points)
        delta = points - self.xyz[v, u]
        return np.where(ok, np.sum(delta * self.normals[v, u], axis=-1), np.nan)

    def weights(self, points, measured):
        v, u, ok, _ = self._pixels(points)
        weight = np.where(self.inside[v, u], np.where(measured, self.share, 0.25), self.outside)
        return np.where(ok, weight, 0.0)
