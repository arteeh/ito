"""A depth model's maps for the ToF mesh, in a separate below-normal-priority process.

The chosen method's model (ito.recon.<method>.model.Model) turns a camera frame, with what the
ToF read when it was exposed, into a depth map. Here the map is scaled to the ToF where the
model's depth is not metric, blended with the measured surfaces (anchor), and made a dense mesh
(dense.py), all off the mesh worker. The worker sends every posed frame; the model takes the
newest whenever it is free, at its own rate, never a queue of stale maps.

Only real zone centres validate scale; no ToF min/max bounds or invented prompt pixels here (a
model whose input is a dense prompt, as Prompt Depth Anything's, makes its own from the zones).
"""

import ctypes
import importlib
import pickle
import queue
import sys
import time

import numpy as np

from ito import trace

NEAREST = 0.05


def _lower_priority():
    if sys.platform == "win32":
        kernel = ctypes.windll.kernel32
        kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000)
    else:
        import os

        os.nice(5)


class Mailbox:
    """The newest frame for the depth model, each new one in place of the last, in shared
    memory: the model takes it whenever it is free, so it neither works on a stale frame (a
    queue keeps them all, in order) nor waits for the worker to send one. put() in the worker,
    take() in the depth process."""

    def __init__(self, context, size):
        self.lock = context.Lock()
        self.wake = context.BoundedSemaphore(1)
        self.wake.acquire()
        self.data = context.RawArray("B", size)
        self.length = context.RawValue("Q", 0)
        self.sequence = context.RawValue("Q", 0)
        self.taken = 0

    def put(self, item):
        data = pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)
        with self.lock:
            memoryview(self.data).cast("B")[: len(data)] = data
            self.length.value = len(data)
            self.sequence.value += 1
        try:
            self.wake.release()
        except ValueError:  # already woken, for the one before
            pass

    def take(self):
        while True:
            self.wake.acquire()
            with self.lock:
                if self.sequence.value == self.taken:
                    continue  # woken for one it took already
                self.taken = self.sequence.value
                data = bytes(memoryview(self.data).cast("B")[: self.length.value])
            return pickle.loads(data)


def run(mailbox, results, method, model_size, tracing=None):
    """The process: the mailbox's newest (key, image, support, intrinsics, camera, zones,
    sensor_from_camera, shares) -> results ("depth", key, DepthMap, dense mesh, ms), or
    ("skipped", key, ms) for a frame the model could not map from the ToF it had (too little of
    it to prompt the model, or to scale a relative map); ("ready", device) once loaded,
    ("error", text) on failure (the process then ends, and the mesh worker reports it)."""
    from ito.recon import METHODS

    label = METHODS[method].label
    try:
        trace.start(tracing, "depth")
        _lower_priority()
        import torch

        from . import dense

        if not torch.cuda.is_available():
            raise RuntimeError("it needs an NVIDIA GPU with CUDA")
        torch.set_num_threads(2)
        model = importlib.import_module(f"ito.recon.{method}.model").Model(model_size)
        stream = torch.cuda.Stream(priority=0)  # CUDA's least-urgent priority, not high priority
        scale = None
        results.put(("ready", torch.cuda.get_device_name()))
        gpu = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        while True:
            with trace.span("depth.wait"):
                item = mailbox.take()
            key, image, support, intrinsics, camera, zones, sensor, shares = item
            k = trace.key(key)
            started = time.perf_counter()
            with trace.span("depth.predict", k) as span:
                with torch.inference_mode(), torch.cuda.stream(stream):
                    gpu[0].record(stream)
                    depth = model.predict(image, intrinsics, support, zones, sensor)
                    gpu[1].record(stream)
                if trace.enabled():
                    gpu[1].synchronize()
                    gpu_ms = gpu[0].elapsed_time(gpu[1])
                    span.fields["gpu_ms"] = gpu_ms
                    begun = span.start
                    trace.add("gpu.depth.predict", begun, begun + int(gpu_ms * 1e6), k, gpu=True)
            if depth is None:
                results.put(("skipped", key, (time.perf_counter() - started) * 1000))
                continue
            if model.metric:
                # A coherent robust scalar correction is legitimate; copying sparse ranges into
                # unobserved pixels is not. Never clamp the rest of the room to the ToF's range.
                candidate = metric_scale(depth, support)
                if candidate is not None:
                    scale = candidate if scale is None else 0.9 * scale + 0.1 * candidate
                depth *= 1.0 if scale is None else scale
            else:
                # Relative depth: every map has a scale of its own, which only the ToF can give.
                own = metric_scale(depth, support, bounded=False)
                if own is None:
                    results.put(("skipped", key, (time.perf_counter() - started) * 1000))
                    continue
                depth *= own
            depth[~np.isfinite(depth) | (depth <= NEAREST)] = np.nan
            inference_ms = (time.perf_counter() - started) * 1000
            size = depth.shape[::-1]
            with trace.span("depth.anchor", k):
                depth = anchor(depth, zones, sensor, intrinsics, size, shares["inside"])
            with trace.span("depth.normals", k):
                mapped = DepthMap(depth, intrinsics, size, zones.fov, sensor[:3, :3], **shares)
            with trace.span("depth.dense", k):
                direct = dense.mesh(mapped, camera, image)
            with trace.span("depth.send", k):
                results.put(("depth", key, mapped, direct, inference_ms))
    except Exception as exc:
        try:
            results.put(("error", f"{label} failed: {exc}"), timeout=1)
        except queue.Full:
            pass


def metric_scale(depth, support, *, bounded=True):
    """Median scale only with >=8 agreeing real ToF centres, away from depth edges; `bounded`
    (a metric map's correction) only between 0.5 and 2."""
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
    if len(agreeing) < max(8, len(ratio) * 0.6) or (bounded and not 0.5 < med < 2.0):
        return None
    return float(np.median(agreeing))


def support(zones, world_from_sensor, camera, intrinsics, size):
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

    Made in the depth process, on the GPU (a few ms; NumPy took ~35 there): its arrays are
    NumPy's once made, its tensors (`gpu`) stay for dense.mesh and are not sent on.
    """

    def __init__(self, depth, intrinsics, size, fov, camera_to_sensor, inside, outside):
        import torch

        h, w = depth.shape
        fx, fy, cx, cy = intrinsics
        sx, sy = w / size[0], h / size[1]
        self.k = (fx * sx, fy * sy, (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5)
        d = torch.from_numpy(np.ascontiguousarray(depth, np.float32)).cuda()
        d = torch.where(d > NEAREST, d, torch.nan)
        v, u = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=d.device),
            torch.arange(w, dtype=torch.float32, device=d.device),
            indexing="ij",
        )
        fx, fy, cx, cy = self.k
        rays = torch.stack([(u - cx) / fx, -(v - cy) / fy, -torch.ones_like(u)], -1)
        xyz = rays * d[..., None]
        tangents = []
        for axis in (1, 0):
            before, after = torch.roll(xyz, 1, axis), torch.roll(xyz, -1, axis)
            db = (before[..., 2] - xyz[..., 2]).abs()
            da = (after[..., 2] - xyz[..., 2]).abs()
            edge = [slice(None), slice(None)]
            edge[axis] = 0
            db[tuple(edge)] = torch.inf
            edge[axis] = -1
            da[tuple(edge)] = torch.inf
            tangents.append(torch.where((db < da)[..., None], xyz - before, after - xyz))
        normals = torch.linalg.cross(tangents[1], tangents[0], dim=-1)
        norm = torch.linalg.norm(normals, dim=-1, keepdim=True)
        normals = normals / torch.clamp_min(norm, 1e-9)
        # Consistent sign: normals face the camera (towards positive signed distance).
        normals = normals * torch.where((normals * rays).sum(-1, keepdim=True) > 0, -1.0, 1.0)
        to_sensor = torch.as_tensor(np.asarray(camera_to_sensor), dtype=torch.float64)
        sensor_rays = rays.double() @ to_sensor.to(d.device).T
        across = torch.atan2(sensor_rays[..., 0], -sensor_rays[..., 2])
        up = torch.asin(
            torch.clamp(sensor_rays[..., 1] / torch.linalg.norm(sensor_rays, dim=-1), -1, 1)
        )
        seen = (sensor_rays[..., 2] < 0) & (across.abs() <= fov[0] / 2) & (up.abs() <= fov[1] / 2)
        self.gpu = d, xyz, normals, seen
        self.depth, self.xyz, self.normals, self.inside = (t.cpu().numpy() for t in self.gpu)
        self.count = int(np.isfinite(self.depth).sum())
        self.share, self.outside = inside, outside

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "gpu"}

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
