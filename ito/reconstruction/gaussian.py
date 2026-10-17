"""Persistent surfaces in the connection's fixed odometry frame.

Estimator nodes are observation provenance, never owners of surface lifetime. Anchors
are frozen when admitted; forgetting or correcting an estimator node cannot move a room.

A new observation only adds Gaussians where the map, rendered from that view, does not
already explain it: an uncovered pixel, or a surface clearly in front of the map's. Map
Gaussians the view sees through gather contradictions and retire after repeated
evidence; Gaussians at the observed surface gather support. A revisit therefore refines
the room it finds instead of laying a second one over it.
"""

import os
import threading
from dataclasses import dataclass
from importlib.util import find_spec

import numpy as np
from scipy.ndimage import (
    binary_opening,
    distance_transform_edt,
    maximum_filter,
    median_filter,
    minimum_filter,
)
from scipy.spatial.transform import Rotation

from .slots import KEEP, GaussianSlots

C0 = 0.28209479177387814
MAX_VIEWS = 24
DEPTH_TOLERANCE = 0.08  # Of depth: two estimates closer than this are one surface.
COVERED = 0.85  # Rendered opacity above which a pixel is explained by the map.
BLEND = 0.2  # Colour difference below which a view's light refines a surface's colour.
BLENDED = 8192  # At most this many colour refinements per observation.
MAX_ELONGATION = 10.0  # Longest seed axis over its pixel footprint (grazing floors).
EDGE = 1.08  # Depth ratio to both neighbours that marks a pixel as between two surfaces.
COARSE = 4.5  # Pixels: a mapped Gaussian this large is refined by the view (seeds are ~2).
AXES = np.diag([1.0, -1.0, -1.0, 1.0])  # Ito camera (-Z forward, +Y up) to OpenCV.
# ITO_FIT=view trains whole retained views (training.py) instead of patches (fitting.py).
FIT = os.environ.get("ITO_FIT", "local")


@dataclass
class Observation:
    node: int
    pose: np.ndarray
    image: np.ndarray
    depth: np.ndarray
    intrinsics: tuple
    captured: float
    sequence: int
    steps: int = 0
    features: tuple | None = None
    estimated_pose: np.ndarray | None = None
    estimated_map_pose: np.ndarray | None = None
    fitting_pose: np.ndarray | None = None
    refitting: bool = False
    tile_steps: np.ndarray | None = None
    # The camera's own frame at a finer grid than image, for photometric training only.
    photo: np.ndarray | None = None
    photo_intrinsics: tuple | None = None
    # The network's own depth before views agreed on it: later views may yet confirm it.
    raw_depth: np.ndarray | None = None
    filled: int = 0  # retained views there were when training last filled this view's holes


class GaussianBackend(GaussianSlots):
    def __init__(self, *args, precision="full", **options):
        super().__init__(*args, **options)
        self.precision = precision
        # Held by whatever changes the map; a trainer thread takes it between its steps.
        self.lock = threading.RLock()
        self.views = []
        self.anchors = {}
        self.next_anchor = 0
        self.fit_steps = 0
        self.next_key = 0
        self.fitter = None
        self.refits = 0
        self.pending_fit = None
        self.can_fit = (
            self.xp is not np and find_spec("torch") is not None and find_spec("gsplat") is not None
        )
        if self.can_fit:
            import torch  # Register Torch's allocator before the live VRAM manager is created.

            self.can_fit = torch.cuda.is_available()

    def revise(self, poses):
        registration = getattr(self, "registration", None)
        if registration is not None:
            registration.revise(poses, self.views)
        changed = False
        for view in self.views:
            pose = poses.get(view.node)
            if pose is None:
                continue
            mapped = pose if registration is None else registration.map_from_estimator @ pose
            previous = view.pose if view.estimated_map_pose is None else view.estimated_map_pose
            delta = mapped @ np.linalg.inv(previous)
            angle = np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1))
            if angle < 1e-5 and np.linalg.norm(delta[:3, 3]) < 1e-5:
                continue
            view.estimated_pose = pose.copy()
            view.estimated_map_pose = mapped.copy()
            # Gauge changes cancel in the persistent frame. Relative corrections
            # change the fitting camera, never the published surface or its anchor.
            fitting = view.pose if view.fitting_pose is None else view.fitting_pose
            view.fitting_pose = delta @ fitting
            if angle * view.intrinsics[0] > 2 or np.linalg.norm(delta[:3, 3]) > 0.01:
                view.steps = 0
                view.tile_steps = None
                view.refitting = True
                self.refits += 1
            changed = True
        if changed:
            self.recover()

    def fit(self):
        """One bounded training step; False when there was nothing to train."""
        if not self.can_fit or not self.views:
            return False
        if self.fitter is None:
            if FIT == "view":
                from .training import ViewFit

                self.fitter = ViewFit(self)
            else:
                from .fitting import LocalFit

                self.fitter = LocalFit(self)
        return self.fitter.step()

    def start_training(self, available, stopped, failed):
        """With ITO_FIT=view, train on a thread of its own beside tracking; the thread, or
        None when training stays inline (fit())."""
        if FIT != "view" or not self.can_fit or os.environ.get("ITO_FIT_THREAD", "1") == "0":
            return None
        from .training import ViewFit

        self.fitter = ViewFit(self)
        return self.fitter.start(available, stopped, failed)

    def recover(self):
        if self.fitter is not None:
            self.fitter.active = None

    def victims(self, candidates, count):
        # Surfaces no view has shown in front, or that the views keep contradicting, go
        # before useful old ones, however new they are. Age orders only useful data.
        useless = (
            ((self.contribution[candidates] < 2) & (self.support[candidates] < 2))
            | (self.contradictions[candidates] > self.support[candidates])
            | (self.records[candidates, 0, 3] < 0.02)
        )
        weak = candidates[useless]
        order = np.lexsort((self.seen[weak], self.support[weak]))
        first = weak[order[:count]]
        return np.concatenate((first, self.oldest(candidates[~useless], count - len(first))))

    def integrate(self, rgb, depth, camera, now):
        i = self.intrinsics
        y, x = np.mgrid[: i.height, : i.width]
        points = np.stack(((x - i.cx) / i.fx * depth, -(y - i.cy) / i.fy * depth, -depth), -1)
        self.observe(
            points, rgb, np.isfinite(depth) & (depth > 0), camera, (i.fx, i.fy, i.cx, i.cy), now
        )

    def centers(self, camera, intrinsics, shape):
        """Live Gaussians whose footprint reaches this view: ids, depth, pixel."""
        fx, fy, cx, cy = intrinsics
        height, width = shape
        live = np.flatnonzero((self.keys >= 0) & ~self.retiring)
        local = (self.records[live, 0, :3] - camera[:3, 3]) @ camera[:3, :3]
        z = -local[:, 2]
        safe = np.maximum(z, 0.01)
        u = cx + fx * local[:, 0] / safe
        v = cy - fy * local[:, 1] / safe
        reach = 3 * fx * self.records[live, 1, :3].max(-1) / safe
        inside = (z > 0.01) & (u > -reach) & (u < width + reach)
        inside &= (v > -reach) & (v < height + reach)
        return live[inside], z[inside], u[inside], v[inside]

    def render(self, camera, intrinsics, shape, ids, z, u, v):
        """The map's opacity, depth and Gaussian width (shorter axis) in this view's pixels."""
        height, width = shape
        size = intrinsics[0] * self.records[ids, 1, :2].min(-1) / z
        if self.can_fit and len(ids):
            import torch
            from gsplat import rasterization

            fx, fy, cx, cy = intrinsics
            with self.gpu_chunk("coverage"), torch.no_grad():
                records = torch.as_tensor(self.records[ids], device="cuda")
                view = torch.as_tensor(np.linalg.inv(camera @ AXES), device="cuda")
                calibration = torch.tensor(
                    [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device="cuda", dtype=torch.float32
                )
                image, alpha, _ = rasterization(
                    means=records[:, 0, :3],
                    quats=records[:, 2],
                    scales=records[:, 1, :3],
                    opacities=records[:, 0, 3],
                    colors=torch.as_tensor(size, device="cuda", dtype=torch.float32)[:, None],
                    viewmats=view.float()[None],
                    Ks=calibration[None],
                    width=width,
                    height=height,
                    near_plane=0.01,
                    far_plane=1000,
                    render_mode="RGB+ED",
                )
                alpha = alpha[0, ..., 0].cpu().numpy()
                image = image[0].cpu().numpy()
            seen = alpha > 0.01
            pixels = np.where(seen, image[..., 0] / np.maximum(alpha, 0.01), 0)
            return alpha, np.where(seen, image[..., 1], 0), pixels
        if not len(ids):
            return (np.zeros(shape, np.float32),) * 3
        # Without CUDA: nearest centre per pixel, spread over its projected footprint.
        u, v = np.rint(u).astype(int), np.rint(v).astype(int)
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        ids, z, u, v, size = ids[inside], z[inside], u[inside], v[inside], size[inside]
        pixels = v * width + u
        nearest = np.full(height * width, np.inf, np.float32)
        np.minimum.at(nearest, pixels, z)
        front = z == nearest[pixels]
        pixels, size = pixels[front], size[front]
        sizes = np.zeros(height * width, np.float32)
        sizes[pixels] = size
        nearest, sizes = nearest.reshape(shape), sizes.reshape(shape)
        distance, index = distance_transform_edt(~np.isfinite(nearest), return_indices=True)
        index = tuple(index)
        covered = distance <= np.clip(sizes[index], 1, 3)
        return (
            covered.astype(np.float32),
            np.where(covered, nearest[index], 0),
            np.where(covered, sizes[index], 0),
        )

    def observe(
        self,
        points,
        image,
        valid,
        camera,
        intrinsics,
        now,
        *,
        node=None,
        registered=True,
        photo=None,
        raw_depth=None,
    ):
        """registered: the camera pose was aligned to retained views (or is given).
        photo: (the camera's frame at a finer grid, its intrinsics) for training."""
        points = np.asarray(points, np.float32)
        height, width = image.shape[:2]
        points = points.reshape(height, width, 3)
        valid = (
            valid.reshape(height, width) & np.isfinite(points).all(-1) & (points[..., 2] < -0.01)
        )
        depth = np.where(valid, -points[..., 2], 0)
        ids, z, u, v = self.centers(camera, intrinsics, depth.shape)
        alpha, mapped, size = self.render(camera, intrinsics, depth.shape, ids, z, u, v)
        closest = minimum_filter(np.where(valid, depth, np.inf), size=3)
        # The same surface drawn with Gaussians much larger than this view's pixels: a
        # closer view replaces them with finer ones. Resolution only ever increases.
        coarse = (alpha >= COVERED) & (size > COARSE)
        coarse &= valid & (np.abs(depth - mapped) <= DEPTH_TOLERANCE * mapped)
        if len(ids):
            pu, pv = np.rint(u).astype(int), np.rint(v).astype(int)
            on = (pu >= 0) & (pu < width) & (pv >= 0) & (pv < height)
            ids, z, pu, pv = ids[on], z[on], pu[on], pv[on]
            tolerance = DEPTH_TOLERANCE * z + 2 * self.records[ids, 1, :3].max(-1)
            observed = depth[pv, pu]
            front = mapped[pv, pu]
            # In front of or at the map's rendered surface: this view shows it.
            shown = (front <= 0) | (z <= front + tolerance)
            agree = shown & (observed > 0) & (np.abs(observed - z) <= tolerance)
            # Seen through only if no pixel around its centre sees it: a silhouette a
            # pixel off is not free space.
            nearest = closest[pv, pu]
            through = np.isfinite(nearest) & (nearest > z + tolerance)
            supported = ids[agree]
            self.support[supported] = np.minimum(self.support[supported] + 1, 1000)
            self.seen[supported] = now
            self.contradictions[supported] = 0
            error = np.abs(observed[agree] - z[agree]) / z[agree]
            self.uncertainty[supported] += 0.1 * (error - self.uncertainty[supported])
            self.contribution[ids[shown & (observed > 0)]] += 1
            # Light differs between views (exposure, a lamp on the camera): average it
            # into surfaces that already match, never across texture a view misplaces.
            seen = image[pv[agree], pu[agree]].astype(np.float32) / 255
            color = self.records[supported, 3, :3] * C0 + 0.5
            change = np.abs(seen - color).max(-1)
            near = np.flatnonzero((change < BLEND) & (change > 0.01))
            near = near[np.argsort(-change[near])[:BLENDED]]
            blended = supported[near]
            weight = 1 / np.minimum(self.support[blended], 10)[:, None]
            color = color[near] + weight * (seen[near] - color[near])
            self.records[blended, 3, :3] = (color - 0.5) / C0
            self.dirty[blended] = True
            self.captures[blended] = self.source_captured
            self.sequences[blended] = self.source_sequence
            contradicted = ids[through]
            self.contradictions[contradicted] = np.minimum(
                self.contradictions[contradicted].astype(int) + 1, 255
            )
            limit = np.where(self.support[contradicted] >= 10, 6, 3)
            gone = contradicted[self.contradictions[contradicted] >= limit]
            own = intrinsics[0] * self.records[ids, 1, :2].min(-1) / z
            # Only a Gaussian whose whole footprint this view refines; a larger one
            # would leave a hole where it reached beyond the refined pixels.
            if coarse.any():
                inner = distance_transform_edt(coarse)[pv, pu] > 2 * own
                gone = np.union1d(gone, ids[agree & inner & (own > COARSE)])
            if len(gone):
                self.captures[gone] = self.source_captured
                self.retire(gone)
        tolerance = DEPTH_TOLERANCE * mapped
        ahead = valid & (alpha >= COVERED) & (depth < mapped - tolerance)
        # A sliver beside a silhouette is misalignment, not a new surface in front. An
        # observation not aligned to the map may place a known object beside itself:
        # it fills holes but lays nothing over mapped surfaces.
        ahead = binary_opening(ahead, np.ones((3, 3), bool)) & registered
        missing = valid & ((alpha < COVERED) | ahead | coarse)
        # Learned depth ramps across a silhouette; a pixel between its neighbours'
        # nearest and farthest surface belongs to neither and would seed a streak.
        farthest = maximum_filter(depth, size=3)
        missing &= ~((depth > closest * EDGE) & (depth * EDGE < farthest))
        self.seed(points, image, depth, missing, camera, intrinsics, now)
        self.keep_view(camera, image, depth, intrinsics, node, photo, raw_depth)

    def seed(self, points, image, depth, missing, camera, intrinsics, now):
        height, width = depth.shape
        # A three-pixel lattice has a calibrated footprint; colour edges keep one-pixel
        # samples. Surface tangents follow depth instead of making camera-facing discs.
        edge = np.zeros(depth.shape, bool)
        color = image.astype(np.float32) / 255
        edge[:, 1:] |= np.max(np.abs(color[:, 1:] - color[:, :-1]), -1) > 0.08
        edge[1:] |= np.max(np.abs(color[1:] - color[:-1]), -1) > 0.08
        yy, xx = np.mgrid[:height, :width]
        seeds = np.flatnonzero(missing & (edge | ((xx % 3 == 0) & (yy % 3 == 0))))
        needed = self.count + len(seeds)
        if len(self.keys) < min(needed, self.budget):
            self.grow(min(self.budget, max(needed, 2 * len(self.keys))))
        free = np.flatnonzero(self.keys[: self.budget] < 0)
        count = min(len(seeds), len(free), max(0, self.budget - self.count))
        if count < len(seeds):
            candidates = np.flatnonzero((self.keys >= 0) & ~self.retiring)
            self.retire(self.victims(candidates, len(seeds) - count), pressure=True)
            seeds = seeds[np.linspace(0, len(seeds) - 1, count, dtype=int)]
        if not count:
            return
        slots = free[:count]
        anchor = self.next_anchor
        self.next_anchor += 1
        self.anchors[anchor] = camera.copy()
        x, y = xx.ravel()[seeds], yy.ravel()[seeds]
        p = points[y, x]
        fx, fy, _, _ = intrinsics
        d = -p[:, 2]
        # Learned depth has pixel noise; differentiating it directly makes long,
        # nearly edge-on splats. Filter only the tangent estimate, not the centers.
        filtered = median_filter(depth, size=3)
        tangent_points = points * (filtered / np.maximum(depth, 1e-6))[..., None]

        def tangent(dx, dy):
            # The shorter one-sided difference: never across a depth edge.
            here = tangent_points[y, x]
            after = tangent_points[
                np.minimum(y + 2 * dy, height - 1), np.minimum(x + 2 * dx, width - 1)
            ]
            before = tangent_points[np.maximum(y - 2 * dy, 0), np.maximum(x - 2 * dx, 0)]
            forward, backward = (after - here) / 2, (here - before) / 2
            shorter = np.linalg.norm(forward, axis=1) < np.linalg.norm(backward, axis=1)
            return np.where(shorter[:, None], forward, backward)

        dx, dy = tangent(1, 0), tangent(0, 1)
        footprint = d / fx
        smooth = np.isfinite(dx).all(-1) & np.isfinite(dy).all(-1)
        smooth &= np.linalg.norm(dx, axis=1) < MAX_ELONGATION * footprint
        smooth &= np.linalg.norm(dy, axis=1) < MAX_ELONGATION * footprint
        flat = ~smooth
        dx[flat] = np.stack((footprint[flat], 0 * d[flat], 0 * d[flat]), -1)
        dy[flat] = np.stack((0 * d[flat], -d[flat] / fy, 0 * d[flat]), -1)
        axis = dx / np.maximum(np.linalg.norm(dx, axis=1, keepdims=True), 1e-8)
        normal = np.cross(dx, dy)
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)
        normal = np.where(normal[:, 2:3] < 0, -normal, normal)
        side = np.cross(normal, axis)
        basis = camera[:3, :3] @ np.stack((axis, side, normal), -1)
        degenerate = np.abs(np.linalg.det(basis)) < 0.5
        basis[degenerate] = camera[:3, :3]
        quat = Rotation.from_matrix(basis).as_quat()[:, [3, 0, 1, 2]]
        spacing = np.where(edge[y, x], 0.7, 2.1)
        scales = np.stack(
            (
                np.linalg.norm(dx, axis=1) * spacing,
                np.abs((dy * side).sum(-1)) * spacing,
                footprint * 0.4,
            ),
            -1,
        )
        scales[:, :2] = np.maximum(scales[:, :2], footprint[:, None] * 0.5)
        records = np.zeros((count, 4, 4), np.float32)
        records[:, 0, :3] = p @ camera[:3, :3].T + camera[:3, 3]
        records[:, 0, 3] = 0.995
        records[:, 1, :3] = np.maximum(scales, 0.0001)
        records[:, 1, 3] = KEEP
        records[:, 2] = quat
        records[:, 3, :3] = (color[y, x] - 0.5) / C0
        self.records[slots] = records
        self.keys[slots] = np.arange(self.next_key, self.next_key + count)
        self.next_key += count
        self.owners[slots] = anchor
        self.support[slots] = 1
        self.uncertainty[slots] = 0.1
        self.contribution[slots] = 1
        self.contradictions[slots] = 0
        self.seen[slots] = now
        self.captures[slots] = self.source_captured
        self.sequences[slots] = self.source_sequence
        self.dirty[slots] = True
        owners = set(self.owners[self.keys >= 0])
        self.anchors = {k: v for k, v in self.anchors.items() if k in owners}

    def keep_view(self, camera, image, depth, intrinsics, node, photo=None, raw_depth=None):
        if self.views and self.source_captured - self.views[-1].captured < 0.5:
            return
        self.views.append(
            Observation(
                self.source_sequence if node is None else node,
                camera.copy(),
                image.copy(),
                depth.copy(),
                intrinsics,
                self.source_captured,
                self.source_sequence,
                estimated_pose=getattr(self, "observation_estimate", camera).copy(),
                estimated_map_pose=getattr(self, "observation_map_estimate", camera).copy(),
                fitting_pose=camera.copy(),
                photo=None if photo is None else photo[0],
                photo_intrinsics=None if photo is None else photo[1],
                raw_depth=None if raw_depth is None else raw_depth.copy(),
            )
        )
        if len(self.views) > MAX_VIEWS:
            # Keep three recent observations plus a diverse archive. Among two
            # redundant archived views retain the one with more fitting evidence;
            # ties keep the older anchor instead of chaining through recent noise.
            archived = self.views[:-3]
            poses = np.array([v.pose for v in archived])
            centers = poses[:, :3, 3]
            distance = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
            directions = poses[:, :3, 2]
            distance += 1 - np.clip(directions @ directions.T, -1, 1)
            distance[np.tril_indices(len(poses))] = np.inf
            a, b = np.unravel_index(np.argmin(distance), distance.shape)
            discard = a if archived[a].steps < archived[b].steps else b
            del self.views[discard]
