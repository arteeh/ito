"""Persistent surfaces in the connection's fixed odometry frame.

Estimator nodes are observation provenance, never owners of surface lifetime. Anchors
are frozen when admitted; forgetting or correcting an estimator node cannot move a room.
"""

from dataclasses import dataclass
from importlib.util import find_spec

import numpy as np
from scipy.ndimage import distance_transform_edt, median_filter
from scipy.spatial.transform import Rotation

from .slots import KEEP, GaussianSlots

C0 = 0.28209479177387814
MAX_VIEWS = 24


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
    exposure: np.ndarray | None = None
    features: tuple | None = None
    estimated_pose: np.ndarray | None = None
    estimated_map_pose: np.ndarray | None = None
    fitting_pose: np.ndarray | None = None
    refitting: bool = False


class GaussianBackend(GaussianSlots):
    def __init__(self, *args, precision="full", **options):
        super().__init__(*args, **options)
        self.precision = precision
        self.views = []
        self.anchors = {}
        self.next_anchor = 0
        self.fit_steps = 0
        self.next_key = 0
        self.fitter = None
        self.refits = 0
        self.pending_fit = None
        self.fusion_cursor = 0
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
                view.refitting = True
                self.refits += 1
            changed = True
        if changed:
            self.recover()

    def fit(self):
        if not self.can_fit or not self.views:
            return
        if self.fitter is None:
            from .fitting import LocalFit

            self.fitter = LocalFit(self)
        self.fitter.step()
        if self.pending_fit is not None and self.fitter.active is not None:
            self.fitter.step()

    def recover(self):
        if self.fitter is not None:
            self.fitter.active = None

    def victims(self, candidates, count):
        # Redundant/unsupported records go before useful old surfaces, including when
        # the redundant record was just admitted. Age breaks ties only in useful data.
        useless = (
            (self.contribution[candidates] < 0.25)
            | ((self.support[candidates] < 2) & (self.contradictions[candidates] > 0))
            | ((self.support[candidates] < 3) & (self.uncertainty[candidates] > 0.2))
            | (self.records[candidates, 0, 3] < 0.01)
        )
        weak = candidates[useless]
        order = np.lexsort((self.support[weak], -self.uncertainty[weak], self.contribution[weak]))
        first = weak[order[:count]]
        return np.concatenate((first, self.oldest(candidates[~useless], count - len(first))))

    def integrate(self, rgb, depth, camera, now):
        i = self.intrinsics
        y, x = np.mgrid[: i.height, : i.width]
        points = np.stack(((x - i.cx) / i.fx * depth, -(y - i.cy) / i.fy * depth, -depth), -1)
        self.observe(
            points, rgb, np.isfinite(depth) & (depth > 0), camera, (i.fx, i.fy, i.cx, i.cy), now
        )

    def project(self, camera, intrinsics, shape):
        fx, fy, cx, cy = intrinsics
        height, width = shape
        live = np.flatnonzero((self.keys >= 0) & ~self.retiring)
        local = (self.records[live, 0, :3] - camera[:3, 3]) @ camera[:3, :3]
        z = -local[:, 2]
        safe = np.maximum(z, 0.01)
        u = np.rint(cx + fx * local[:, 0] / safe).astype(int)
        v = np.rint(cy - fy * local[:, 1] / safe).astype(int)
        inside = (z > 0.01) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        live, z, u, v = live[inside], z[inside], u[inside], v[inside]
        local = local[inside]
        pixels = v * width + u
        nearest_depth = np.full(height * width, np.inf, z.dtype)
        np.minimum.at(nearest_depth, pixels, z)
        winners = np.flatnonzero(z == nearest_depth[pixels])
        surface = np.full(height * width, -1, np.int64)
        surface[pixels[winners]] = live[winners]
        surface = surface.reshape(height, width)
        if len(winners):
            records = self.records[live[winners]]
            centers = local[winners]
            depths = z[winners]
            jacobian = (
                np.stack(
                    (
                        (camera[:3, 0] + centers[:, 0:1] / depths[:, None] * camera[:3, 2]) * fx,
                        (camera[:3, 1] + centers[:, 1:2] / depths[:, None] * camera[:3, 2]) * fy,
                    ),
                    1,
                )
                / depths[:, None, None]
            )
            rotations = Rotation.from_quat(records[:, 2, [1, 2, 3, 0]]).as_matrix()
            projected = (jacobian @ rotations) * records[:, None, 1, :3]
            covariance = projected @ projected.transpose(0, 2, 1)
            a, b, c = covariance[:, 0, 0], covariance[:, 0, 1], covariance[:, 1, 1]
            minor = np.maximum(0, (a + c - np.sqrt((a - c) ** 2 + 4 * b * b)) / 2)
            radius = np.zeros(height * width, np.float32)
            radius[pixels[winners]] = np.clip(1.5 * np.sqrt(minor), 1.5, 3)
            radius = radius.reshape(height, width)
            distance, nearest = distance_transform_edt(surface < 0, return_indices=True)
            surface = surface[tuple(nearest)]
            surface[distance > radius[tuple(nearest)]] = -1
        return surface, live, z, u, v

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
        confidence=None,
        registered=True,
    ):
        points = np.asarray(points, np.float32)
        height, width = image.shape[:2]
        points = points.reshape(height, width, 3)
        valid = (
            valid.reshape(height, width) & np.isfinite(points).all(-1) & (points[..., 2] < -0.01)
        )
        depth = np.where(valid, -points[..., 2], 0)
        surface, live, z, u, v = self.project(camera, intrinsics, depth.shape)
        if len(live):
            observed = depth[v, u]
            tolerance = np.maximum(0.05 * z, 2 * self.records[live, 1, :3].max(-1))
            depth_agreement = (observed > 0) & (np.abs(observed - z) <= tolerance)
            agreement = depth_agreement.copy()
            peers = surface[v, u]
            agreement &= peers == live
            color_error = np.abs(
                self.records[live, 3, :3] * C0 + 0.5 - image[v, u].astype(np.float32) / 255
            ).max(-1)
            agreement &= color_error < 0.15
            supported = live[agreement]
            self.support[supported] += 1
            self.seen[supported] = now
            self.uncertainty[supported] = 0.9 * self.uncertainty[supported] + 0.1 * (
                np.maximum(
                    np.abs(observed[agreement] - z[agreement]) / np.maximum(z[agreement], 0.01),
                    color_error[agreement],
                )
            )
            self.contradictions[live[depth_agreement]] = 0
            contradicted = live[
                (observed > z + tolerance * np.where(self.support[live] >= 20, 2, 1))
                & (observed > 0)
            ]
            self.contradictions[contradicted] = np.minimum(
                self.contradictions[contradicted].astype(int) + 1, 255
            )
            gone = contradicted[self.contradictions[contradicted] >= 4]
            if registered:
                exposed = depth_agreement & (peers == live)
                visible_ids = live[exposed]
                self.uncertainty[visible_ids] = (
                    0.9 * self.uncertainty[visible_ids] + 0.1 * color_error[exposed]
                )
                wrong = live[exposed & (color_error > 0.25)]
                self.photo_contradictions[wrong] = np.minimum(
                    self.photo_contradictions[wrong].astype(int) + 1, 255
                )
                self.photo_contradictions[live[exposed & (color_error < 0.15)]] = 0
                rejected = live[
                    self.photo_contradictions[live] >= np.where(self.support[live] >= 20, 8, 4)
                ]
                self.contribution[rejected] = 0
                gone = np.union1d(gone, rejected)
            self.captures[gone] = self.source_captured
            self.retire(gone)
            winners, counts = np.unique(surface[surface >= 0], return_counts=True)
            # Keep evidence from other views: an occluded background is useful on
            # return even when its center loses today's depth buffer.
            self.contribution[winners] = np.maximum(self.contribution[winners], counts)
            # Contributions guide pressure pruning. Two centers sharing a pixel are
            # not proof of redundancy: their Gaussian footprints can cover different edges.
            own, peer = self.records[live], self.records[peers]
            peer_local = (peer[:, 0, :3] - camera[:3, 3]) @ camera[:3, :3]
            peer_z = np.maximum(-peer_local[:, 2], 0.01)
            own_local = (own[:, 0, :3] - camera[:3, 3]) @ camera[:3, :3]
            fx, fy, _, _ = intrinsics
            separation = own_local[:, :2] / z[:, None] - peer_local[:, :2] / peer_z[:, None]
            separation *= [fx, fy]
            redundant = (peers != live) & (peers >= 0)
            redundant &= np.linalg.norm(separation, axis=1) < 0.5
            redundant &= (np.abs(z - peer_z) < 0.05 * z) & depth_agreement
            redundant &= np.abs(own[:, 3, :3] - peer[:, 3, :3]).max(-1) * C0 < 0.02
            redundant &= (peer[:, 0, 3] > 0.9) & (own[:, 0, 3] > 0.9)
            redundant &= (
                peer[:, 1, :2] / peer_z[:, None] >= 0.95 * own[:, 1, :2] / z[:, None]
            ).all(-1)
            duplicates = live[redundant]
            self.redundancy[duplicates] = np.minimum(
                self.redundancy[duplicates].astype(int) + 1, 255
            )
            self.redundancy[live[peers == live]] = 0
            duplicates = live[self.redundancy[live] >= 4]
            self.contribution[duplicates] = 0
            self.captures[duplicates] = self.source_captured
            self.retire(duplicates)
            surface[np.isin(surface, gone)] = -1
            # Fuse only the frontmost matched surface. A bounded radial correction
            # refines depth without sliding a sample across the current image. Colour
            # updates fit the observed radiance; occluded background keeps its evidence.
            visible = (peers == live) & (observed > 0) & ~self.retiring[live]
            visible &= np.abs(observed - z) <= 3 * tolerance
            visible &= color_error < 0.15
            selected = np.flatnonzero(visible)
            if len(selected) > 4096:
                selected = selected[(np.arange(4096) + self.fusion_cursor) % len(selected)]
                self.fusion_cursor += len(selected)
            matched = live[selected]
            if len(matched):
                fx, fy, cx, cy = intrinsics
                local = (self.records[matched, 0, :3] - camera[:3, 3]) @ camera[:3, :3]
                mz = -local[:, 2]
                pu = cx + fx * local[:, 0] / mz
                pv = cy - fy * local[:, 1] / mz
                from scipy.ndimage import map_coordinates

                rgb = (
                    np.stack(
                        [
                            map_coordinates(
                                image[..., channel].astype(np.float32),
                                [pv, pu],
                                order=1,
                                mode="nearest",
                                prefilter=False,
                            )
                            for channel in range(3)
                        ],
                        -1,
                    )
                    / 255
                )
                weight = 1 / np.minimum(self.support[matched] + 1, 12)
                self.records[matched, 3, :3] += weight[:, None] * (
                    (rgb - 0.5) / C0 - self.records[matched, 3, :3]
                )
                offset = np.clip((observed[selected] - mz) * 0.05, -mz / fx * 0.1, mz / fx * 0.1)
                ray = self.records[matched, 0, :3] - camera[:3, 3]
                self.records[matched, 0, :3] += ray * (offset / mz)[:, None]
                self.dirty[matched] = True
                self.captures[matched] = self.source_captured
                self.sequences[matched] = self.source_sequence
                # Cancel a private fit whose base records have just received evidence.
                self.recover()
        covered = surface >= 0
        mapped_depth = np.zeros_like(depth)
        centers = self.records[surface[covered], 0, :3] - camera[:3, 3]
        mapped_depth[covered] = -(centers @ camera[:3, 2])
        missing = valid & ((surface < 0) | (depth < 0.8 * mapped_depth))
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
        slots = free[:count]
        if count:
            anchor = self.next_anchor
            self.next_anchor += 1
            self.anchors[anchor] = camera.copy()
            x, y = xx.ravel()[seeds], yy.ravel()[seeds]
            p = points[y, x]
            fx, fy, _, _ = intrinsics
            d = -p[:, 2]
            # Learned depth has pixel noise; differentiating it directly makes long,
            # nearly edge-on splats. Filter only the tangent estimate, not the centers.
            filtered = median_filter(depth, size=5)
            tangent_points = points * (filtered / np.maximum(depth, 1e-6))[..., None]
            dx = (
                tangent_points[y, np.minimum(x + 2, width - 1)]
                - tangent_points[y, np.maximum(x - 2, 0)]
            ) / 4
            dy = (
                tangent_points[np.minimum(y + 2, height - 1), x]
                - tangent_points[np.maximum(y - 2, 0), x]
            ) / 4
            smooth = np.isfinite(dx).all(-1) & np.isfinite(dy).all(-1)
            smooth &= (np.linalg.norm(dx, axis=1) < d * 0.04) & (
                np.linalg.norm(dy, axis=1) < d * 0.04
            )
            dx[~smooth] = np.stack(
                (d[~smooth] / fx, np.zeros((~smooth).sum()), np.zeros((~smooth).sum())), -1
            )
            dy[~smooth] = np.stack(
                (np.zeros((~smooth).sum()), -d[~smooth] / fy, np.zeros((~smooth).sum())), -1
            )
            axis = dx / np.maximum(np.linalg.norm(dx, axis=1, keepdims=True), 1e-8)
            normal = np.cross(dx, dy)
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)
            normal = np.where(normal[:, 2:3] < 0, -normal, normal)
            tangent = np.cross(normal, axis)
            basis = camera[:3, :3] @ np.stack((axis, tangent, normal), -1)
            degenerate = np.abs(np.linalg.det(basis)) < 0.5
            basis[degenerate] = camera[:3, :3]
            quat = Rotation.from_matrix(basis).as_quat()[:, [3, 0, 1, 2]]
            spacing = np.where(edge[y, x], 0.7, 1.8)
            scales = np.stack(
                (
                    np.linalg.norm(dx, axis=1) * spacing,
                    np.abs((dy * tangent).sum(-1)) * spacing,
                    d / fx * 0.4,
                ),
                -1,
            )
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
            self.uncertainty[slots] = (
                0.1 if confidence is None else (1 - np.asarray(confidence).reshape(-1)[seeds])
            )
            self.contribution[slots] = 1
            self.contradictions[slots] = 0
            self.redundancy[slots] = 0
            self.photo_contradictions[slots] = 0
            self.seen[slots] = now
            self.captures[slots] = self.source_captured
            self.sequences[slots] = self.source_sequence
            self.dirty[slots] = True
        if not self.views or self.source_captured - self.views[-1].captured >= 0.5:
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
        owners = set(self.owners[self.keys >= 0])
        self.anchors = {k: v for k, v in self.anchors.items() if k in owners}
