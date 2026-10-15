"""Online gsplat mapper for the oracle intervention; never used to align camera poses.

Retains calibrated RGB views, fits center/quaternion/three scales/opacity/SH0 and
per-image exposure with Adam, and densifies from image gradients. Source views move
with bounded map-side corrections to their supplied anchor. Native gsplat/Adam
parameters stay fp32 in both
settings; Half uses bf16/fp16 image loss and Full uses fp32, without an OOM fallback.
Derived from fix/gsplat-mapper; the experiment uses production GL's classic rasterization.
"""

import math
from contextlib import contextmanager
from dataclasses import dataclass, replace

import numpy as np

from ito.protocol import Intrinsics
from ito.render.pose import validate_pose

MAX_GAUSSIANS = 1_000_000
C0 = 0.28209479177387814
CV = np.diag([1.0, -1.0, -1.0, 1.0])  # Ito camera axes to OpenCV's (+Y down, +Z forward)

ANCHORS = 128  # keyframe slots; slot 0 holds Gaussians whose keyframe aged out, in world
VIEWS = 24  # training views kept (images on the GPU)
RECENT = 2.0  # seconds: views this new get half the training steps
SEED_COVERED = 0.5  # a new view seeds Gaussians where the map's alpha is below this
SEED_NEARER = 0.8  # or where its points lie this much nearer than the map's surface
# A view's surface replaces the Gaussians it sees through: those this much nearer than it
# (relative) go, as the point map's carve. 0 keeps them for training to sort out.
CARVE = 0.0
CARVE_CELL = 4  # pixels per cell of the view's (nearest-surface) depth image
SEED_MAX = 80_000  # Gaussians seeded per keyframe
SEED_OPACITY = 0.995
# Under the budget, what goes: "stale" = the Gaussians no recent view saw (newest
# surroundings matter most), "opacity" = the faintest.
EVICT = "stale"
RESERVE = 0.1  # stale eviction keeps this share of the budget free for densification
REFINE_EVERY = 400  # steps
GROW_GRAD2D = 0.0002  # mean image-plane gradient (gsplat's DefaultStrategy units)
GROW_SCALE3D = 0.01  # of the scene scale: smaller ones clone, larger ones split
PRUNE_OPACITY = 0.005
PRUNE_SCALE3D = 0.1  # of the scene scale
WORLD = 1 << 62  # publication key of slot 0 (node ids are tracking's, from 0)
FILTER = 0.0  # Mip-Splatting 3D filter, in pixels of the closest training view


@dataclass(frozen=True)
class Keyframe:
    """A tracked frame for the map. points: metric camera-local, registered to colors and
    confidence (zero excludes); empty for a view that only trains. anchor: the keyframe
    node this view is held relative to (its own node for a keyframe)."""

    node: int
    rgb: np.ndarray  # uint8 camera-resolution H,W,3, owned by the caller until submit
    pose: np.ndarray  # world-from-camera
    intrinsics: Intrinsics
    points: object  # N,3 metric camera-local points (numpy or torch)
    colors: object  # N,3 uint8, registered to points
    confidence: object  # N
    captured_at: float
    depth: np.ndarray | None = None  # optional registered axial metres (driver depth)
    anchor: int | None = None
    sparse_depth: bool = False  # sensor zones; zero-confidence dense pixels stay excluded

    def __post_init__(self):
        i = self.intrinsics
        if self.rgb.dtype != np.uint8 or self.rgb.shape != (i.height, i.width, 3):
            raise ValueError("Keyframe RGB must match camera intrinsics")
        validate_pose(self.pose)
        if not np.isfinite(self.captured_at):
            raise ValueError("Keyframe capture time must be finite")
        if self.depth is not None and self.depth.shape != self.rgb.shape[:2]:
            raise ValueError("Keyframe depth must be registered to RGB")


@dataclass
class View:
    slot: int  # anchor slot the view's pose is relative to
    relative: np.ndarray  # anchor-from-camera
    image: object  # H,W,3 uint8 on the GPU
    intrinsics: Intrinsics
    captured_at: float
    depth: object = None
    depth_weight: object = None
    depth_support: object = None
    steps: int = 0
    exposure: object = None  # per-view colour affine (log gain, offset), trained
    exposure_optimizer: object = None
    keyframe: bool = False
    relative_gpu: object = None
    calibration: object = None


def quaternion_product(a, b):
    import torch

    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat(
        (
            aw * bw - (av * bv).sum(-1, keepdim=True),
            aw * bv + bw * av + torch.linalg.cross(av.expand_as(bv), bv),
        ),
        -1,
    )


def rotation_quaternion(matrix):
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(matrix).as_quat()
    return np.array([w, x, y, z])


def transform_gaussians(means, scales, quats, opacity, owner, poses, rotations, footprint):
    import torch

    pose = poses[owner]
    means = (pose[:, :3, :3] @ means[..., None])[..., 0] + pose[:, :3, 3]
    quats = quaternion_product(rotations[owner], torch.nn.functional.normalize(quats, dim=-1))
    scale2 = scales.exp().square()
    filtered = scale2 + footprint[:, None].square()
    opacity = opacity.sigmoid() * torch.sqrt(scale2.prod(-1) / filtered.prod(-1))
    return dict(means=means, quats=quats, scales=filtered.sqrt(), opacities=opacity)


class GaussianMap:
    LEARNING_RATES = dict(means=0.00016, scales=0.005, quats=0.001, opacities=0.05, sh0=0.0025)

    def __init__(
        self, budget=MAX_GAUSSIANS, *, anchors=ANCHORS, views=VIEWS, seed=0, precision="full"
    ):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Optimized Gaussian mapping needs an NVIDIA CUDA device")
        from scipy.interpolate import RBFInterpolator, griddata

        self.interpolate_depth = griddata
        self.smooth_depth = RBFInterpolator
        from gsplat import rasterization

        self.rasterize = rasterization
        self.transform = torch.compile(transform_gaussians, dynamic=True, fullgraph=True)
        from gsplat.strategy import ops

        self.ops = ops
        self.torch = torch
        self.budget = int(budget)
        if self.budget < 1 or anchors < 2 or views < 1:
            raise ValueError("Mapper budgets must be positive")
        if precision not in ("full", "half"):
            raise ValueError("Mapper precision must be full or half")
        self.precision = precision
        self.loss_dtype = (
            torch.float32
            if precision == "full"
            else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        )
        self.max_anchors, self.max_views = anchors, views
        self.generator = np.random.default_rng(seed)
        self.origin_node = None
        self.anchors = {}  # node -> dict(slot, pose_cpu); slot 0 is never a node's
        self.views = {}  # node -> View, oldest first
        self.changed = set()
        self.urgent = set()
        self.steps = 0
        self.image_iterations = 0
        self.iterations = {}
        self.pose_cache = None
        self.keyframes = []
        self.training_ids = None
        self.training_window = None
        self.pose_delta = torch.nn.Parameter(torch.zeros((anchors, 6), device="cuda"))
        self.pose_optimizer = torch.optim.Adam([self.pose_delta], lr=0.002, fused=True)
        self.scene_scale = None
        self.cv = self.tensor(CV)
        self.poses = torch.eye(4, device="cuda").repeat(anchors, 1, 1)
        self.slot_poses = np.tile(np.eye(4), (anchors, 1, 1))
        self.rotations = torch.zeros((anchors, 4), device="cuda")
        self.rotations[:, 0] = 1
        empty = dict(
            means=torch.empty((0, 3)),
            scales=torch.empty((0, 3)),
            quats=torch.empty((0, 4)),
            opacities=torch.empty(0),
            sh0=torch.empty((0, 1, 3)),
        )
        self.p = torch.nn.ParameterDict({k: torch.nn.Parameter(v.cuda()) for k, v in empty.items()})
        self.p["owner"] = torch.nn.Parameter(
            torch.empty(0, device="cuda", dtype=torch.long), requires_grad=False
        )
        self.optimizers = {
            k: torch.optim.Adam([self.p[k]], lr=lr, eps=1e-15, fused=True)
            for k, lr in self.LEARNING_RATES.items()
        }
        # Per-Gaussian running state, kept in step with the parameters by gsplat's ops.
        self.state = dict(
            grad2d=torch.zeros(0, device="cuda"),
            count=torch.zeros(0, device="cuda"),
            filter=torch.zeros(0, device="cuda"),
            origin=torch.empty((0, 3), device="cuda"),
            normal=torch.empty((0, 3), device="cuda"),
            seed_scale=torch.empty((0, 3), device="cuda"),
            support=torch.zeros(0, device="cuda"),
            rejected=torch.zeros(0, device="cuda"),
            seen=torch.zeros(0, device="cuda"),  # capture time of the newest view it was in
        )

    def tensor(self, value):
        return self.torch.as_tensor(value, device="cuda", dtype=self.torch.float32)

    @property
    def count(self):
        return len(self.p["means"])

    # -- anchors and views ---------------------------------------------------------------

    def correct(self, poses):
        for node, pose in poses.items():
            a = self.anchors.get(node)
            if a is None:
                continue
            pose = validate_pose(pose)
            if np.array_equal(a["pose_cpu"], pose):
                continue
            self._set_anchor(a["slot"], pose)
            a["pose_cpu"] = np.array(pose, dtype=np.float64, copy=True)
            self.changed.add(node)
            self.urgent.add(node)

    def _set_anchor(self, slot, pose):
        with self.torch.no_grad():
            self.pose_delta[slot].zero_()
            for key in ("exp_avg", "exp_avg_sq"):
                value = self.pose_optimizer.state.get(self.pose_delta, {}).get(key)
                if value is not None:
                    value[slot].zero_()
        self.pose_cache = None
        self.slot_poses[slot] = pose
        self.poses[slot] = self.tensor(pose)
        self.rotations[slot] = self.tensor(rotation_quaternion(np.asarray(pose)[:3, :3]))

    def forget(self, nodes):
        """Drop keyframes and their Gaussians (tracking says they were wrong or gone)."""
        slots = []
        for node in nodes:
            a = self.anchors.pop(node, None)
            if a is not None:
                slots.append(a["slot"])
                self.changed.add(node)
                self.urgent.add(node)
        for node in [n for n, v in self.views.items() if v.slot in slots]:
            del self.views[node]
        if slots:
            gone = self.torch.isin(self.p["owner"], self.torch.tensor(slots, device="cuda"))
            self._remove(gone)

    def _retire_oldest_anchor(self):
        """Bake the oldest keyframe's Gaussians into world slot 0: kept, no longer moved."""
        torch = self.torch
        node = next(iter(self.anchors))
        slot = self.anchors.pop(node)["slot"]
        with torch.no_grad():
            mine = self.p["owner"] == slot
            corrected, rotations = self.corrected_poses()
            pose = corrected[slot]
            self.p["means"][mine] = self.p["means"][mine] @ pose[:3, :3].T + pose[:3, 3]
            self.p["quats"][mine] = quaternion_product(
                rotations[slot : slot + 1], self.p["quats"][mine]
            )
            self.state["origin"] = torch.where(
                mine[:, None],
                self.state["origin"] @ pose[:3, :3].T + pose[:3, 3],
                self.state["origin"],
            )
            self.state["normal"] = torch.where(
                mine[:, None], self.state["normal"] @ pose[:3, :3].T, self.state["normal"]
            )
            self.p["owner"][mine] = 0
        for view in self.views.values():
            if view.slot == slot:
                view.relative = pose.detach().cpu().numpy() @ view.relative
                view.relative_gpu = self.tensor(view.relative)
                view.slot = 0
        # Republished under key 0; its own key's records go.
        self.changed.update((WORLD, node))
        self.urgent.update((WORLD, node))

    def anchor_pose(self, slot):
        return self.slot_poses[slot]

    def resize(self, budget):
        with self.structure():
            self.budget = int(budget)
            self._enforce_budget(self.budget)

    @contextmanager
    def structure(self):
        parameters, state, budget = dict(self.p.items()), self.state.copy(), self.budget
        saved = {}
        for key, optimizer in self.optimizers.items():
            states = optimizer.state.copy()
            states.update({p: values.copy() for p, values in states.items()})
            saved[key] = (states, [g["params"][:] for g in optimizer.param_groups])
        try:
            yield
        except self.torch.cuda.OutOfMemoryError:
            self.p = self.torch.nn.ParameterDict(parameters)
            self.state, self.budget = state, budget
            self.training_ids = None
            for key, (states, groups) in saved.items():
                optimizer = self.optimizers[key]
                optimizer.state = states
                for group, values in zip(optimizer.param_groups, groups, strict=True):
                    group["params"] = values
            raise

    def recover(self):
        self.pose_cache = None
        # Adam may fail between allocating its first and second moment buffers.
        for optimizer in [
            *self.optimizers.values(),
            self.pose_optimizer,
            *(v.exposure_optimizer for v in self.views.values()),
        ]:
            for parameter, state in list(optimizer.state.items()):
                if any(
                    k not in state or state[k].shape != parameter.shape
                    for k in ("exp_avg", "exp_avg_sq")
                ):
                    del optimizer.state[parameter]
            optimizer.zero_grad(set_to_none=True)

    def view_pose(self, view):
        return self.anchor_pose(view.slot) @ view.relative

    def add(self, frame):
        """An allocation failure drops this observation and retains the previous map."""
        torch = self.torch
        parameters = dict(self.p.items())
        state = self.state.copy()
        scene_scale = self.scene_scale
        origin_node = self.origin_node
        keyframes, iterations = self.keyframes[:], self.iterations.copy()
        learning_rates = {k: [g["lr"] for g in o.param_groups] for k, o in self.optimizers.items()}
        changed, urgent = self.changed.copy(), self.urgent.copy()
        anchors = {n: a.copy() for n, a in self.anchors.items()}
        views = {n: replace(v) for n, v in self.views.items()}
        saved_delta = self.pose_delta.detach().clone()
        saved_pose_state = {
            k: v.clone() if torch.is_tensor(v) else v
            for k, v in self.pose_optimizer.state.get(self.pose_delta, {}).items()
        }
        poses, rotations, slot_poses = (
            self.poses.clone(),
            self.rotations.clone(),
            self.slot_poses.copy(),
        )
        optimizer_states = {}
        for key, optimizer in self.optimizers.items():
            saved = optimizer.state.copy()
            # gsplat replaces tensors inside each mutable per-parameter state dict.
            saved.update({parameter: values.copy() for parameter, values in saved.items()})
            optimizer_states[key] = (saved, [g["params"][:] for g in optimizer.param_groups])
        # Retirement bakes an old anchor in-place; preserve those three arrays too.
        baked = (
            {k: self.p[k].detach().clone() for k in ("means", "quats", "owner")}
            if (len(self.anchors) >= self.max_anchors - 1)
            else {}
        )
        try:
            self._add(frame)
        except torch.cuda.OutOfMemoryError:
            self.p = torch.nn.ParameterDict(parameters)
            self.scene_scale, self.changed, self.urgent = scene_scale, changed, urgent
            self.origin_node = origin_node
            self.keyframes, self.iterations = keyframes, iterations
            self.pose_cache = None
            self.training_ids = None
            self.state, self.anchors, self.views = state, anchors, views
            self.poses, self.rotations, self.slot_poses = poses, rotations, slot_poses
            with torch.no_grad():
                self.pose_delta.copy_(saved_delta)
                self.pose_optimizer.state[self.pose_delta] = saved_pose_state
                for name, value in baked.items():
                    self.p[name].copy_(value)
            for key, (saved, groups) in optimizer_states.items():
                optimizer = self.optimizers[key]
                optimizer.state = saved
                for group, values in zip(optimizer.param_groups, groups, strict=True):
                    group["params"] = values
                for group, rate in zip(optimizer.param_groups, learning_rates[key], strict=True):
                    group["lr"] = rate
            raise

    def _add(self, frame):
        """A view; a keyframe (anchor == node or None) also seeds where the map is thin."""
        torch = self.torch
        anchor = frame.node if frame.anchor is None else frame.anchor
        pose = validate_pose(frame.pose).astype(np.float64)
        if frame.anchor is None and self.anchors:
            last_node = next(reversed(self.anchors))
            last = self.anchors[last_node]
            relative_motion = np.linalg.inv(last["pose_cpu"]) @ pose
            angle = math.acos(float(np.clip((np.trace(relative_motion[:3, :3]) - 1) / 2, -1, 1)))
            if (
                frame.captured_at - last["captured_at"] < 1.0
                and angle < math.radians(30)
                and np.linalg.norm(relative_motion[:3, 3]) < 0.30
            ):
                anchor = last_node
        if anchor == frame.node:
            if frame.node in self.anchors:
                self.correct({frame.node: pose})
                return
            if len(self.anchors) >= self.max_anchors - 1:
                self._retire_oldest_anchor()
            used = {a["slot"] for a in self.anchors.values()}
            slot = next(s for s in range(1, self.max_anchors) if s not in used)
            if self.origin_node is None:
                self.origin_node = frame.node
            self.anchors[frame.node] = dict(
                slot=slot, pose_cpu=pose.copy(), captured_at=frame.captured_at
            )
            self._set_anchor(slot, pose)
            relative = np.eye(4)
        elif anchor in self.anchors:
            slot = self.anchors[anchor]["slot"]
            relative = np.linalg.inv(self.anchor_pose(slot)) @ pose
        else:
            slot, relative = 0, pose
        if frame.node in self.views:
            return
        while len(self.views) >= self.max_views:
            del self.views[next(iter(self.views))]
        exposure = torch.nn.Parameter(torch.zeros((2, 3), device="cuda"))
        depth_weight, depth_support = self._depth_reliability(frame)
        view = View(
            slot,
            relative,
            torch.as_tensor(frame.rgb, device="cuda").clone(),
            frame.intrinsics,
            float(frame.captured_at),
            None if frame.depth is None else self.tensor(frame.depth),
            depth_weight=depth_weight,
            depth_support=depth_support,
            relative_gpu=self.tensor(relative),
            calibration=self.tensor(
                [
                    [frame.intrinsics.fx, 0, frame.intrinsics.cx],
                    [0, frame.intrinsics.fy, frame.intrinsics.cy],
                    [0, 0, 1],
                ]
            ),
            exposure=exposure,
            exposure_optimizer=torch.optim.Adam([exposure], lr=0.001, fused=True),
        )
        self.views[frame.node] = view
        self.iterations[frame.node] = 0
        view.keyframe = anchor == frame.node
        if view.keyframe:
            self.keyframes.append(frame.node)
        if frame.points is not None:
            self._seed(frame, slot, pose, relative)
        self._invalidate_candidates()

    def _depth_reliability(self, frame):
        """Only independent raw observations can confirm depth, never the fitted map."""
        if frame.depth is None:
            return None, None
        torch = self.torch
        i = frame.intrinsics
        depth = self.tensor(frame.depth)
        confidence = self.tensor(frame.confidence).reshape(depth.shape).clamp(0, 1)
        valid = torch.isfinite(depth) & (depth > 0) & torch.isfinite(confidence)
        confidence = torch.where(valid, confidence, 0)
        depth = torch.where(valid, depth, 0)
        support = torch.zeros_like(depth)
        covered = torch.zeros_like(valid)
        agreement = torch.zeros_like(depth)
        yy, xx = torch.meshgrid(
            torch.arange(i.height, device="cuda"),
            torch.arange(i.width, device="cuda"),
            indexing="ij",
        )
        for previous in list(self.views.values())[-3:]:
            if previous.depth is None:
                continue
            old = previous.depth
            reliable = torch.isfinite(old) & (old > 0)
            reliable &= previous.depth_weight > 0.25
            local = torch.stack(
                ((xx - i.cx) / i.fx * old, -(yy - i.cy) / i.fy * old, -old), -1
            ).reshape(-1, 3)
            transform = self.tensor(np.linalg.inv(frame.pose) @ self.view_pose(previous))
            local = local @ transform[:3, :3].T + transform[:3, 3]
            z = -local[:, 2]
            u = (i.fx * local[:, 0] / z.clamp_min(0.01) + i.cx).round().long()
            v = (i.cy - i.fy * local[:, 1] / z.clamp_min(0.01)).round().long()
            inside = reliable.reshape(-1) & (z > 0.01)
            inside &= (u >= 0) & (u < i.width) & (v >= 0) & (v < i.height)
            nearest = torch.full((i.height * i.width,), torch.inf, device="cuda")
            nearest.scatter_reduce_(0, v[inside] * i.width + u[inside], z[inside], "amin")
            nearest = nearest.reshape(depth.shape)
            seen = torch.isfinite(nearest)
            difference = (nearest - depth).abs() / (0.03 + 0.06 * depth.clamp_min(0))
            agreement = torch.maximum(agreement, torch.where(seen, (-difference.square()).exp(), 0))
            support += seen & (difference < 1)
            covered |= seen
        weight = confidence * torch.where(covered, agreement, 1)
        return weight.detach(), support.detach()

    def _check_surfaces(self, frame):
        """Two contradictory views can retire a provisional floater, not a supported wall."""
        if not self.count or frame.depth is None:
            return
        torch = self.torch
        with torch.no_grad():
            view = self.views[frame.node]
            i = frame.intrinsics
            camera = self.tensor(np.linalg.inv(frame.pose))
            means = self.world()["means"]
            local = means @ camera[:3, :3].T + camera[:3, 3]
            z = -local[:, 2]
            u = (i.fx * local[:, 0] / z.clamp_min(0.01) + i.cx).round().long()
            v = (i.cy - i.fy * local[:, 1] / z.clamp_min(0.01)).round().long()
            inside = (z > 0.01) & (u >= 0) & (u < i.width) & (v >= 0) & (v < i.height)
            u, v = u.clamp(0, i.width - 1), v.clamp(0, i.height - 1)
            depth = view.depth[v, u]
            evidence = inside & (view.depth_weight[v, u] > 0.4)
            tolerance = 0.03 + 0.06 * depth
            self.state["support"] = self.state["support"] + (
                evidence & ((z - depth).abs() < tolerance)
            )
            contradiction = evidence & (view.depth_support[v, u] > 0) & (z < depth - tolerance)
            self.state["rejected"] = self.state["rejected"] + contradiction
            self._remove((self.state["rejected"] >= 2) & (self.state["support"] < 3))

    def _seed(self, frame, slot, pose, relative):
        self._check_surfaces(frame)
        torch = self.torch
        seed_points, seed_confidence = frame.points, frame.confidence
        if self.views[frame.node].depth_weight is not None:
            seed_confidence = self.views[frame.node].depth_weight
        measured_depth = None
        if frame.depth is not None:
            valid_depth = np.isfinite(frame.depth) & (frame.depth > 0)
            if frame.sparse_depth and 4 <= np.count_nonzero(valid_depth) < frame.depth.size // 20:
                measured_depth = self.tensor(valid_depth).bool()
                # ToF supplies zones, not object boundaries. This smooth inverse-depth
                # prior initializes colour-bearing surfaces; only measured zones enter
                # the depth loss. RGB observations subsequently refine the geometry.
                yy, xx = np.mgrid[: frame.intrinsics.height, : frame.intrinsics.width]
                sites = np.stack((xx[valid_depth], yy[valid_depth]), -1)
                inverse = 1 / frame.depth[valid_depth]
                nearest = self.interpolate_depth(sites, inverse, (xx, yy), method="nearest")
                # Colour-guided regularization suppresses zone noise without claiming
                # measured object boundaries. Original zones alone enter the depth loss.
                i = frame.intrinsics
                features = np.concatenate(
                    (np.stack(((xx - i.cx) / i.fx, (yy - i.cy) / i.fy), -1), frame.rgb / 128), -1
                )
                dense = self.smooth_depth(
                    features[valid_depth],
                    inverse,
                    kernel="linear",
                    degree=0,
                    smoothing=0.10,
                )(features.reshape(-1, 5)).reshape(xx.shape)
                dense = 1 / np.clip(np.where(np.isfinite(dense), dense, nearest), 0.25, 20)
                i = frame.intrinsics
                seed_points = np.stack(
                    ((xx - i.cx) / i.fx * dense, -(yy - i.cy) / i.fy * dense, -dense), -1
                )
                seed_confidence = np.ones_like(dense)
        points = torch.as_tensor(seed_points, device="cuda").detach().reshape(-1, 3).float()
        colors = torch.as_tensor(frame.colors, device="cuda").detach().reshape(-1, 3).float()
        confidence = torch.as_tensor(seed_confidence, device="cuda").detach().reshape(-1)
        valid = (
            torch.isfinite(points).all(-1)
            & torch.isfinite(confidence.float())
            & (confidence.float() > 0.25)
            & (points[:, 2] < -0.01)
        )
        points, colors = points[valid], colors[valid] / 255
        if len(points) < 4:
            return
        if self.scene_scale is None:
            depth = -points[:, 2]
            self.scene_scale = float(max(0.5, torch.quantile(depth[::7], 0.9)))
            for k in ("means",):
                for group in self.optimizers[k].param_groups:
                    group["lr"] = self.LEARNING_RATES[k] * self.scene_scale
        # Where does the map already explain this view? Project each point into it.
        i = frame.intrinsics
        z = -points[:, 2]
        u = (i.fx * points[:, 0] / z + i.cx).long().clamp(0, i.width - 1)
        v = (i.cy - i.fy * points[:, 1] / z).long().clamp(0, i.height - 1)
        if CARVE > 0 and self.count:
            self._carve(pose, i, u, v, z)
        if self.count:
            with torch.no_grad():
                explained_rgb, depth, alpha, _ = self.render(pose, i)
            # Expected depth is already normalised by alpha.
            nearer = z < SEED_NEARER * depth[v, u]
            # A depth disagreement alone must not paint a second opaque floor
            # over a map that already explains the observed colour.
            nearer &= (explained_rgb[v, u] - colors).abs().mean(-1) > 0.10
            view = self.views[frame.node]
            if view.depth_support is not None:
                nearer &= (view.depth_support[v, u] >= 2) & (view.depth_weight[v, u] > 0.4)
            if measured_depth is not None:
                # Completed ToF rays are initialization priors, not new metric evidence.
                # Noise or extrapolation must not stack opaque replacement surfaces.
                nearer &= measured_depth[v, u]
            fresh = (alpha[v, u] < SEED_COVERED) | nearer
            points, colors, z = points[fresh], colors[fresh], z[fresh]
        # Sample a regular two-pixel grid, retaining one-pixel detail at colour edges.
        # Unlike random sparse seeds this supplies a defined calibrated footprint.
        u = (i.fx * points[:, 0] / z + i.cx).round().long().clamp(0, i.width - 1)
        v = (i.cy - i.fy * points[:, 1] / z).round().long().clamp(0, i.height - 1)
        image = self.tensor(frame.rgb) / 255
        edge = torch.zeros((i.height, i.width), device="cuda", dtype=torch.bool)
        edge[:, 1:] |= (image[:, 1:] - image[:, :-1]).abs().amax(-1) > 0.08
        edge[1:] |= (image[1:] - image[:-1]).abs().amax(-1) > 0.08
        detail = edge[v, u]
        keep = detail | ((u % 2 == 0) & (v % 2 == 0))
        ids = torch.nonzero(keep, as_tuple=True)[0]
        n = min(len(ids), SEED_MAX, self.budget)
        if n < 4:
            return
        if len(ids) > n:
            ids = ids[torch.linspace(0, len(ids) - 1, n, device="cuda").long()]
        points, colors, z, detail = points[ids], colors[ids], z[ids], detail[ids]
        spacing = torch.where(detail, 0.70, 1.40)
        # Calibrated surface tangents avoid camera-facing disks opening cracks on floors.
        grid = torch.as_tensor(seed_points, device="cuda", dtype=torch.float32).reshape(
            i.height, i.width, 3
        )
        su, sv = u[ids], v[ids]
        dx = (grid[sv, (su + 1).clamp_max(i.width - 1)] - grid[sv, (su - 1).clamp_min(0)]) / 2
        dy = (grid[(sv + 1).clamp_max(i.height - 1), su] - grid[(sv - 1).clamp_min(0), su]) / 2
        smooth = (
            torch.isfinite(dx).all(-1)
            & torch.isfinite(dy).all(-1)
            & (dx.norm(dim=-1) < z * 0.02)
            & (dy.norm(dim=-1) < z * 0.02)
            & (dx[:, 2].abs() < 0.01 + z * 0.01)
            & (dy[:, 2].abs() < 0.01 + z * 0.01)
        )
        fallback_x = torch.stack((z / i.fx, torch.zeros_like(z), torch.zeros_like(z)), -1)
        fallback_y = torch.stack((torch.zeros_like(z), -z / i.fy, torch.zeros_like(z)), -1)
        dx = torch.where(smooth[:, None], dx, fallback_x)
        dy = torch.where(smooth[:, None], dy, fallback_y)
        axis_x = torch.nn.functional.normalize(dx, dim=-1)
        normal = torch.nn.functional.normalize(torch.linalg.cross(dx, dy), dim=-1)
        normal = torch.where(normal[:, 2:3] < 0, -normal, normal)
        axis_y = torch.linalg.cross(normal, axis_x)
        basis = torch.stack((axis_x, axis_y, normal), -1)
        qw = (1 + basis.diagonal(dim1=-2, dim2=-1).sum(-1)).clamp_min(1e-6).sqrt() / 2
        quat = torch.stack(
            (
                qw,
                (basis[:, 2, 1] - basis[:, 1, 2]) / (4 * qw),
                (basis[:, 0, 2] - basis[:, 2, 0]) / (4 * qw),
                (basis[:, 1, 0] - basis[:, 0, 1]) / (4 * qw),
            ),
            -1,
        )
        quat = quaternion_product(self.tensor(rotation_quaternion(relative[:3, :3]))[None], quat)
        scales = torch.stack(
            (dx.norm(dim=-1) * spacing, (dy * axis_y).sum(-1).abs() * spacing, z / i.fx * 0.15), -1
        )
        self._enforce_budget(self.budget - n)
        rotation = self.tensor(relative[:3, :3])
        values = dict(
            means=points @ rotation.T + self.tensor(relative[:3, 3]),
            scales=scales.clamp_min(0.0001).log(),
            quats=torch.nn.functional.normalize(quat, dim=-1),
            opacities=torch.full((n,), math.log(SEED_OPACITY / (1 - SEED_OPACITY)), device="cuda"),
            sh0=((colors - 0.5) / C0)[:, None],
            owner=torch.full((n,), slot, device="cuda", dtype=torch.long),
        )
        self._append(
            values,
            dict(
                filter=z / i.fx * FILTER,
                origin=values["means"].detach().clone(),
                normal=normal @ rotation.T,
                seed_scale=scales.clamp_min(0.0001),
                support=torch.ones(n, device="cuda"),
                seen=torch.full((n,), frame.captured_at, device="cuda"),
            ),
        )
        self.changed.add(self.key(slot))

    def _carve(self, pose, i, u, v, z):
        """Drop Gaussians in front of the view's surface: it sees through them."""
        torch = self.torch
        cw, ch = -(-i.width // CARVE_CELL), -(-i.height // CARVE_CELL)
        surface = torch.full((ch * cw,), float("inf"), device="cuda")
        surface.scatter_reduce_(0, (v // CARVE_CELL) * cw + u // CARVE_CELL, z, "amin")
        with torch.no_grad():
            camera = torch.linalg.inv(self.tensor(pose))
            means = self.world()["means"]
            local = means @ camera[:3, :3].T + camera[:3, 3]
            depth = -local[:, 2]
            ahead = depth > 0.01
            gu = i.fx * local[:, 0] / depth.clamp_min(0.01) + i.cx
            gv = i.cy - i.fy * local[:, 1] / depth.clamp_min(0.01)
            inside = ahead & (gu >= 0) & (gu < i.width) & (gv >= 0) & (gv < i.height)
            cell = (gv.clamp(0, i.height - 1).long() // CARVE_CELL) * cw + (
                gu.clamp(0, i.width - 1).long() // CARVE_CELL
            )
            seen = surface[cell]
            self._remove(inside & torch.isfinite(seen) & (depth < seen * (1 - CARVE) - 0.02))

    def _append(self, values, state):
        torch = self.torch
        n = len(values["means"])

        def param_fn(name, p):
            return torch.nn.Parameter(
                torch.cat((p.detach(), values[name].to(p.dtype))), requires_grad=p.requires_grad
            )

        def optimizer_fn(key, v):
            return torch.cat((v, torch.zeros((n, *v.shape[1:]), device="cuda")))

        self.ops._update_param_with_optimizer(param_fn, optimizer_fn, self.p, self.optimizers)
        self._invalidate_candidates()
        for k, v in self.state.items():
            extra = state.get(k, torch.zeros(n, device="cuda"))
            self.state[k] = torch.cat((v, extra))

    def key(self, slot):
        """The publication key of an anchor slot; None for a slot just forgotten."""
        if slot == 0:
            return WORLD
        return next((n for n, a in self.anchors.items() if a["slot"] == slot), None)

    def _remove(self, mask):
        if mask.any():
            slots = self.torch.unique(self.p["owner"][mask]).tolist()
            self.ops.remove(self.p, self.optimizers, self.state, mask)
            self._invalidate_candidates()
            self.changed.update(k for k in map(self.key, slots) if k is not None)

    def _enforce_budget(self, budget):
        """Down to budget: the least recently seen (faintest first among equals), or the
        faintest."""
        excess = self.count - max(0, budget)
        if excess > 0:
            torch = self.torch
            score = self.p["opacities"].detach().sigmoid()
            if EVICT == "stale":
                score = self.state["seen"] + 1e-3 * score
            weakest = torch.topk(score, excess, largest=False).indices
            mask = torch.zeros(self.count, dtype=torch.bool, device="cuda")
            mask[weakest] = True
            self._remove(mask)

    # -- rendering -----------------------------------------------------------------------

    def world(self, train=False, ids=None, corrected=None):
        """Differentiable world-space Gaussians, with calibrated surface footprints."""
        p = {k: v if train else v.detach() for k, v in self.p.items()}
        if ids is not None:
            p = {k: v[ids] for k, v in p.items()}
        corrected, rotations = self.corrected_poses(train) if corrected is None else corrected
        footprint = self.state["filter"] if ids is None else self.state["filter"][ids]
        return self.transform(
            p["means"],
            p["scales"],
            p["quats"],
            p["opacities"],
            p["owner"],
            corrected,
            rotations,
            footprint,
        )

    def inverse_pose(self, pose):
        rotation = pose[..., :3, :3].transpose(-1, -2)
        top = self.torch.cat((rotation, -rotation @ pose[..., :3, 3:4]), -1)
        return self.torch.cat((top, pose[..., 3:4, :]), -2)

    def render(self, pose, intrinsics, *, train=False, world=None):
        """GPU colour, expected axial depth, alpha and gsplat's info; differentiable in a
        camera-local pose delta and, with train, in the map."""
        torch = self.torch
        i = intrinsics
        pose = self.tensor(pose) if not torch.is_tensor(pose) else pose
        if not self.count:
            zero = torch.zeros((i.height, i.width), device="cuda")
            return zero[..., None].expand(-1, -1, 3), zero, zero, {}
        if world is None:
            world = self.world(train)
        K = self.tensor([[i.fx, 0, i.cx], [0, i.fy, i.cy], [0, 0, 1]])
        sh0 = self.p["sh0"] if train else self.p["sh0"].detach()
        pred, alpha, info = self.rasterize(
            **world,
            colors=sh0,
            sh_degree=0,
            viewmats=self.inverse_pose(pose @ self.cv)[None],
            Ks=K[None],
            width=i.width,
            height=i.height,
            packed=False,
            near_plane=0.01,
            far_plane=1000,
            rasterize_mode="classic",
            render_mode="RGB+ED",
        )
        return pred[0, ..., :3], pred[0, ..., 3], alpha[0, ..., 0], info

    # -- training ------------------------------------------------------------------------

    def corrected_poses(self, train=False):
        """A bounded SE(3) correction used only by the map, never returned to tracking."""
        torch = self.torch
        delta = self.pose_delta if train else self.pose_delta.detach()
        # The world slot and first keyframe define the map gauge; fitting must not
        # drift the entire map away from its unchanged published camera path.
        active = torch.ones((self.max_anchors, 1), device="cuda")
        active[0] = 0
        origin = self.anchors.get(self.origin_node)
        if origin is not None:
            active[origin["slot"]] = 0
        delta = delta * active
        translation = 0.10 * delta[:, :3].tanh()
        angle = math.radians(5) * delta[:, 3:].tanh()
        theta = (angle.square().sum(-1, keepdim=True) + 1e-12).sqrt()
        q = torch.cat(((theta / 2).cos(), angle * (0.5 * torch.sinc(theta / (2 * math.pi)))), -1)
        w, x, y, z = q.unbind(-1)
        rotation = torch.stack(
            (
                1 - 2 * (y * y + z * z),
                2 * (x * y - w * z),
                2 * (x * z + w * y),
                2 * (x * y + w * z),
                1 - 2 * (x * x + z * z),
                2 * (y * z - w * x),
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x * x + y * y),
            ),
            -1,
        ).reshape(-1, 3, 3)
        top = torch.cat((rotation, translation[..., None]), -1)
        bottom = self.poses[:, 3:4]
        correction = torch.cat((top, bottom), -2)
        return self.poses @ correction, quaternion_product(self.rotations, q)

    def _invalidate_candidates(self):
        self.training_ids = None

    def _pick(self):
        nodes = list(self.views)
        # Focus new keyframes until they have 100 image iterations; keep overlap in play.
        pending = [n for n in nodes if self.views[n].keyframe and self.views[n].steps < 100]
        first = pending[0] if pending else nodes[-1]
        recent = nodes[-6:]
        other = (
            pending[1]
            if len(pending) > 1
            else nodes[-1]
            if first != nodes[-1]
            else recent[int(self.generator.integers(len(recent)))]
        )
        return [first, other] if first != other else [first]

    def step(self):
        if not self.count or not self.views:
            return
        torch = self.torch
        chosen = self._pick()
        # Batched calibrated views share one map transform and optimizer update.
        poses, matrices, targets, depths, views = [], [], [], [], []
        pose_step = self.steps % 8 == 0
        if pose_step:
            corrected, rotations = self.corrected_poses(True)
        else:
            if self.pose_cache is None:
                self.pose_cache = self.corrected_poses()
            corrected, rotations = self.pose_cache
        window = tuple(dict.fromkeys([*list(self.views)[-6:], *chosen]))
        if self.training_ids is None or window != self.training_window:
            self.training_window = window
            with torch.no_grad():
                owner = self.p["owner"]
                base = corrected.detach()[owner]
                means = (base[:, :3, :3] @ self.p["means"][..., None])[..., 0] + base[:, :3, 3]
                visible = torch.zeros(self.count, dtype=torch.bool, device="cuda")
                # A local window bounds training independently of archived map size.
                for node in window:
                    view = self.views[node]
                    camera = self.inverse_pose(corrected[view.slot].detach() @ view.relative_gpu)
                    local = means @ camera[:3, :3].T + camera[:3, 3]
                    z = -local[:, 2]
                    i = view.intrinsics
                    u = i.fx * local[:, 0] / z.clamp_min(0.01) + i.cx
                    v = i.cy - i.fy * local[:, 1] / z.clamp_min(0.01)
                    visible |= (
                        (z > 0.01)
                        & (u > -32)
                        & (u < i.width + 32)
                        & (v > -32)
                        & (v < i.height + 32)
                    )
                self.training_ids = torch.nonzero(visible, as_tuple=True)[0]
        ids = self.training_ids
        width, height = (
            self.views[chosen[0]].intrinsics.width,
            self.views[chosen[0]].intrinsics.height,
        )
        for node in chosen:
            view = self.views[node]
            i = view.intrinsics
            pose = corrected[view.slot] @ view.relative_gpu
            poses.append(self.inverse_pose(pose @ self.cv))
            matrices.append(view.calibration)
            targets.append(view.image)
            depths.append(view.depth)
            views.append(view)
        if not len(ids):
            return
        world = self.world(True, ids, (corrected, rotations))
        pred, alpha, info = self.rasterize(
            **world,
            colors=self.p["sh0"][ids],
            sh_degree=0,
            viewmats=torch.stack(poses),
            Ks=torch.stack(matrices),
            width=width,
            height=height,
            packed=False,
            near_plane=0.01,
            far_plane=1000,
            rasterize_mode="classic",
            render_mode="RGB+ED",
        )
        info["means2d"].retain_grad()
        target = torch.stack(targets).to(self.loss_dtype) / 255
        exposure = torch.stack([v.exposure for v in views])
        rgb = (pred[..., :3] * exposure[:, None, None, 0].exp() + exposure[:, None, None, 1]).to(
            self.loss_dtype
        )
        with torch.autocast("cuda", dtype=self.loss_dtype, enabled=self.precision == "half"):
            reliability = torch.stack(
                [
                    torch.ones((height, width), device="cuda")
                    if v.depth_weight is None
                    else 0.2 + 0.8 * torch.where(v.depth > 0, v.depth_weight, 1)
                    for v in views
                ]
            ).to(self.loss_dtype)
            loss = ((rgb - target).abs() * reliability[..., None]).mean()
            # Local colour gradients replace the expensive offline SSIM window.
            loss = (
                loss
                + 0.1 * ((rgb[:, 1:] - rgb[:, :-1]) - (target[:, 1:] - target[:, :-1])).abs().mean()
            )
            loss = (
                loss
                + 0.1
                * ((rgb[:, :, 1:] - rgb[:, :, :-1]) - (target[:, :, 1:] - target[:, :, :-1]))
                .abs()
                .mean()
            )
        for b, depth in enumerate(depths):
            if depth is not None:
                mask = torch.isfinite(depth) & (depth > 0)
                measured = torch.where(mask, depth, 1)
                difference = (pred[b, ..., 3] - measured).abs() / measured.clamp_min(0.1)
                weight = views[b].depth_weight
                # Cauchy influence decays for incompatible depth instead of dragging
                # already supported geometry toward each new neural guess.
                robust = 0.05 * torch.log1p((difference / 0.05).square())
                loss = loss + 0.05 * (
                    torch.where(mask, robust, 0) * weight
                ).sum() / weight.sum().clamp_min(1)
        displacement = self.p["means"][ids] - self.state["origin"][ids]
        radius = 0.005 + self.state["seed_scale"][ids].amax(-1) * 4
        loss = loss + 0.005 * (displacement / radius[:, None]).square().mean()
        q = torch.nn.functional.normalize(self.p["quats"][ids], dim=-1)
        w, x, y, z = q.unbind(-1)
        normal = torch.stack(
            (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)), -1
        )
        agreement = (normal * self.state["normal"][ids]).sum(-1).square()
        loss = loss + 0.002 * (1 - agreement).clamp_min(0).mean()
        loss = loss + 0.01 * exposure.square().mean()
        if pose_step:
            slots = torch.tensor([v.slot for v in views], device="cuda")
            loss = loss + 0.2 * self.pose_delta[slots].square().mean()
        loss.backward()
        if self.steps % 100 == 0 and not torch.isfinite(loss):
            raise RuntimeError("non-finite Gaussian training loss")
        grads = info["means2d"].grad.detach()
        seen = (info["radii"] > 0).all(-1)
        with torch.no_grad():
            self.state["grad2d"][ids] += (grads.norm(dim=-1) * max(width, height) / 2).sum(0)
            self.state["count"][ids] += seen.sum(0)
            self.state["seen"][ids] = torch.where(
                seen.any(0), max(v.captured_at for v in views), self.state["seen"][ids]
            )
        for optimizer in (
            *self.optimizers.values(),
            *((self.pose_optimizer,) if pose_step else ()),
            *(v.exposure_optimizer for v in views),
        ):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            # A depth-pixel disk may refine, but must not grow into a thick blob.
            reference = self.state["seed_scale"][ids]
            self.p["scales"][ids] = torch.minimum(self.p["scales"][ids], (reference * 2).log())
        if pose_step:
            self.pose_cache = None
        for node, view in zip(chosen, views, strict=True):
            view.steps += 1
            self.iterations[node] += 1
            with torch.no_grad():
                view.exposure.clamp_(-0.3, 0.3)
        self.steps += 1
        self.image_iterations += len(views)
        if self.steps % REFINE_EVERY == 0:
            self._refine()
        self.changed.update(self.anchors)
        self.changed.add(WORLD)

    def _refine(self):
        with self.structure():
            self._refine_parameters()

    def _refine_parameters(self):
        torch = self.torch
        with torch.no_grad():
            if EVICT == "stale" and self.count > self.budget * (1 - RESERVE / 2):
                self._enforce_budget(int(self.budget * (1 - RESERVE)))
            grads = self.state["grad2d"] / self.state["count"].clamp_min(1)
            high = grads > GROW_GRAD2D
            room = self.budget - self.count
            if room <= 0:
                high[:] = False
            elif int(high.sum()) > room // 2:
                # Under the budget the steepest grow first; a split adds one.
                keep = torch.topk(torch.where(high, grads, -1.0), room // 2).indices
                high[:] = False
                high[keep] = True
            small = self.p["scales"].exp().max(-1).values <= GROW_SCALE3D * self.scene_scale
            clone = high & small
            if clone.any():
                self.ops.duplicate(self.p, self.optimizers, self.state, clone)
            split = torch.cat(
                (high & ~small, torch.zeros(int(clone.sum()), dtype=torch.bool, device="cuda"))
            )
            if split.any():
                self.ops.split(self.p, self.optimizers, self.state, split)
            prune = (self.p["opacities"].sigmoid() < PRUNE_OPACITY) | (
                self.p["scales"].exp().max(-1).values > PRUNE_SCALE3D * self.scene_scale
            )
            self._remove(prune)
            self.state["grad2d"].zero_()
            self.state["count"].zero_()
            self._update_filter()
        self._invalidate_candidates()
        self.changed.update(self.anchors)
        self.changed.add(WORLD)

    def _update_filter(self):
        # Calibrated tangent footprints already include the sampling footprint.
        # A 3D mip filter would attenuate thin, near-opaque surface seeds.
        self.state["filter"].zero_()

    # -- publication ---------------------------------------------------------------------

    def keys(self):
        """Publication keys: anchor nodes, and WORLD for Gaussians whose keyframe aged out."""
        return [WORLD, *self.anchors]

    def sizes(self):
        """Gaussians per publication key."""
        counts = self.torch.bincount(self.p["owner"], minlength=self.max_anchors).tolist()
        sizes = {node: counts[a["slot"]] for node, a in self.anchors.items()}
        sizes[WORLD] = counts[0]
        return sizes

    def records_for(self, nodes):
        """World-space ring records (filter baked in) per key; empty for a key that is gone."""
        torch = self.torch
        slots = {n: 0 if n == WORLD else self.anchors.get(n, {}).get("slot") for n in nodes}
        wanted = [s for s in slots.values() if s is not None]
        result = {n: np.empty((0, 4, 4), np.float32) for n in nodes}
        if not wanted or not self.count:
            return result
        with torch.no_grad():
            owner = self.p["owner"]
            selected = torch.isin(owner, torch.tensor(wanted, device="cuda"))
            order = torch.argsort(owner[selected], stable=True)
            world = {k: v[selected][order] for k, v in self.world().items()}
            r = torch.zeros((len(order), 4, 4), device="cuda")
            r[:, 0, :3] = world["means"]
            r[:, 0, 3] = world["opacities"]
            r[:, 1, :3] = world["scales"]
            r[:, 1, 3] = 1e30  # never fades
            r[:, 2] = world["quats"]
            r[:, 3, :3] = self.p["sh0"][selected][order][:, 0]
            owners = owner[selected][order].cpu().numpy()
            r = r.cpu().numpy()
        for node, slot in slots.items():
            if slot is not None:
                lo, hi = np.searchsorted(owners, [slot, slot + 1])
                result[node] = r[lo:hi]
        return result

    def records(self, node):
        return self.records_for([node])[node]
