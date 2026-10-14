"""Online gsplat mapper for the oracle intervention; never used to align camera poses.

Retains calibrated RGB views, fits center/quaternion/three scales/opacity/SH0 and
per-image exposure with Adam, and densifies from image gradients. Source views move
only with their supplied anchor. Native gsplat/Adam parameters stay fp32 in both
settings; Half uses bf16/fp16 image loss and Full uses fp32, without an OOM fallback.
Derived from fix/gsplat-mapper; the experiment uses production GL's classic rasterization.
"""

import math
from dataclasses import dataclass, replace

import numpy as np

from ito.protocol import Intrinsics
from ito.render.pose import validate_pose

MAX_GAUSSIANS = 1_000_000
C0 = 0.28209479177387814
CV = np.diag([1.0, -1.0, -1.0, 1.0])  # Ito camera axes to OpenCV's (+Y down, +Z forward)

ANCHORS = 128  # keyframe slots; slot 0 holds Gaussians whose keyframe aged out, in world
VIEWS = 96  # training views kept (images on the GPU)
RECENT = 2.0  # seconds: views this new get half the training steps
SEED_COVERED = 0.5  # a new view seeds Gaussians where the map's alpha is below this
SEED_NEARER = 0.8  # or where its points lie this much nearer than the map's surface
# A view's surface replaces the Gaussians it sees through: those this much nearer than it
# (relative) go, as the point map's carve. 0 keeps them for training to sort out.
CARVE = 0.15
CARVE_CELL = 4  # pixels per cell of the view's (nearest-surface) depth image
SEED_MAX = 6000  # Gaussians seeded per keyframe
SEED_OPACITY = 0.1
# Under the budget, what goes: "stale" = the Gaussians no recent view saw (newest
# surroundings matter most), "opacity" = the faintest.
EVICT = "stale"
RESERVE = 0.1  # stale eviction keeps this share of the budget free for densification
REFINE_EVERY = 100  # steps
GROW_GRAD2D = 0.0002  # mean image-plane gradient (gsplat's DefaultStrategy units)
GROW_SCALE3D = 0.01  # of the scene scale: smaller ones clone, larger ones split
PRUNE_OPACITY = 0.005
PRUNE_SCALE3D = 0.1  # of the scene scale
WORLD = 1 << 62  # publication key of slot 0 (node ids are tracking's, from 0)
FILTER = math.sqrt(0.2)  # Mip-Splatting 3D filter, in pixels of the closest training view


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
    steps: int = 0
    exposure: object = None  # per-view colour affine (log gain, offset), trained
    exposure_optimizer: object = None


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


def ssim(x, y):
    """RGB SSIM with the standard 11-pixel, sigma 1.5 Gaussian window."""
    import torch
    import torch.nn.functional as F

    x, y = x.permute(2, 0, 1)[None], y.permute(2, 0, 1)[None]
    k = torch.arange(11, device=x.device, dtype=x.dtype) - 5
    k = torch.exp(-k.square() / (2 * 1.5**2))
    k /= k.sum()
    z = torch.cat((x, y, x * x, y * y, x * y), 0)
    z = F.conv2d(z, k.view(1, 1, 1, 11).expand(3, 1, 1, 11), groups=3)
    z = F.conv2d(z, k.view(1, 1, 11, 1).expand(3, 1, 11, 1), groups=3)
    mx, my, xx, yy, xy = z.unbind(0)
    return (
        ((2 * mx * my + 0.01**2) * (2 * (xy - mx * my) + 0.03**2))
        / ((mx.square() + my.square() + 0.01**2) * (xx - mx.square() + yy - my.square() + 0.03**2))
    ).mean()


def rotation_quaternion(matrix):
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(matrix).as_quat()
    return np.array([w, x, y, z])


class GaussianMap:
    LEARNING_RATES = dict(means=0.00016, scales=0.005, quats=0.001, opacities=0.05, sh0=0.0025)

    def __init__(
        self, budget=MAX_GAUSSIANS, *, anchors=ANCHORS, views=VIEWS, seed=0, precision="full"
    ):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Optimized Gaussian mapping needs an NVIDIA CUDA device")
        from gsplat import rasterization

        self.rasterize = rasterization
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
        self.anchors = {}  # node -> dict(slot, pose_cpu); slot 0 is never a node's
        self.views = {}  # node -> View, oldest first
        self.changed = set()
        self.urgent = set()
        self.steps = 0
        self.scene_scale = None
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
            k: torch.optim.Adam([self.p[k]], lr=lr, eps=1e-15)
            for k, lr in self.LEARNING_RATES.items()
        }
        # Per-Gaussian running state, kept in step with the parameters by gsplat's ops.
        self.state = dict(
            grad2d=torch.zeros(0, device="cuda"),
            count=torch.zeros(0, device="cuda"),
            filter=torch.zeros(0, device="cuda"),
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
            pose = self.poses[slot]
            self.p["means"][mine] = self.p["means"][mine] @ pose[:3, :3].T + pose[:3, 3]
            self.p["quats"][mine] = quaternion_product(
                self.rotations[slot : slot + 1], self.p["quats"][mine]
            )
            self.p["owner"][mine] = 0
        for view in self.views.values():
            if view.slot == slot:
                view.relative = self.anchor_pose(slot) @ view.relative
                view.slot = 0
        # Republished under key 0; its own key's records go.
        self.changed.update((WORLD, node))
        self.urgent.update((WORLD, node))

    def anchor_pose(self, slot):
        return self.slot_poses[slot]

    def resize(self, budget):
        self.budget = int(budget)
        self._enforce_budget(self.budget)

    def view_pose(self, view):
        return self.anchor_pose(view.slot) @ view.relative

    def add(self, frame):
        """An allocation failure drops this observation and retains the previous map."""
        torch = self.torch
        parameters = dict(self.p.items())
        state = self.state.copy()
        anchors = {n: a.copy() for n, a in self.anchors.items()}
        views = {n: replace(v) for n, v in self.views.items()}
        poses, rotations, slot_poses = (
            self.poses.clone(),
            self.rotations.clone(),
            self.slot_poses.copy(),
        )
        optimizer_states = {
            k: (o.state.copy(), [g["params"][:] for g in o.param_groups])
            for k, o in self.optimizers.items()
        }
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
            self.state, self.anchors, self.views = state, anchors, views
            self.poses, self.rotations, self.slot_poses = poses, rotations, slot_poses
            with torch.no_grad():
                for name, value in baked.items():
                    self.p[name].copy_(value)
            for key, (saved, groups) in optimizer_states.items():
                optimizer = self.optimizers[key]
                optimizer.state = saved
                for group, values in zip(optimizer.param_groups, groups, strict=True):
                    group["params"] = values
            raise

    def _add(self, frame):
        """A view; a keyframe (anchor == node or None) also seeds where the map is thin."""
        torch = self.torch
        anchor = frame.node if frame.anchor is None else frame.anchor
        pose = validate_pose(frame.pose).astype(np.float64)
        if anchor == frame.node:
            if frame.node in self.anchors:
                self.correct({frame.node: pose})
                return
            if len(self.anchors) >= self.max_anchors - 1:
                self._retire_oldest_anchor()
            used = {a["slot"] for a in self.anchors.values()}
            slot = next(s for s in range(1, self.max_anchors) if s not in used)
            self.anchors[frame.node] = dict(slot=slot, pose_cpu=pose.copy())
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
        view = View(
            slot,
            relative,
            torch.as_tensor(frame.rgb, device="cuda").clone(),
            frame.intrinsics,
            float(frame.captured_at),
            None if frame.depth is None else self.tensor(frame.depth),
            exposure=exposure,
            exposure_optimizer=torch.optim.Adam([exposure], lr=0.001),
        )
        self.views[frame.node] = view
        if frame.points is not None:
            self._seed(frame, slot, pose, relative)

    def _seed(self, frame, slot, pose, relative):
        torch = self.torch
        points = torch.as_tensor(frame.points, device="cuda").detach().reshape(-1, 3).float()
        colors = torch.as_tensor(frame.colors, device="cuda").detach().reshape(-1, 3).float()
        confidence = torch.as_tensor(frame.confidence, device="cuda").detach().reshape(-1)
        valid = (
            torch.isfinite(points).all(-1)
            & torch.isfinite(confidence.float())
            & (confidence.float() > 0)
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
                _, depth, alpha, _ = self.render(pose, i)
            # Expected depth is already normalised by alpha.
            fresh = (alpha[v, u] < SEED_COVERED) | (z < SEED_NEARER * depth[v, u])
            points, colors, z = points[fresh], colors[fresh], z[fresh]
        n = min(len(points), SEED_MAX, self.budget)
        if n < 4:
            return
        ids = torch.as_tensor(
            np.sort(self.generator.choice(len(points), n, replace=False)), device="cuda"
        )
        points, colors, z = points[ids], colors[ids], z[ids]
        # A sparse camera grid supplies spacing; optimization fits all three scales.
        scale = (z / i.fx * math.sqrt(i.width * i.height / n)).clamp(0.002, 0.08)
        self._enforce_budget(self.budget - n)
        rotation = self.tensor(relative[:3, :3])
        values = dict(
            means=points @ rotation.T + self.tensor(relative[:3, 3]),
            scales=scale.log()[:, None].expand(-1, 3).clone(),
            quats=self.tensor(rotation_quaternion(relative[:3, :3]))[None].expand(n, -1),
            opacities=torch.full((n,), math.log(SEED_OPACITY / (1 - SEED_OPACITY)), device="cuda"),
            sh0=((colors - 0.5) / C0)[:, None],
            owner=torch.full((n,), slot, device="cuda", dtype=torch.long),
        )
        self._append(
            values,
            dict(filter=z / i.fx * FILTER, seen=torch.full((n,), frame.captured_at, device="cuda")),
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

    def world(self, train=False):
        """World-space Gaussians with the 3D filter applied (scales and opacity)."""
        torch = self.torch
        p = {k: v if train else v.detach() for k, v in self.p.items()}
        owner = p["owner"]
        pose = self.poses[owner]
        means = torch.bmm(pose[:, :3, :3], p["means"][:, :, None])[:, :, 0] + pose[:, :3, 3]
        quats = quaternion_product(
            self.rotations[owner], torch.nn.functional.normalize(p["quats"], dim=-1)
        )
        scale2 = p["scales"].exp().square()
        filtered = scale2 + self.state["filter"][:, None].square()
        opacity = p["opacities"].sigmoid() * torch.sqrt(scale2.prod(-1) / filtered.prod(-1))
        return dict(means=means, quats=quats, scales=filtered.sqrt(), opacities=opacity)

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
            viewmats=torch.linalg.inv(pose @ self.tensor(CV))[None],
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

    def _pick(self):
        nodes = list(self.views)
        newest = self.views[nodes[-1]]
        if newest.steps < 5:
            return nodes[-1]
        recent = [n for n in nodes if self.views[n].captured_at > newest.captured_at - RECENT]
        pool = recent if self.generator.random() < 0.5 else nodes
        return pool[int(self.generator.integers(len(pool)))]

    def step(self):
        if not self.count or not self.views:
            return
        torch = self.torch
        node = self._pick()
        view = self.views[node]
        view.steps += 1
        target = view.image.to(self.loss_dtype) / 255
        pose = self.poses[view.slot].double() @ torch.as_tensor(view.relative, device="cuda")
        pred, depth, alpha, info = self.render(pose.float(), view.intrinsics, train=True)
        info["means2d"].retain_grad()
        corrected = (pred * view.exposure[0].exp() + view.exposure[1]).to(self.loss_dtype)
        loss = 0.8 * (corrected - target).abs().mean() + 0.01 * view.exposure.square().mean()
        if min(target.shape[:2]) >= 11:
            with torch.autocast("cuda", dtype=self.loss_dtype, enabled=self.precision == "half"):
                loss = loss + 0.2 * (1 - ssim(corrected, target).float())
        if view.depth is not None:
            mask = torch.isfinite(view.depth) & (view.depth > 0) & (alpha > 0.5)
            if mask.any():
                loss = (
                    loss
                    + 0.05
                    * (
                        (depth[mask] - view.depth[mask]).abs() / view.depth[mask].clamp_min(0.1)
                    ).mean()
                )
        loss.backward()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite Gaussian training loss")
        self._accumulate(info, view.captured_at)
        for optimizer in self.optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        view.exposure_optimizer.step()
        view.exposure_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            view.exposure.clamp_(-0.3, 0.3)
        self.steps += 1
        if self.steps % REFINE_EVERY == 0:
            self._refine()
        self.changed.update(self.anchors)
        self.changed.add(WORLD)

    def _accumulate(self, info, captured_at):
        grads = info["means2d"].grad.clone()
        grads[..., 0] *= info["width"] / 2.0
        grads[..., 1] *= info["height"] / 2.0
        seen = (info["radii"] > 0).all(-1)[0]
        self.state["grad2d"][seen] += grads[0][seen].norm(dim=-1)
        self.state["count"][seen] += 1
        self.state["seen"][seen] = self.state["seen"][seen].clamp_min(captured_at)

    def _refine(self):
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
        self.changed.update(self.anchors)
        self.changed.add(WORLD)

    def _update_filter(self):
        """Mip-Splatting: each Gaussian's smallest footprint over (a sample of) the views."""
        torch = self.torch
        world = self.world()["means"]
        closest = torch.full((len(world),), float("inf"), device="cuda")
        nodes = list(self.views)
        for node in nodes[:: max(1, len(nodes) // 48)]:
            view = self.views[node]
            i = view.intrinsics
            camera = np.linalg.inv(self.view_pose(view) @ CV)
            local = world @ self.tensor(camera[:3, :3]).T + self.tensor(camera[:3, 3])
            z = local[:, 2]
            u = i.fx * local[:, 0] / z + i.cx
            v = i.fy * local[:, 1] / z + i.cy
            inside = (
                (z > 0.01)
                & (u > -0.15 * i.width)
                & (u < 1.15 * i.width)
                & (v > -0.15 * i.height)
                & (v < 1.15 * i.height)
            )
            closest = torch.minimum(closest, torch.where(inside, z / i.fx, float("inf")))
        seen = torch.isfinite(closest)
        self.state["filter"] = torch.where(seen, closest * FILTER, self.state["filter"])

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
