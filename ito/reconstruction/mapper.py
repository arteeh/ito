"""Keyframe-local Gaussian map. Poses and points use Ito axes (+Y up, -Z forward).

The engine is single-owner: the background worker calls add/correct/forget/step;
tracking submits immutable Keyframes through OnlineMapper. Rendering and alignment
are also exposed here for offline evaluation. SH degree 0 is intentional: the live
ring has one RGB coefficient, and colour then needs no rotation on pose correction.
"""

from dataclasses import dataclass, replace

import numpy as np

from ito.protocol import Intrinsics
from ito.render.pose import validate_pose

MAX_GAUSSIANS = 400_000
C0 = 0.28209479177387814


@dataclass(frozen=True)
class Keyframe:
    node: int
    rgb: np.ndarray  # uint8 camera-resolution H,W,3, owned by the caller until submit
    pose: np.ndarray  # world-from-camera
    intrinsics: Intrinsics
    points: object  # N,3 metric camera-local points (numpy or torch)
    colors: object  # N,3 uint8, registered to points
    confidence: object  # N; zero excludes, positive finite values admit a point
    captured_at: float
    depth: np.ndarray | None = None  # optional registered axial metres

    def __post_init__(self):
        i = self.intrinsics
        if self.rgb.dtype != np.uint8 or self.rgb.shape != (i.height, i.width, 3):
            raise ValueError("Keyframe RGB must match camera intrinsics")
        validate_pose(self.pose)
        if not np.isfinite(self.captured_at):
            raise ValueError("Keyframe capture time must be finite")
        if self.depth is not None and self.depth.shape != self.rgb.shape[:2]:
            raise ValueError("Keyframe depth must be registered to RGB")


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


def pose_delta(delta):
    """Camera-local rotation vector (rad), translation (m); finite derivative at zero."""
    import torch

    x, y, z = delta[:3].unbind()
    zero = x * 0
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    result = torch.eye(4, device=delta.device, dtype=delta.dtype)
    result[:3, :3] = torch.matrix_exp(skew)
    result[:3, 3] = delta[3:]
    return result


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


class GaussianMap:
    def __init__(self, budget=MAX_GAUSSIANS, *, max_keyframes=48):
        import torch

        from .gaussian_runtime import prepare

        self.rasterize = prepare()
        self.torch = torch
        self.budget = min(int(budget), MAX_GAUSSIANS)
        if self.budget < 1 or max_keyframes < 1:
            raise ValueError("Mapper budgets must be positive")
        self.max_keyframes = max_keyframes
        self.anchors = {}
        self.changed = set()
        self.urgent = set()
        self.structure_changed = False
        self.steps = self.cursor = 0
        self.p = torch.nn.ParameterDict()
        self.optimizer = None
        self.owners = torch.empty(0, device="cuda", dtype=torch.long)
        self.poses = torch.eye(4, device="cuda").repeat(max_keyframes, 1, 1)
        self.rotations = torch.zeros((max_keyframes, 4), device="cuda")
        self.rotations[:, 0] = 1

    def tensor(self, value):
        return self.torch.as_tensor(value, device="cuda", dtype=self.torch.float32)

    @property
    def count(self):
        return len(self.owners)

    def _install(self, values, origins, owners):
        """Resize parameters and Adam moments together; new rows start with zero momentum."""
        torch = self.torch
        old = self.p
        self.p = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(v.detach()) for k, v in values.items()}
        )
        if self.optimizer is None:
            rates = dict(means=0.00016, scales=0.005, quats=0.001, opacities=0.05, colors=0.0025)
            self.optimizer = torch.optim.Adam(
                [{"params": [v], "lr": rates[k], "name": k} for k, v in self.p.items()], eps=1e-15
            )
        else:
            for group in self.optimizer.param_groups:
                k = group["name"]
                state = self.optimizer.state.pop(old[k], {})
                for name, value in state.items():
                    if isinstance(value, torch.Tensor) and value.ndim:
                        resized = value.new_zeros((len(origins), *value.shape[1:]))
                        existing = origins >= 0
                        resized[existing] = value[origins[existing]]
                        state[name] = resized
                group["params"] = [self.p[k]]
                self.optimizer.state[self.p[k]] = state
        self.owners = owners

    def correct(self, poses):
        from scipy.spatial.transform import Rotation

        for node, pose in poses.items():
            if node not in self.anchors:
                continue
            pose = validate_pose(pose)
            a = self.anchors[node]
            if np.array_equal(a.get("pose_cpu"), pose):
                continue
            a["pose_cpu"] = np.array(pose, copy=True)
            slot = a["slot"]
            self.poses[slot] = self.tensor(pose)
            q = Rotation.from_matrix(pose[:3, :3]).as_quat()
            self.rotations[slot] = self.tensor(q[[3, 0, 1, 2]])
            self.changed.add(node)
            self.urgent.add(node)

    def forget(self, nodes):
        torch = self.torch
        slots = []
        for node in nodes:
            a = self.anchors.pop(node, None)
            if a is not None:
                slots.append(a["slot"])
                self.changed.add(node)
                self.urgent.add(node)
        if slots:
            keep = ~torch.isin(self.owners, torch.tensor(slots, device="cuda"))
            origins = torch.where(keep)[0]
            self._install({k: v[keep] for k, v in self.p.items()}, origins, self.owners[keep])

    def resize(self, budget):
        self.budget = min(int(budget), MAX_GAUSSIANS)
        while self.count > self.budget:
            self.forget([next(iter(self.anchors))])

    def add(self, frame):
        torch = self.torch
        if frame.node in self.anchors:
            self.correct({frame.node: frame.pose})
            return
        points = self.tensor(frame.points).detach().reshape(-1, 3)
        colors = self.tensor(frame.colors).detach().reshape(-1, 3) / 255
        confidence = self.tensor(frame.confidence).detach().reshape(-1)
        valid = (
            torch.isfinite(points).all(-1)
            & torch.isfinite(colors).all(-1)
            & torch.isfinite(confidence)
            & (confidence > 0)
            & (points[:, 2] < -0.01)
        )
        points, colors = points[valid], colors[valid]
        if not len(points):
            return
        n = min(len(points), 6000, self.budget)
        ids = torch.linspace(0, len(points) - 1, n, device="cuda").long()
        points, colors = points[ids], colors[ids]
        while self.anchors and (
            self.count + n > self.budget or len(self.anchors) >= self.max_keyframes
        ):
            self.forget([next(iter(self.anchors))])
        used = {a["slot"] for a in self.anchors.values()}
        slot = next(s for s in range(self.max_keyframes) if s not in used)
        footprint = (-points[:, 2] / frame.intrinsics.fx * 2).clamp(0.0001, 0.2)
        quats = torch.zeros((n, 4), device="cuda")
        quats[:, 0] = 1
        new = dict(
            means=points.clone(),
            scales=footprint[:, None].expand(-1, 3).log().clone(),
            quats=quats,
            opacities=torch.full((n,), 0.0, device="cuda"),
            colors=colors,
        )
        values = {
            k: torch.cat((self.p[k].detach(), v)) if k in self.p else v for k, v in new.items()
        }
        origins = torch.cat(
            (
                torch.arange(self.count, device="cuda"),
                torch.full((n,), -1, device="cuda", dtype=torch.long),
            )
        )
        owners = torch.cat((self.owners, torch.full((n,), slot, device="cuda", dtype=torch.long)))
        self._install(values, origins, owners)
        exposure = torch.nn.Parameter(torch.zeros((2, 3), device="cuda"))
        self.anchors[frame.node] = dict(
            slot=slot,
            exposure=exposure,
            exposure_optimizer=torch.optim.Adam([exposure], lr=0.001),
            frame=replace(frame, points=None, colors=None, confidence=None),
        )
        self.correct({frame.node: frame.pose})

    def world(self, train_node=None):
        torch = self.torch
        p = {k: v if train_node is not None else v.detach() for k, v in self.p.items()}
        pose = self.poses[self.owners]
        means = torch.bmm(p["means"][:, None], pose[:, :3, :3].transpose(1, 2))[:, 0]
        return dict(
            means=means + pose[:, :3, 3],
            quats=quaternion_product(
                self.rotations[self.owners], torch.nn.functional.normalize(p["quats"], dim=-1)
            ),
            scales=p["scales"].exp(),
            opacities=p["opacities"].sigmoid(),
            colors=p["colors"],
        )

    def render(self, pose, intrinsics, delta=None, *, train_node=None, world=None):
        """GPU colour, expected axial depth, alpha; differentiable in camera-local delta."""
        torch = self.torch
        i = intrinsics
        pose = self.tensor(pose)
        if delta is not None:
            pose = pose @ pose_delta(delta)
        cv = torch.diag(self.tensor([1, -1, -1, 1]))
        K = self.tensor([[i.fx, 0, i.cx], [0, i.fy, i.cy], [0, 0, 1]])
        if not self.count and world is None:
            zero = torch.zeros((i.height, i.width), device="cuda")
            return zero[..., None].expand(-1, -1, 3), zero, zero
        pred, alpha, _ = self.rasterize(
            **(self.world(train_node) if world is None else world),
            viewmats=torch.linalg.inv(pose @ cv)[None],
            Ks=K[None],
            width=i.width,
            height=i.height,
            packed=False,
            near_plane=0.01,
            far_plane=1000,
            rasterize_mode="antialiased",
            render_mode="RGB+ED",
        )
        return pred[0, ..., :3], pred[0, ..., 3], alpha[0, ..., 0]

    def step(self):
        if not self.count:
            return
        torch = self.torch
        nodes = list(self.anchors)
        node = nodes[-1] if self.steps % 3 else nodes[self.cursor % len(nodes)]
        if self.steps % 3 == 0:
            self.cursor += 1
        a = self.anchors[node]
        f = a["frame"]
        target = self.tensor(f.rgb) / 255
        pred, depth, _ = self.render(self.poses[a["slot"]], f.intrinsics, train_node=node)
        corrected = pred * a["exposure"][0].exp() + a["exposure"][1]
        loss = 0.8 * (corrected - target).abs().mean() + 0.01 * a["exposure"].square().mean()
        if min(target.shape[:2]) >= 11:
            loss = loss + 0.2 * (1 - ssim(corrected, target))
        if f.depth is not None:
            measured = self.tensor(f.depth)
            mask = torch.isfinite(measured) & (measured > 0)
            if mask.any():
                loss = (
                    loss
                    + 0.05
                    * ((depth[mask] - measured[mask]).abs() / measured[mask].clamp_min(0.1)).mean()
                )
        self.optimizer.zero_grad(set_to_none=True)
        a["exposure_optimizer"].zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite Gaussian training loss")
        grad = self.p["means"].grad.detach().norm(dim=-1)
        self.optimizer.step()
        a["exposure_optimizer"].step()
        with torch.no_grad():
            self.p["scales"].clamp_(-10, 0)
            self.p["opacities"].clamp_(-10, 10)
            self.p["colors"].clamp_(0, 1)
            self.p["quats"].copy_(torch.nn.functional.normalize(self.p["quats"], dim=-1))
            a["exposure"].clamp_(-0.3, 0.3)
        self.steps += 1
        if self.steps % 100 == 0:
            self._refine(grad)
        self.changed.update(nodes)

    def _refine(self, grad):
        self.structure_changed = True
        torch = self.torch
        with torch.no_grad():
            p = self.p
            keep = p["opacities"].sigmoid() > 0.01
            room = max(0, self.budget - int(keep.sum()))
            n = min(room, max(1, len(grad) // 10))
            candidates = torch.where(keep & (grad > 0.0002))[0]
            ids = candidates[torch.topk(grad[candidates], min(n, len(candidates))).indices]
            values = {k: torch.cat((v[keep], v[ids])).clone() for k, v in p.items()}
            if len(ids):
                values["means"][-len(ids) :] += torch.randn_like(p["means"][ids]) * (
                    p["scales"][ids].exp() * 0.3
                )
                values["scales"][-len(ids) :] -= np.log(1.6)
                values["opacities"][-len(ids) :] -= 0.7
            origins = torch.cat((torch.where(keep)[0], torch.full_like(ids, -1)))
            owners = torch.cat((self.owners[keep], self.owners[ids]))
            self._install(values, origins, owners)

    def records(self, node):
        torch = self.torch
        if node not in self.anchors:
            return np.empty((0, 4, 4), np.float32)
        slot = self.anchors[node]["slot"]
        with torch.no_grad():
            selected = self.owners == slot
            p = {k: v[selected] for k, v in self.p.items()}
            pose = self.poses[slot]
            r = torch.zeros((len(p["means"]), 4, 4), device="cuda")
            r[:, 0, :3] = p["means"] @ pose[:3, :3].T + pose[:3, 3]
            r[:, 0, 3] = p["opacities"].sigmoid()
            r[:, 1, :3] = p["scales"].exp()
            r[:, 1, 3] = 1e30
            r[:, 2] = quaternion_product(self.rotations[slot : slot + 1], p["quats"])
            r[:, 3, :3] = (p["colors"] - 0.5) / C0
            return r.cpu().numpy()

    def align(self, rgb, pose0, intrinsics, *, steps=30):
        """Return (world-from-camera, sigma[rotation rad, translation m], inlier pixels).

        Frozen map, robust photometric fit. Degenerate/unsupported fits return the input
        with infinite sigma and zero inliers. Sigma is a local linearized estimate,
        not a calibrated sensor uncertainty; fusion must impose its own noise floor.
        """
        torch = self.torch
        failure = (np.array(pose0, copy=True), np.full(6, np.inf), 0)
        if not self.anchors:
            return failure
        world = self.world()
        target = self.tensor(rgb) / 255
        base = self.tensor(validate_pose(pose0))
        delta = torch.zeros(6, device="cuda", requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=0.001)
        with torch.no_grad():
            initial, _, alpha = self.render(base, intrinsics, world=world)
            mask = alpha > 0.7
            if int(mask.sum()) < 128:
                return failure
            initial_loss = (initial[mask] - target[mask]).abs().mean()
        for _ in range(steps):
            pred, _, alpha = self.render(base, intrinsics, delta, world=world)
            residual = pred[mask] - target[mask]
            loss = (residual.square() + 0.01**2).sqrt().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                delta[:3].clamp_(-0.1, 0.1)
                delta[3:].clamp_(-0.2, 0.2)
        with torch.no_grad():
            pred, _, alpha = self.render(base, intrinsics, delta, world=world)
            residual = pred - target
            inliers = mask & (alpha > 0.7) & (residual.abs().mean(-1) < 0.1)
            n = int(inliers.sum())
            if n < 128 or residual[mask].abs().mean() > initial_loss:
                return failure
            # Six finite differences avoid unsupported second derivatives of the CUDA
            # rasterizer. Sample a bounded set of pixels for the 6x6 information matrix.
            ids = torch.where(inliers.flatten())[0][:: max(1, n // 2048)]
            jac = []
            for axis in range(6):
                offset = delta.detach().clone()
                offset[axis] += 0.0001
                shifted = self.render(base, intrinsics, offset, world=world)[0]
                jac.append(((shifted - pred).reshape(-1, 3)[ids] / 0.0001).flatten())
            J = torch.stack(jac, dim=1).double()
            H = J.T @ J
            eigen = torch.linalg.eigvalsh(H)
            if not torch.isfinite(H).all() or eigen[0] < eigen[-1] * 1e-6:
                return failure
            variance = residual.reshape(-1, 3)[ids].square().mean().double().clamp_min(1e-6)
            sigma = (torch.linalg.inv(H).diagonal() * variance).sqrt()
            return ((base @ pose_delta(delta)).cpu().numpy(), sigma.cpu().numpy(), n)
