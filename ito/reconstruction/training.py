"""Photometric refinement over whole retained views, the way MonoGS and MoonSplat map.

LocalFit polishes 32-pixel patches and freezes settled surfaces. A ViewFit unit takes one
retained view and the views overlapping it, every Gaussian any of them sees, and trains them
together against those photos for STEPS Adam steps: colour, opacity, shape and position, with
the views' agreed depth as a prior so geometry moves only where the photos say so. Where the
photos keep pulling at a Gaussian it splits or clones (more detail), where it fades it goes.

The CPU map stays the source of truth: a unit uploads its Gaussians, trains them on CUDA and
writes back only those still live and visibly changed, which then publish like any update.
"""

import numpy as np

from .gaussian import AXES, C0

STEPS = 30
PARTNERS = 2  # overlapping views trained beside the chosen one
MAX_UNIT = 400_000
SSIM_WEIGHT = 0.2
DEPTH_WEIGHT = 0.1
SCALE_WEIGHT = 0.01  # flatness: a surface's Gaussians stay thin across it
RATES = dict(means=2e-4, scales=5e-3, quats=1e-3, opacity=5e-2, color=2.5e-3)
GROW_GRADIENT = 8e-4  # mean absolute screen-space gradient (normalised) asking for detail
GROW_SHARE = 0.05  # at most this share of a unit's Gaussians is cloned or split per unit
SPLIT_PIXELS = 6.0  # a Gaussian larger than this in its view splits; smaller ones clone
PRUNE_OPACITY = 0.01
CHANGED = (1e-4, 1 / 255)  # metres, colour: smaller changes are not worth publishing


def ssim(a, b, torch):
    """Mean structural similarity of two (3, H, W) images, 11-tap Gaussian window."""
    x = torch.arange(11, device=a.device, dtype=a.dtype) - 5
    g = torch.exp(-(x**2) / (2 * 1.5**2))
    g = g / g.sum()
    window = (g[:, None] * g[None, :]).expand(3, 1, 11, 11).contiguous()

    def blur(v):
        return torch.nn.functional.conv2d(v[None], window, padding=5, groups=3)[0]

    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a**2
    var_b = blur(b * b) - mu_b**2
    cov = blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    value = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    )
    return value.mean()


class ViewFit:
    def __init__(self, backend):
        import torch
        from gsplat import rasterization

        self.torch = torch
        self.rasterize = rasterization
        self.backend = backend
        self.active = None
        self.serial = 0
        self.units = 0
        self.grown = 0
        self.pruned = 0

    def tensor(self, value, dtype=None):
        torch = self.torch
        return torch.as_tensor(value, device="cuda", dtype=dtype or torch.float32)

    def choose(self):
        b = self.backend
        views = list(b.views)
        recent = views[-3:]
        pool = recent if self.serial % 4 else views
        self.serial += 1
        return min(pool, key=lambda v: v.steps)

    def partners(self, view):
        b = self.backend
        pose = view.fitting_pose if view.fitting_pose is not None else view.pose
        scored = []
        for other in b.views:
            if other is view:
                continue
            o = other.fitting_pose if other.fitting_pose is not None else other.pose
            cosine = float(o[:3, 2] @ pose[:3, 2])
            distance = float(np.linalg.norm(o[:3, 3] - pose[:3, 3]))
            if cosine > 0.7 and distance < 0.8:
                scored.append((other.steps + 50 * (1 - cosine) + 20 * distance, other))
        scored.sort(key=lambda item: item[0])
        return [other for _, other in scored[:PARTNERS]]

    def prepare(self):
        b = self.backend
        if not b.views or not b.count:
            return False
        view = self.choose()
        targets = [view, *self.partners(view)]
        chosen = []
        for observation in targets:
            pose = (
                observation.fitting_pose
                if observation.fitting_pose is not None
                else observation.pose
            )
            ids, *_ = b.centers(pose, observation.intrinsics, observation.depth.shape)
            chosen.append(ids)
        ids = np.unique(np.concatenate(chosen))
        if not len(ids):
            view.steps += STEPS
            return False
        if len(ids) > MAX_UNIT:
            ids = np.sort(np.random.default_rng(self.serial).choice(ids, MAX_UNIT, replace=False))
        torch = self.torch
        original = self.tensor(b.records[ids].copy())
        p = {
            "means": torch.nn.Parameter(original[:, 0, :3].clone()),
            "scales": torch.nn.Parameter(original[:, 1, :3].clamp_min(1e-6).log()),
            "quats": torch.nn.Parameter(original[:, 2].clone()),
            "opacity": torch.nn.Parameter(torch.logit(original[:, 0, 3].clamp(0.001, 0.999))),
            "color": torch.nn.Parameter(original[:, 3, :3].clone()),
        }
        optimizer = torch.optim.Adam(
            [{"params": [value], "lr": RATES[key]} for key, value in p.items()], eps=1e-15
        )
        prepared = []
        for observation in targets:
            pose = (
                observation.fitting_pose
                if observation.fitting_pose is not None
                else observation.pose
            )
            h, w = observation.depth.shape
            fx, fy, cx, cy = observation.intrinsics
            prepared.append(
                dict(
                    view=observation,
                    camera=self.tensor(np.linalg.inv(pose @ AXES))[None],
                    calibration=self.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])[None],
                    target=self.tensor(observation.image).permute(2, 0, 1) / 255,
                    depth=self.tensor(observation.depth),
                    size=(w, h),
                    focal=fx,
                )
            )
        self.active = dict(
            ids=ids,
            keys=b.keys[ids].copy(),
            original=original,
            p=p,
            optimizer=optimizer,
            targets=prepared,
            steps=0,
            gradient=torch.zeros(len(ids), device="cuda"),
            seen=torch.zeros(len(ids), device="cuda"),
            pixels=torch.zeros(len(ids), device="cuda"),
        )
        return True

    def step(self):
        b, torch = self.backend, self.torch
        if self.active is None and not self.prepare():
            return False
        a = self.active
        ids, p = a["ids"], a["p"]
        if ids.max() >= len(b.keys):
            self.active = None
            return False
        target = a["targets"][a["steps"] % len(a["targets"])]
        w, h = target["size"]
        with b.gpu_chunk("fit_forward"):
            rendered, alpha, info = self.rasterize(
                means=p["means"],
                quats=p["quats"],
                scales=p["scales"].exp(),
                opacities=p["opacity"].sigmoid(),
                colors=p["color"][:, None],
                sh_degree=0,
                viewmats=target["camera"],
                Ks=target["calibration"],
                width=w,
                height=h,
                packed=False,
                near_plane=0.01,
                far_plane=1000,
                render_mode="RGB+ED",
                absgrad=True,
            )
            info["means2d"].retain_grad()
            rgb = rendered[0, ..., :3].permute(2, 0, 1)
            observed = target["target"]
            l1 = (rgb - observed).abs().mean()
            loss = (1 - SSIM_WEIGHT) * l1 + SSIM_WEIGHT * (1 - ssim(rgb, observed, torch))
            known = target["depth"]
            drawn = rendered[0, ..., 3]
            mask = (known > 0) & (alpha[0, ..., 0].detach() > 0.5)
            if mask.any():
                error = (drawn[mask] - known[mask]).abs() / known[mask]
                loss = loss + DEPTH_WEIGHT * error.mean()
            scales = p["scales"].exp()
            thinnest = scales.min(-1).values
            loss = loss + SCALE_WEIGHT * (thinnest / scales.max(-1).values.clamp_min(1e-6)).mean()
        with b.gpu_chunk("fit_backward"):
            loss.backward()
            with torch.no_grad():
                grad = info["means2d"].absgrad[0] if hasattr(info["means2d"], "absgrad") else None
                if grad is None:
                    grad = info["means2d"].grad[0]
                visible = info["radii"][0] > 0
                if visible.dim() > 1:
                    visible = visible.all(-1)
                # Normalised to the image, as 3DGS densifies.
                norm = (grad * torch.tensor([w / 2, h / 2], device="cuda")).norm(dim=-1)
                a["gradient"] += torch.where(visible, norm, 0)
                a["seen"] += visible
                radius = info["radii"][0]
                if radius.dim() > 1:
                    radius = radius.max(-1).values
                a["pixels"] = torch.maximum(a["pixels"], radius.float())
            a["optimizer"].step()
            a["optimizer"].zero_grad(set_to_none=True)
            with torch.no_grad():
                p["quats"].copy_(torch.nn.functional.normalize(p["quats"], dim=-1))
                p["opacity"].clamp_(-7, 7)
                p["color"].clamp_(-0.5 / C0, 0.5 / C0)
        target["view"].steps += 1
        b.fit_steps += 1
        a["steps"] += 1
        if a["steps"] < STEPS:
            return True
        self.active = None
        self.finish(a)
        return True

    def finish(self, a):
        b, torch = self.backend, self.torch
        ids, p = a["ids"], a["p"]
        with b.gpu_chunk("fit_readback"), torch.no_grad():
            records = a["original"].clone()
            records[:, 0, :3] = p["means"]
            records[:, 0, 3] = p["opacity"].sigmoid()
            records[:, 1, :3] = p["scales"].exp()
            records[:, 2] = p["quats"]
            records[:, 3, :3] = p["color"]
            gradient = (a["gradient"] / a["seen"].clamp_min(1)).cpu().numpy()
            pixels = a["pixels"].cpu().numpy()
            records = records.cpu().numpy()
        self.units += 1
        live = (ids < len(b.keys)) & ~b.retiring[np.minimum(ids, len(b.keys) - 1)]
        live &= b.keys[np.minimum(ids, len(b.keys) - 1)] == a["keys"]
        live &= np.isfinite(records).all(axis=(1, 2))
        old = b.records[ids]
        moved = np.linalg.norm(records[:, 0, :3] - old[:, 0, :3], axis=1) > CHANGED[0]
        recoloured = np.abs(records[:, 3, :3] - old[:, 3, :3]).max(-1) * C0 > CHANGED[1]
        reshaped = (
            np.abs(np.log(records[:, 1, :3] / np.maximum(old[:, 1, :3], 1e-9))).max(-1) > 0.02
        )
        refaded = np.abs(records[:, 0, 3] - old[:, 0, 3]) > 0.02
        changed = live & (moved | recoloured | reshaped | refaded)
        keep = old[:, 1, 3]
        records[:, 1, 3] = keep
        records[:, 3, 3] = old[:, 3, 3]
        written = ids[changed]
        b.records[written] = records[changed]
        b.dirty[written] = True
        for target in a["targets"]:
            view = target["view"]
            older = b.captures[written] > view.captured
            b.captures[written[older]] = view.captured
            b.sequences[written[older]] = view.sequence
        faded = ids[live & (records[:, 0, 3] < PRUNE_OPACITY)]
        if len(faded):
            b.retire(faded)
            self.pruned += len(faded)
        self.grow(ids[live], records[live], gradient[live], pixels[live])

    def grow(self, ids, records, gradient, pixels):
        """Clone small Gaussians the photos keep pulling at; split large ones in two."""
        b = self.backend
        wanted = np.flatnonzero((gradient > GROW_GRADIENT) & (records[:, 0, 3] > 0.05))
        if not len(wanted):
            return
        limit = max(1, int(GROW_SHARE * len(ids)))
        wanted = wanted[np.argsort(-gradient[wanted])[:limit]]
        free = np.flatnonzero(b.keys[: b.budget] < 0)
        room = min(len(wanted), len(free), max(0, b.budget - b.count))
        if room <= 0:
            return
        wanted = wanted[:room]
        parents = ids[wanted]
        children = records[wanted].copy()
        split = pixels[wanted] > SPLIT_PIXELS
        if split.any():
            # Two halves of the parent along its longest axis, each at 1/1.6 the size.
            from scipy.spatial.transform import Rotation

            quat = children[split, 2]
            rotation = Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_matrix()
            scales = children[split, 1, :3]
            axis = np.argmax(scales, axis=1)
            offset = (
                rotation[np.arange(len(axis)), :, axis]
                * scales[np.arange(len(axis)), axis][:, None]
            )
            parent_records = b.records[parents[split]]
            parent_records[:, 0, :3] -= 0.5 * offset
            parent_records[:, 1, :3] /= 1.6
            b.records[parents[split]] = parent_records
            b.dirty[parents[split]] = True
            children[split, 0, :3] += 0.5 * offset
            children[split, 1, :3] /= 1.6
        slots = free[:room]
        b.records[slots] = children
        b.keys[slots] = np.arange(b.next_key, b.next_key + room)
        b.next_key += room
        b.owners[slots] = b.owners[parents]
        b.support[slots] = b.support[parents]
        b.uncertainty[slots] = b.uncertainty[parents]
        b.contribution[slots] = 1
        b.contradictions[slots] = 0
        b.seen[slots] = b.seen[parents]
        b.captures[slots] = b.captures[parents]
        b.sequences[slots] = b.sequences[parents]
        b.dirty[slots] = True
        self.grown += room
