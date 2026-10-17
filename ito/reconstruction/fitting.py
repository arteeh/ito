"""Small photometric fits, using the same anisotropic Gaussian records as the display.

A unit trains one patch of a retained view, the one that disagrees most with the map, plus
a nearby view of it, for STEPS Adam steps. Only that crop and at most one publication
packet of parameters live on CUDA; the CPU map takes the result whole after readback.
Recent views get most units, so provisional surfaces are refined first.
"""

import numpy as np

from .gaussian import C0

PATCH = 32
HALO = 6
MAX_LOCAL = 4096
STEPS = 12


class LocalFit:
    def __init__(self, backend):
        import torch
        from gsplat import rasterization

        self.torch = torch
        self.rasterize = rasterization
        self.backend = backend
        self.active = None
        self.serial = 0
        self.dtype = (
            torch.float32
            if backend.precision == "full"
            else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        )

    def tensor(self, value):
        return self.torch.as_tensor(value, device="cuda", dtype=self.torch.float32)

    def prepare(self):
        b = self.backend
        if not b.views or not b.count:
            return False
        # Recent views get three quarters of the work; archived views still constrain
        # a surface after its estimator node has left the graph.
        recent = list(b.views)[-3:]
        pool = recent if self.serial % 4 else list(b.views)
        view = min(pool, key=lambda v: v.steps)
        pose = view.pose if view.fitting_pose is None else view.fitting_pose
        h, w = view.image.shape[:2]
        cols = (w + PATCH - 1) // PATCH
        rows = (h + PATCH - 1) // PATCH
        ids = np.flatnonzero((b.keys >= 0) & ~b.retiring)
        local = (b.records[ids, 0, :3] - pose[:3, 3]) @ pose[:3, :3]
        z = -local[:, 2]
        fx, fy, cx, cy = view.intrinsics
        u = cx + fx * local[:, 0] / np.maximum(z, 0.01)
        v = cy - fy * local[:, 1] / np.maximum(z, 0.01)
        radius = 3 * fx * b.records[ids, 1, :3].max(-1) / np.maximum(z, 0.01) + 2
        inside = (z > 0.01) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        visible = np.flatnonzero(inside)
        if not len(visible):
            return False
        px, py = u[visible].astype(int), v[visible].astype(int)
        pixel = py * w + px
        nearest = np.full(h * w, np.inf, z.dtype)
        np.minimum.at(nearest, pixel, z[visible])
        front = z[visible] == nearest[pixel]
        visible, px, py = visible[front], px[front], py[front]
        error = np.abs(b.records[ids[visible], 3, :3] * C0 + 0.5 - view.image[py, px] / 255).mean(
            -1
        )
        provisional = 1 / np.sqrt(b.support[ids[visible]] + 1)
        tile_ids = (py // PATCH) * cols + px // PATCH
        priority = np.bincount(
            tile_ids,
            weights=error + 0.1 * b.uncertainty[ids[visible]] + 0.02 * provisional,
            minlength=cols * rows,
        )
        samples = np.bincount(tile_ids, minlength=cols * rows)
        if view.tile_steps is None:
            view.tile_steps = np.zeros(cols * rows, np.uint32)
        priority /= np.maximum(samples, 1) * np.sqrt(view.tile_steps + 1)
        tile = int(np.argmax(priority))
        x, y = (tile % cols) * PATCH, (tile // cols) * PATCH
        patch = PATCH
        while True:
            left, top = max(0, x - HALO), max(0, y - HALO)
            right, bottom = min(w, x + patch + HALO), min(h, y + patch + HALO)
            selected = (z > 0.01) & (u + radius >= left) & (u - radius < right)
            selected &= (v + radius >= top) & (v - radius < bottom)
            if selected.sum() <= MAX_LOCAL or patch <= 8:
                break
            patch //= 2
            # Rotate through subtiles instead of starving a dense surface forever.
            subdivisions = PATCH // patch
            part = int(view.tile_steps[tile]) % subdivisions**2
            x = (tile % cols) * PATCH + (part % subdivisions) * patch
            y = (tile // cols) * PATCH + (part // subdivisions) * patch
        ids, distance = ids[selected], z[selected]
        view.tile_steps[tile] += 1
        self.serial += 1
        if not len(ids) or len(ids) > MAX_LOCAL:
            view.steps += STEPS
            return False
        torch = self.torch
        original = self.tensor(b.records[ids].copy())
        p = {
            "means": torch.nn.Parameter(original[:, 0, :3].clone()),
            "scales": torch.nn.Parameter(original[:, 1, :3].log()),
            "quats": torch.nn.Parameter(original[:, 2].clone()),
            "opacity": torch.nn.Parameter(torch.logit(original[:, 0, 3].clamp(0.001, 0.999))),
            "color": torch.nn.Parameter(original[:, 3, :3].clone()),
        }
        rates = dict(means=0.0002, scales=0.003, quats=0.001, opacity=0.02, color=0.02)
        optimizer = torch.optim.Adam(
            [{"params": [value], "lr": rates[key]} for key, value in p.items()], fused=True
        )
        nearby = [
            v
            for v in b.views
            if v is not view
            and v.pose[:3, 2] @ view.pose[:3, 2] > 0.94
            and np.linalg.norm(v.pose[:3, 3] - view.pose[:3, 3]) < 0.3
        ]
        other = min(nearby, key=lambda v: v.steps, default=None)
        targets = []
        center = np.median(b.records[ids, 0, :3], axis=0)
        for observation in [view] + ([] if other is None else [other]):
            fitted_camera = (
                observation.pose if observation.fitting_pose is None else observation.fitting_pose
            )
            ih, iw = observation.depth.shape
            ox, oy, ocx, ocy = observation.intrinsics
            if observation is view:
                tx, ty = x, y
            else:
                local_center = (center - fitted_camera[:3, 3]) @ fitted_camera[:3, :3]
                if local_center[2] >= -0.01:
                    continue
                tx = int(ocx - ox * local_center[0] / local_center[2]) - patch // 2
                ty = int(ocy + oy * local_center[1] / local_center[2]) - patch // 2
                tx, ty = np.clip(tx, 0, iw - patch), np.clip(ty, 0, ih - patch)
            left, top = max(0, tx - HALO), max(0, ty - HALO)
            right, bottom = min(iw, tx + patch + HALO), min(ih, ty + patch + HALO)
            camera = np.linalg.inv(fitted_camera @ np.diag([1.0, -1.0, -1.0, 1.0]))
            calibration = [[ox, 0, ocx - left], [0, oy, ocy - top], [0, 0, 1]]
            target = self.tensor(observation.image[top:bottom, left:right]) / 255
            depth = self.tensor(observation.depth[top:bottom, left:right])
            interior = torch.zeros_like(depth, dtype=torch.bool)
            interior[
                ty - top : min(ty + patch, ih) - top, tx - left : min(tx + patch, iw) - left
            ] = True
            targets.append(
                dict(
                    view=observation,
                    camera=self.tensor(camera)[None],
                    calibration=self.tensor(calibration)[None],
                    target=target,
                    depth=depth,
                    mask=(depth > 0) & interior,
                )
            )
        self.active = dict(
            ids=ids,
            keys=b.keys[ids].copy(),
            original=original,
            p=p,
            optimizer=optimizer,
            targets=targets,
            steps=0,
            mature=self.tensor(
                (b.support[ids] >= 20) & (b.uncertainty[ids] < 0.03) & ~np.bool_(view.refitting)
            ).bool(),
            # No Gaussian grows past eight of this view's pixels (seeds are about two).
            largest=self.tensor(
                np.log(np.maximum(b.records[ids, 1, :3], 8 * distance[:, None] / fx))
            ),
        )
        return True

    def step(self):
        b, torch = self.backend, self.torch
        if self.active is None and not self.prepare():
            return False
        a = self.active
        ids, p = a["ids"], a["p"]
        if (
            ids.max() >= len(b.keys)
            or not np.array_equal(b.keys[ids], a["keys"])
            or b.retiring[ids].any()
        ):
            self.active = None
            return False
        target_index = a["steps"] % len(a["targets"])
        target_view = a["targets"][target_index]
        h, w = target_view["target"].shape[:2]
        with b.gpu_chunk("fit_forward"):
            pred, alpha, info = self.rasterize(
                means=p["means"],
                quats=p["quats"],
                scales=p["scales"].exp(),
                opacities=p["opacity"].sigmoid(),
                colors=p["color"][:, None],
                sh_degree=0,
                viewmats=target_view["camera"],
                Ks=target_view["calibration"],
                width=w,
                height=h,
                packed=False,
                near_plane=0.01,
                far_plane=1000,
                rasterize_mode="classic",
                render_mode="RGB+ED",
            )
            target = target_view["target"].to(self.dtype)
            rgb = pred[0, ..., :3].to(self.dtype)
            observed_depth = target_view["depth"]
            depth_error = (pred[0, ..., 3] - observed_depth) / observed_depth.clamp_min(0.01)
            mask = target_view["mask"] & (alpha[0, ..., 0].detach() > 0.9)
            mask &= depth_error.detach().abs() < 0.15
            loss = ((rgb - target).abs() * mask[..., None]).sum() / mask.sum().clamp_min(1) / 3
            loss = loss + 0.02 * (depth_error.abs() * mask).sum() / mask.sum().clamp_min(1)
        with b.gpu_chunk("fit_backward"):
            loss.backward()
            a["optimizer"].step()
            a["optimizer"].zero_grad(set_to_none=True)
        with b.gpu_chunk("fit_readback"), torch.no_grad():
            original = a["original"]
            bound = original[:, 1, :3].max(-1).values[:, None] * 0.25
            p["means"].copy_(
                original[:, 0, :3] + (p["means"] - original[:, 0, :3]).clamp(-bound, bound)
            )
            p["means"][a["mature"]] = original[a["mature"], 0, :3]
            p["scales"].copy_(p["scales"].clamp(original[:, 1, :3].log() - 0.2, a["largest"]))
            p["quats"].copy_(torch.nn.functional.normalize(p["quats"], dim=-1))
            p["opacity"].clamp_(-5, 7)
            p["color"].clamp_(-0.5 / C0, 0.5 / C0)
        target_view["view"].steps += 1
        b.fit_steps += 1
        a["steps"] += 1
        if a["steps"] < STEPS:
            return True
        # The unit is published whole, as one revision, or not at all.
        self.active = None
        with b.gpu_chunk("fit_readback"), torch.no_grad():
            records = original.clone()
            records[:, 0, :3] = p["means"]
            records[:, 0, 3] = p["opacity"].sigmoid()
            records[:, 1, :3] = p["scales"].exp()
            records[:, 2] = p["quats"]
            records[:, 3, :3] = p["color"]
            records = records.cpu().numpy()
        if not np.isfinite(records).all():
            return True
        b.records[ids] = records
        b.dirty[ids] = True
        b.pending_fit = ids.copy()
        for target in a["targets"]:
            view = target["view"]
            older = b.captures[ids] > view.captured
            b.captures[ids[older]] = view.captured
            b.sequences[ids[older]] = view.sequence
        # A Gaussian the fit made transparent adds nothing to any view.
        faded = ids[records[:, 0, 3] < 0.02]
        if len(faded):
            b.retire(faded)
        return True
