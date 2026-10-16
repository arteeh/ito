"""Small photometric fits, using the same anisotropic Gaussian records as the display.

Only a cropped view and at most one publication packet of parameters live on CUDA.
The CPU map is committed after the complete fit and its readback succeed.
"""

import numpy as np

from .gaussian import C0

PATCH = 64
HALO = 12
MAX_LOCAL = 4096
STEPS = 4


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
        h, w = view.image.shape[:2]
        cols = (w + PATCH - 1) // PATCH
        rows = (h + PATCH - 1) // PATCH
        tile = (view.steps // STEPS) % (cols * rows)
        x, y = (tile % cols) * PATCH, (tile // cols) * PATCH
        left, top = max(0, x - HALO), max(0, y - HALO)
        right, bottom = min(w, x + PATCH + HALO), min(h, y + PATCH + HALO)
        ids = np.flatnonzero((b.keys >= 0) & ~b.retiring)
        local = (b.records[ids, 0, :3] - view.pose[:3, 3]) @ view.pose[:3, :3]
        z = -local[:, 2]
        fx, fy, cx, cy = view.intrinsics
        u = cx + fx * local[:, 0] / np.maximum(z, 0.01)
        v = cy - fy * local[:, 1] / np.maximum(z, 0.01)
        radius = 3 * fx * b.records[ids, 1, :3].max(-1) / np.maximum(z, 0.01) + 2
        selected = (z > 0.01) & (u + radius >= left) & (u - radius < right)
        selected &= (v + radius >= top) & (v - radius < bottom)
        ids = ids[selected]
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
            "exposure": torch.nn.Parameter(
                self.tensor(np.zeros((2, 3)) if view.exposure is None else view.exposure)
            ),
        }
        rates = dict(
            means=0.0001, scales=0.002, quats=0.0005, opacity=0.01, color=0.01, exposure=0.001
        )
        optimizer = torch.optim.Adam(
            [{"params": [value], "lr": rates[key]} for key, value in p.items()], fused=True
        )
        camera = np.linalg.inv(view.pose @ np.diag([1.0, -1.0, -1.0, 1.0]))
        calibration = [[fx, 0, cx - left], [0, fy, cy - top], [0, 0, 1]]
        target = self.tensor(view.image[top:bottom, left:right]) / 255
        mask = self.tensor(view.depth[top:bottom, left:right] > 0).bool()
        interior = torch.zeros_like(mask)
        interior[y - top : min(y + PATCH, h) - top, x - left : min(x + PATCH, w) - left] = True
        self.active = dict(
            ids=ids,
            keys=b.keys[ids].copy(),
            original=original,
            p=p,
            optimizer=optimizer,
            view=view,
            camera=self.tensor(camera)[None],
            calibration=self.tensor(calibration)[None],
            target=target,
            mask=mask & interior,
            steps=0,
            mature=self.tensor(b.support[ids] >= 3).bool(),
        )
        return True

    def step(self):
        b, torch = self.backend, self.torch
        if self.active is None and not self.prepare():
            return
        a = self.active
        ids, p = a["ids"], a["p"]
        if (
            ids.max() >= len(b.keys)
            or not np.array_equal(b.keys[ids], a["keys"])
            or b.retiring[ids].any()
        ):
            self.active = None
            return
        h, w = a["target"].shape[:2]
        with b.gpu_chunk("fit_forward"):
            pred, alpha, info = self.rasterize(
                means=p["means"],
                quats=p["quats"],
                scales=p["scales"].exp(),
                opacities=p["opacity"].sigmoid(),
                colors=p["color"][:, None],
                sh_degree=0,
                viewmats=a["camera"],
                Ks=a["calibration"],
                width=w,
                height=h,
                packed=False,
                near_plane=0.01,
                far_plane=1000,
                rasterize_mode="classic",
            )
            target = a["target"].to(self.dtype)
            rgb = (pred[0] * p["exposure"][0].exp() + p["exposure"][1]).to(self.dtype)
            mask = a["mask"] & (alpha[0, ..., 0].detach() > 0.5)
            loss = ((rgb - target).abs() * mask[..., None]).sum() / mask.sum().clamp_min(1) / 3
            loss = loss + 0.01 * p["exposure"].square().mean()
        with b.gpu_chunk("fit_backward"):
            loss.backward()
            a["optimizer"].step()
            a["optimizer"].zero_grad(set_to_none=True)
        with b.gpu_chunk("fit_readback"), torch.no_grad():
            original = a["original"]
            bound = original[:, 1, :3].min(-1).values[:, None] * 0.5
            p["means"].copy_(
                original[:, 0, :3] + (p["means"] - original[:, 0, :3]).clamp(-bound, bound)
            )
            p["means"][a["mature"]] = original[a["mature"], 0, :3]
            p["scales"].copy_(
                p["scales"].clamp(original[:, 1, :3].log() - 0.1, original[:, 1, :3].log() + 0.1)
            )
            p["quats"].copy_(torch.nn.functional.normalize(p["quats"], dim=-1))
            p["opacity"].clamp_(-5, 7)
            p["color"].clamp_(-0.5 / C0, 0.5 / C0)
            p["exposure"].clamp_(-0.2, 0.2)
            records = original.clone()
            records[:, 0, :3] = p["means"]
            records[:, 0, 3] = p["opacity"].sigmoid()
            records[:, 1, :3] = p["scales"].exp()
            records[:, 2] = p["quats"]
            records[:, 3, :3] = p["color"]
            records = records.cpu().numpy()
            exposure = p["exposure"].cpu().numpy().copy()
        if not np.isfinite(records).all():
            self.active = None
            return
        b.records[ids] = records
        b.dirty[ids] = True
        older = b.captures[ids] > a["view"].captured
        b.captures[ids[older]] = a["view"].captured
        b.sequences[ids[older]] = a["view"].sequence
        a["view"].exposure = exposure
        a["view"].steps += 1
        a["steps"] += 1
        if a["steps"] >= STEPS:
            self.active = None
