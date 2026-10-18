"""Photometric refinement over whole retained views, the way MonoGS and MoonSplat map.

LocalFit polishes 32-pixel patches and freezes settled surfaces. A ViewFit unit takes one
retained view and the views overlapping it, every Gaussian any of them sees, and trains them
together against those photos for STEPS Adam steps: colour, opacity, shape and position, with
the views' agreed depth as a prior so geometry moves only where the photos say so. Where the
photos keep pulling at a Gaussian it splits or clones (more detail), where it fades it goes.

The CPU map stays the source of truth: a unit uploads its Gaussians, trains them on CUDA and
writes back only those still live and visibly changed, which then publish like any update.
"""

import logging
import math
import os
import time

import numpy as np

from .gaussian import AXES, C0, EXPOSURE, exposed

log = logging.getLogger(__name__)

STEPS = int(os.environ.get("ITO_FIT_STEPS", "30"))  # per unit
PARTNERS = 2  # overlapping views trained beside the chosen one
MAX_UNIT = 400_000
SSIM_WEIGHT = 0.2
DEPTH_WEIGHT = 0.1
SCALE_WEIGHT = 0.01  # flatness: a surface's Gaussians stay thin across it, discs not balls
NEEDLE_WEIGHT = 0.1  # and no axis grows past NEEDLE times the next: no needles
NEEDLE = 3.0
# A Gaussian grows to at most GROWTH times its size when the unit began, and to at most
# WIDEST pixels in the unit's own view: seen from one side, photos happily stretch splats into
# needles and sheets that streak across every other view.
GROWTH = 3.0
WIDEST = 12.0
RATES = dict(means=2e-4, scales=5e-3, quats=1e-3, opacity=5e-2, color=2.5e-3)
GROW_GRADIENT = 8e-4  # mean absolute screen-space gradient (normalised) asking for detail
GROW_SHARE = 0.05  # at most this share of a unit's Gaussians is cloned or split per unit
SPLIT_PIXELS = 6.0  # a Gaussian larger than this in its view splits; smaller ones clone
PRUNE_OPACITY = 0.01
CHANGED = (1e-4, 1 / 255)  # metres, colour: smaller changes are not worth publishing
# ITO_FIT_POSES=1: before training, align the unit's chosen view to the map photometrically
# (the map held still) and train from that pose: the view stops pulling surfaces off-place.
POSES = os.environ.get("ITO_FIT_POSES", "0") != "0"
ALIGN_STEPS = 25
ALIGN_RATES = (1.5e-3, 1.5e-3)  # radians, metres per Adam step at first, decaying tenfold
ALIGN_COVERED = 0.3  # of the view the map must cover to align it
ALIGN_LIMIT = (math.radians(3), 0.05)  # a larger correction is a mismatch, not an alignment
# Holes: where the map shows nothing in a view, its own depth from before the views agreed is
# checked again against every retained view that now faces it (later ones too), and what
# they agree on is seeded. A first glimpse that the next frame missed is not lost for good.
FILL_REFERENCES = 6
FILL_ALPHA = 0.5
# The trainer thread's share of time on the GPU: after a step it rests as long again. A step
# (a 1024 x 576 view, a few hundred thousand Gaussians) takes tens of milliseconds the pilot
# view's frames would otherwise queue behind; at full duty they missed about 5% of 60 Hz
# deadlines, against 2% without training.
DUTY = float(os.environ.get("ITO_FIT_DUTY", "0.5"))
# Every GLOBAL-th unit starts from the least trained of all retained views, the others from
# the least trained of the last three (ITO_FIT_GLOBAL, an experiment; 1: always all views).
GLOBAL = max(1, int(os.environ.get("ITO_FIT_GLOBAL", "4")))
# ITO_AA=1 (an experiment), Mip-Splatting's two filters: training renders "antialiased" (the
# 0.3 px screen dilation keeps a splat's energy, as the pilot's renderer then draws it), and
# no axis of a Gaussian shrinks below MIP_3D of a pixel of the finest view that sees it, so
# nothing is finer than any photo could show (sparkle and gaps up close in the headset).
AA = os.environ.get("ITO_AA", "0") != "0"
MIP_3D = math.sqrt(0.2)
MODE = "antialiased" if AA else "classic"
# ITO_FIT_SPARSE=<weight> (an experiment): 3DGS-MCMC's L1 on opacity. A Gaussian the photos
# do not need fades below PRUNE_OPACITY and retires: floaters and doubled layers thin out.
SPARSE = float(os.environ.get("ITO_FIT_SPARSE", "0") or 0)
# ITO_EXPOSURE=1: each view's colours are the map's times its gain plus its offset; both
# train beside the map and stay near what the frame's own estimate said (gaussian.py).
EXPOSURE_RATES = (2e-3, 1e-3)  # log gain, offset per Adam step
EXPOSURE_WEIGHT = 0.05


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
        self.aligned = 0
        self.filled = 0
        self.seconds = dict(prepare=0.0, finish=0.0)  # CPU time outside the Adam steps
        self.threaded = False

    def start(self, available, stopped, failed):
        """Train on a thread: GPU steps on a stream of their own while tracking runs, the map's
        lock only to start and finish a unit. available(): the renderer has not asked the
        GPU to back off; stopped(): end; failed(exc): an out-of-memory to report."""
        import threading

        self.threaded = True

        def run():
            torch = self.torch
            stream = torch.cuda.Stream(priority=0)
            errors = 0
            with torch.cuda.stream(stream):
                while not stopped():
                    if not available():
                        time.sleep(0.002)
                        continue
                    try:
                        began = time.perf_counter()
                        progressed = self.step()
                        torch.cuda.current_stream().synchronize()
                        if progressed and 0 < DUTY < 1:
                            time.sleep((time.perf_counter() - began) * (1 / DUTY - 1))
                        errors = 0
                    except torch.cuda.OutOfMemoryError as exc:
                        self.active = None
                        failed(exc)
                        torch.cuda.empty_cache()
                        time.sleep(1.0)
                        continue
                    except Exception:
                        self.active = None
                        errors += 1
                        log.exception("Training unit failed")
                        if errors >= 5:
                            log.error("Training stopped after repeated failures")
                            return
                        time.sleep(0.5)
                        continue
                    if not progressed:
                        time.sleep(0.05)

        thread = threading.Thread(target=run, name="map-training", daemon=True)
        thread.start()
        return thread

    def chunk(self, kind):
        """GPU accounting for the worker's admission; a trainer thread keeps out of it."""
        from contextlib import nullcontext

        return nullcontext() if self.threaded else self.backend.gpu_chunk(kind)

    def tensor(self, value, dtype=None):
        torch = self.torch
        return torch.as_tensor(value, device="cuda", dtype=dtype or torch.float32)

    def choose(self):
        b = self.backend
        views = list(b.views)
        recent = views[-3:]
        pool = recent if self.serial % GLOBAL else views
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

    def visible(self, centers, reach, pose, intrinsics, shape):
        """Mask of centers (N x 3, map frame, on CUDA) whose footprint (reach, metres) falls in
        a camera's frame."""
        fx, fy, cx, cy = intrinsics
        height, width = shape
        local = (centers - self.tensor(pose[:3, 3])) @ self.tensor(pose[:3, :3])
        z = -local[:, 2]
        safe = z.clamp_min(0.01)
        u = cx + fx * local[:, 0] / safe
        v = cy - fy * local[:, 1] / safe
        r = 3 * fx * reach / safe
        inside = (z > 0.01) & (u > -r) & (u < width + r) & (v > -r) & (v < height + r)
        return inside

    def prepare(self):
        """Pick a view and its partners and copy the Gaussians they see. The map's lock is held
        only to copy; projecting, uploading and filling holes run beside tracking."""
        b, torch = self.backend, self.torch
        with b.lock:
            if not b.views or not b.count:
                return False
            view = self.choose()
            targets = [view, *self.partners(view)]
            live = np.flatnonzero((b.keys >= 0) & ~b.retiring)
            snapshot = b.records[live].copy()
            keys = b.keys[live].copy()
        if not len(live):
            return False
        centers = self.tensor(snapshot[:, 0, :3])
        reach = self.tensor(snapshot[:, 1, :3].max(-1))
        if view.raw_depth is not None and len(b.views) > view.filled:
            self.fill(view, snapshot, centers, reach)
        seen = torch.zeros(len(live), dtype=torch.bool, device="cuda")
        for observation in targets:
            pose = (
                observation.fitting_pose
                if observation.fitting_pose is not None
                else observation.pose
            )
            seen |= self.visible(
                centers, reach, pose, observation.intrinsics, observation.depth.shape
            )
        chosen = torch.nonzero(seen)[:, 0].cpu().numpy()
        if not len(chosen):
            view.steps += STEPS
            return False
        if len(chosen) > MAX_UNIT:
            rng = np.random.default_rng(self.serial)
            chosen = np.sort(rng.choice(chosen, MAX_UNIT, replace=False))
        ids = live[chosen]
        original = self.tensor(snapshot[chosen])
        first = view.fitting_pose if view.fitting_pose is not None else view.pose
        distance = (-(original[:, 0, :3] - self.tensor(first[:3, 3])) @ self.tensor(first[:3, :3]))[
            :, 2
        ].clamp_min(0.05)
        widest = WIDEST * distance / view.intrinsics[0]
        largest = torch.minimum(
            original[:, 1, :3].clamp_min(1e-6) * GROWTH,
            torch.maximum(widest[:, None], original[:, 1, :3]),
        ).log()
        smallest = None
        if AA:
            finest = torch.full((len(chosen),), float("inf"), device="cuda")
            for observation in targets:
                pose = (
                    observation.fitting_pose
                    if observation.fitting_pose is not None
                    else observation.pose
                )
                fine = observation.photo_intrinsics or observation.intrinsics
                z = (-(original[:, 0, :3] - self.tensor(pose[:3, 3])) @ self.tensor(pose[:3, :3]))[
                    :, 2
                ]
                inside = self.visible(
                    original[:, 0, :3],
                    original[:, 1, :3].max(-1).values,
                    pose,
                    observation.intrinsics,
                    observation.depth.shape,
                )
                finest = torch.where(
                    inside, torch.minimum(finest, z.clamp_min(0.01) / fine[0]), finest
                )
            smallest = torch.where(torch.isfinite(finest), MIP_3D * finest, 0.0).clamp_min(1e-7)
            smallest = torch.minimum(smallest[:, None].log(), largest)
        p = {
            "means": torch.nn.Parameter(original[:, 0, :3].clone()),
            "scales": torch.nn.Parameter(original[:, 1, :3].clamp_min(1e-6).log()),
            "quats": torch.nn.Parameter(original[:, 2].clone()),
            "opacity": torch.nn.Parameter(torch.logit(original[:, 0, 3].clamp(0.001, 0.999))),
            "color": torch.nn.Parameter(original[:, 3, :3].clone()),
        }
        prepared = []
        exposures = []
        for observation in targets:
            pose = (
                observation.fitting_pose
                if observation.fitting_pose is not None
                else observation.pose
            )
            fine = observation.photo is not None
            image = observation.photo if fine else observation.image
            fx, fy, cx, cy = observation.photo_intrinsics if fine else observation.intrinsics
            h, w = image.shape[:2]
            depth = self.tensor(observation.depth)
            if depth.shape != (h, w):
                depth = torch.nn.functional.interpolate(depth[None, None], (h, w))[0, 0]
            exposure = None
            if EXPOSURE and observation.exposure is not None:
                if observation.exposure_prior is None:
                    observation.exposure_prior = observation.exposure.copy()
                prior = self.tensor(observation.exposure_prior)
                exposure = (torch.nn.Parameter(self.tensor(observation.exposure)), prior)
                exposures.append(exposure[0])
            prepared.append(
                dict(
                    view=observation,
                    exposure=exposure,
                    camera=self.tensor(np.linalg.inv(pose @ AXES))[None],
                    calibration=self.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])[None],
                    target=self.tensor(image).permute(2, 0, 1) / 255,
                    depth=depth,
                    size=(w, h),
                    focal=fx,
                )
            )
        groups = [{"params": [value], "lr": RATES[key]} for key, value in p.items()]
        if exposures:
            groups.append({"params": exposures, "lr": EXPOSURE_RATES[0]})
        optimizer = torch.optim.Adam(groups, eps=1e-15)
        self.active = dict(
            ids=ids,
            keys=keys[chosen],
            original=original,
            snapshot=snapshot[chosen],
            p=p,
            optimizer=optimizer,
            targets=prepared,
            steps=0,
            gradient=torch.zeros(len(ids), device="cuda"),
            seen=torch.zeros(len(ids), device="cuda"),
            pixels=torch.zeros(len(ids), device="cuda"),
            largest=largest,
            smallest=smallest,
        )
        if POSES:
            rotation = torch.nn.Parameter(torch.zeros(3, device="cuda"))
            translation = torch.nn.Parameter(torch.zeros(3, device="cuda"))
            self.active |= dict(
                align=ALIGN_STEPS,
                delta=(rotation, translation),
                pose_optimizer=torch.optim.Adam(
                    [
                        {"params": [rotation], "lr": ALIGN_RATES[0]},
                        {"params": [translation], "lr": ALIGN_RATES[1]},
                    ]
                ),
            )
        return True

    def fill(self, view, snapshot, centers, reach):
        """Seed what the retained views now agree on where the map shows nothing."""
        from .consensus import agreed

        b, torch = self.backend, self.torch
        with b.lock:
            view.filled = len(b.views)
            pose = view.fitting_pose if view.fitting_pose is not None else view.pose
            scored = []
            for other in b.views:
                if other is view:
                    continue
                o = other.fitting_pose if other.fitting_pose is not None else other.pose
                cosine = float(o[:3, 2] @ pose[:3, 2])
                distance = float(np.linalg.norm(o[:3, 3] - pose[:3, 3]))
                if cosine > 0.5 and distance < 1.5:
                    scored.append((1 - cosine + 0.2 * distance, other))
            scored.sort(key=lambda item: item[0])
            references = [
                (
                    o.raw_depth if o.raw_depth is not None else o.depth,
                    o.fitting_pose if o.fitting_pose is not None else o.pose,
                    o.intrinsics,
                )
                for _, o in scored[:FILL_REFERENCES]
            ]
        if not references:
            return
        with self.chunk("consensus"):
            depth, _ = agreed(view.raw_depth, pose, view.intrinsics, references, torch)
        shape = depth.shape
        inside = self.visible(centers, reach, pose, view.intrinsics, shape)
        drawn = torch.as_tensor(snapshot, device="cuda")[inside]
        fx, fy, cx, cy = view.intrinsics
        with torch.no_grad():
            _, alpha, _ = self.rasterize(
                means=drawn[:, 0, :3],
                quats=drawn[:, 2],
                scales=drawn[:, 1, :3],
                opacities=drawn[:, 0, 3],
                colors=drawn[:, 3, :3][:, None],
                sh_degree=0,
                viewmats=self.tensor(np.linalg.inv(pose @ AXES))[None],
                Ks=self.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])[None],
                width=shape[1],
                height=shape[0],
                near_plane=0.01,
                far_plane=1000,
            )
        holes = (alpha[0, ..., 0].cpu().numpy() < FILL_ALPHA) & (depth > 0)
        if not holes.any():
            return
        rows, columns = np.mgrid[: shape[0], : shape[1]]
        points = np.stack(
            ((columns - cx) / fx * depth, -(rows - cy) / fy * depth, -depth), -1
        ).astype(np.float32)
        with b.lock:
            before = b.count
            image = view.image if view.exposure is None else exposed(view.image, view.exposure[:3])
            b.seed(points, image, depth, holes, pose, view.intrinsics, view.captured)
            self.filled += b.count - before

    def moved(self, delta):
        """4x4 camera-frame motion of a (rotation vector, translation) delta."""
        torch = self.torch
        w, t = delta
        zero = torch.zeros((), device="cuda")
        skew = torch.stack(
            (
                torch.stack((zero, -w[2], w[1])),
                torch.stack((w[2], zero, -w[0])),
                torch.stack((-w[1], w[0], zero)),
            )
        )
        top = torch.cat((torch.linalg.matrix_exp(skew), t[:, None]), 1)
        return torch.cat((top, torch.tensor([[0.0, 0.0, 0.0, 1.0]], device="cuda")), 0)

    def align(self, a):
        """One step of the chosen view's pose against the map as it stands."""
        torch = self.torch
        p, target = a["p"], a["targets"][0]
        w, h = target["size"]
        with self.chunk("fit_forward"):
            camera = self.moved(a["delta"]) @ target["camera"][0]
            rendered, alpha, _ = self.rasterize(
                means=p["means"].detach(),
                quats=p["quats"].detach(),
                scales=p["scales"].detach().exp(),
                opacities=p["opacity"].detach().sigmoid(),
                colors=p["color"].detach()[:, None],
                sh_degree=0,
                viewmats=camera[None],
                Ks=target["calibration"],
                width=w,
                height=h,
                packed=False,
                near_plane=0.01,
                far_plane=1000,
                render_mode="RGB+ED",
            )
            rgb = rendered[0, ..., :3].permute(2, 0, 1)
            observed = target["target"]
            covered = alpha[0, ..., 0].detach() > 0.9
            if a["align"] == ALIGN_STEPS and float(covered.float().mean()) < ALIGN_COVERED:
                a["align"] = 0
                return
            loss = ((rgb - observed).abs().mean(0) * covered).sum() / covered.sum().clamp_min(1)
            known = target["depth"]
            mask = covered & (known > 0)
            if mask.any():
                drawn = rendered[0, ..., 3]
                loss = (
                    loss + DEPTH_WEIGHT * ((drawn[mask] - known[mask]).abs() / known[mask]).mean()
                )
        with self.chunk("fit_backward"):
            loss.backward()
            done = 1 - a["align"] / ALIGN_STEPS
            for group, rate in zip(a["pose_optimizer"].param_groups, ALIGN_RATES, strict=True):
                group["lr"] = rate * 0.1**done
            a["pose_optimizer"].step()
            a["pose_optimizer"].zero_grad(set_to_none=True)
        a["align"] -= 1
        if a["align"]:
            return
        with torch.no_grad():
            rotation, translation = a["delta"]
            angle, shift = float(rotation.norm()), float(translation.norm())
            if angle > ALIGN_LIMIT[0] or shift > ALIGN_LIMIT[1]:
                return
            camera = self.moved(a["delta"]) @ target["camera"][0]
            target["camera"] = camera[None]
            view = target["view"]
            view.fitting_pose = np.linalg.inv(camera.double().cpu().numpy()) @ AXES
            self.aligned += 1

    def step(self):
        b, torch = self.backend, self.torch
        if self.active is None:
            began = time.perf_counter()
            prepared = self.prepare()
            self.seconds["prepare"] += time.perf_counter() - began
            if not prepared:
                return False
        a = self.active
        ids, p = a["ids"], a["p"]
        if ids.max() >= len(b.keys):
            self.active = None
            return False
        if a.get("align"):
            self.align(a)
            return True
        target = a["targets"][a["steps"] % len(a["targets"])]
        w, h = target["size"]
        with self.chunk("fit_forward"):
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
                rasterize_mode=MODE,
            )
            info["means2d"].retain_grad()
            rgb = rendered[0, ..., :3].permute(2, 0, 1)
            observed = target["target"]
            regular = 0
            if target["exposure"] is not None:
                exposure, start = target["exposure"]
                rgb = rgb * exposure[:3].exp()[:, None, None] + exposure[3:, None, None]
                regular = EXPOSURE_WEIGHT * ((exposure - start) ** 2).sum()
            l1 = (rgb - observed).abs().mean()
            loss = (1 - SSIM_WEIGHT) * l1 + SSIM_WEIGHT * (1 - ssim(rgb, observed, torch))
            loss = loss + regular
            known = target["depth"]
            drawn = rendered[0, ..., 3]
            mask = (known > 0) & (alpha[0, ..., 0].detach() > 0.5)
            if mask.any():
                error = (drawn[mask] - known[mask]).abs() / known[mask]
                loss = loss + DEPTH_WEIGHT * error.mean()
            axes = p["scales"].exp().sort(-1).values
            loss = loss + SCALE_WEIGHT * (axes[:, 0] / axes[:, 1].clamp_min(1e-6)).mean()
            needle = (axes[:, 2] / axes[:, 1].clamp_min(1e-6) - NEEDLE).clamp_min(0)
            loss = loss + NEEDLE_WEIGHT * needle.mean()
            if SPARSE:
                loss = loss + SPARSE * p["opacity"].sigmoid().mean()
        with self.chunk("fit_backward"):
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
                p["scales"].copy_(torch.minimum(p["scales"], a["largest"]))
                if a["smallest"] is not None:
                    p["scales"].copy_(torch.maximum(p["scales"], a["smallest"]))
        target["view"].steps += 1
        b.fit_steps += 1
        a["steps"] += 1
        if a["steps"] < STEPS:
            return True
        began = time.perf_counter()
        # A pose correction (recover()) dropped this unit while it trained: its views moved,
        # so what it learned from them is not written back.
        if self.active is a:
            self.active = None
            self.finish(a)
        self.seconds["finish"] += time.perf_counter() - began
        return True

    def finish(self, a):
        """Write back what training changed, retire what faded, grow where the photos asked;
        the map's lock only for the writing."""
        b, torch = self.backend, self.torch
        ids, p = a["ids"], a["p"]
        with self.chunk("fit_readback"), torch.no_grad():
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
        for target in a["targets"]:
            if target["exposure"] is not None:
                target["view"].exposure = target["exposure"][0].detach().cpu().numpy()
        old = a["snapshot"]
        records[:, 1, 3] = old[:, 1, 3]
        records[:, 3, 3] = old[:, 3, 3]
        finite = np.isfinite(records).all(axis=(1, 2))
        moved = np.linalg.norm(records[:, 0, :3] - old[:, 0, :3], axis=1) > CHANGED[0]
        recoloured = np.abs(records[:, 3, :3] - old[:, 3, :3]).max(-1) * C0 > CHANGED[1]
        reshaped = (
            np.abs(np.log(records[:, 1, :3] / np.maximum(old[:, 1, :3], 1e-9))).max(-1) > 0.02
        )
        refaded = np.abs(records[:, 0, 3] - old[:, 0, 3]) > 0.02
        changed = finite & (moved | recoloured | reshaped | refaded)
        faded = finite & (records[:, 0, 3] < PRUNE_OPACITY)
        children, parents, split_parents = self.children(records, gradient, pixels, finite)
        with b.lock:
            safe = np.minimum(ids, len(b.keys) - 1)
            live = (ids < len(b.keys)) & ~b.retiring[safe] & (b.keys[safe] == a["keys"])
            written = ids[changed & live]
            b.records[written] = records[changed & live]
            b.refined[written] = True
            for target in a["targets"]:
                view = target["view"]
                older = b.captures[written] > view.captured
                b.captures[written[older]] = view.captured
                b.sequences[written[older]] = view.sequence
            gone = ids[faded & live]
            if len(gone):
                b.retire(gone)
                self.pruned += len(gone)
            if children is not None:
                self.grow(ids, live, records, children, parents, split_parents)

    def children(self, records, gradient, pixels, finite):
        """New Gaussians where the photos kept pulling: a clone of a small one, half of a large
        one (split along its longest axis, both halves at 1/1.6 the size). (children, parent
        indices into the unit, which parents shrink to the other half) or Nones."""
        wanted = np.flatnonzero(finite & (gradient > GROW_GRADIENT) & (records[:, 0, 3] > 0.05))
        if not len(wanted):
            return None, None, None
        limit = max(1, int(GROW_SHARE * len(records)))
        wanted = wanted[np.argsort(-gradient[wanted])[:limit]]
        children = records[wanted].copy()
        split = pixels[wanted] > SPLIT_PIXELS
        if split.any():
            from scipy.spatial.transform import Rotation

            quat = children[split, 2]
            rotation = Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_matrix()
            scales = children[split, 1, :3]
            axis = np.argmax(scales, axis=1)
            rows = np.arange(len(axis))
            offset = rotation[rows, :, axis] * scales[rows, axis][:, None]
            records[wanted[split], 0, :3] -= 0.5 * offset
            records[wanted[split], 1, :3] /= 1.6
            children[split, 0, :3] += 0.5 * offset
            children[split, 1, :3] /= 1.6
        return children, wanted, split

    def grow(self, ids, live, records, children, wanted, split):
        """Under the map's lock: add children of parents still live, and write split parents
        as the other half (records: the unit's, where children() shrank them)."""
        b = self.backend
        keep = live[wanted]
        children, wanted, split = children[keep], wanted[keep], split[keep]
        free = np.flatnonzero(b.keys[: b.budget] < 0)
        room = min(len(wanted), len(free), max(0, b.budget - b.count))
        if room <= 0:
            return
        children, wanted, split = children[:room], wanted[:room], split[:room]
        parents = ids[wanted]
        if split.any():
            b.records[parents[split]] = records[wanted[split]]
            b.refined[parents[split]] = True
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
        b.refined[slots] = True
        self.grown += room


class FrameAlign:
    """A new frame laid onto the map photometrically before it is mapped (ITO_REFINE=1): the
    map held still, the frame's pose moved to where its photo and depth agree with what the
    map draws there. The robot's kinematic camera pose is off by its gear play (about two
    degrees in the sim); a frame seeded where it was not doubles every edge it sees."""

    LEVELS = ((0.5, 24), (1.0, 8))  # (resolution, Adam steps): coarse first, wider reach
    RATES = (4e-3, 4e-3)  # radians, metres per Adam step at first, decaying tenfold
    LIMIT = (math.radians(3), 0.05)
    COVERED = 0.5  # of the frame the map must cover

    def __init__(self, backend):
        import torch
        from gsplat import rasterization

        self.torch = torch
        self.rasterize = rasterization
        self.backend = backend
        self.aligned = 0
        self.refused = 0
        self.fit = ViewFit(backend)

    def __call__(self, image, depth, pose, intrinsics):
        """The refined map-from-camera pose of a frame (image H x W x 3 uint8, depth metres
        at intrinsics), or None when the map does not cover it or the move is implausible."""
        b, torch = self.backend, self.torch
        live = np.flatnonzero((b.keys >= 0) & ~b.retiring)
        if len(live) < 1000:
            return None
        snapshot = torch.as_tensor(b.records[live], device="cuda")
        reach = snapshot[:, 1, :3].max(-1).values
        inside = self.fit.visible(snapshot[:, 0, :3], reach, pose, intrinsics, depth.shape)
        drawn = snapshot[inside]
        if len(drawn) < 1000:
            return None
        height, width = depth.shape
        fx, fy, cx, cy = intrinsics
        camera = torch.as_tensor(np.linalg.inv(pose @ AXES), device="cuda", dtype=torch.float32)
        full = torch.as_tensor(image, device="cuda").permute(2, 0, 1).float() / 255
        known_full = torch.as_tensor(depth, device="cuda", dtype=torch.float32)
        rotation = torch.nn.Parameter(torch.zeros(3, device="cuda"))
        translation = torch.nn.Parameter(torch.zeros(3, device="cuda"))
        optimizer = torch.optim.Adam(
            [
                {"params": [rotation], "lr": self.RATES[0]},
                {"params": [translation], "lr": self.RATES[1]},
            ]
        )
        total = sum(steps for _, steps in self.LEVELS)
        done = 0
        for share, steps in self.LEVELS:
            w, h = round(width * share), round(height * share)
            K = torch.tensor(
                [
                    [fx * share, 0, (cx + 0.5) * share - 0.5],
                    [0, fy * share, (cy + 0.5) * share - 0.5],
                    [0, 0, 1],
                ],
                device="cuda",
            )
            if share == 1.0:
                target, known = full, known_full
            else:
                target = torch.nn.functional.interpolate(full[None], (h, w), mode="area")[0]
                known = torch.nn.functional.interpolate(known_full[None, None], (h, w))[0, 0]
            for _ in range(steps):
                moved = self.fit.moved((rotation, translation)) @ camera
                rendered, alpha, _ = self.rasterize(
                    means=drawn[:, 0, :3],
                    quats=drawn[:, 2],
                    scales=drawn[:, 1, :3],
                    opacities=drawn[:, 0, 3],
                    colors=drawn[:, 3, :3][:, None],
                    sh_degree=0,
                    viewmats=moved[None],
                    Ks=K[None],
                    width=w,
                    height=h,
                    packed=False,
                    near_plane=0.01,
                    far_plane=1000,
                    render_mode="RGB+ED",
                )
                covered = alpha[0, ..., 0].detach() > 0.9
                if done == 0 and float(covered.float().mean()) < self.COVERED:
                    self.refused += 1
                    return None
                rgb = rendered[0, ..., :3].permute(2, 0, 1)
                loss = ((rgb - target).abs().mean(0) * covered).sum() / covered.sum().clamp_min(1)
                mask = covered & (known > 0)
                if mask.any():
                    drawn_depth = rendered[0, ..., 3]
                    loss = (
                        loss
                        + DEPTH_WEIGHT
                        * ((drawn_depth[mask] - known[mask]).abs() / known[mask]).mean()
                    )
                loss.backward()
                for group, rate in zip(optimizer.param_groups, self.RATES, strict=True):
                    group["lr"] = rate * 0.1 ** (done / total)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                done += 1
        with torch.no_grad():
            if float(rotation.norm()) > self.LIMIT[0] or float(translation.norm()) > self.LIMIT[1]:
                self.refused += 1
                return None
            moved = self.fit.moved((rotation, translation)) @ camera
        self.aligned += 1
        return np.linalg.inv(moved.double().cpu().numpy()) @ AXES
