"""ITO_FIT=stream (research): one optimiser over every live splat, one retained view per step, the
views' poses refined with the splats, as the live-budget mapper of tools/research/stream.py.

ViewFit trains units (a view and two partners, 30 steps) on copies of the splats they see and
writes them back; on recorded sim routes it made maps worse the more it trained (dark pillars and
smears that live, where it gets ~7 steps/s, rarely reaches). Here every live splat sits in GPU
tensors with its Adam state; each step renders one view (newest views half the time, any retained
view otherwise), takes an Adam step on the splats and on that view's pose correction (the oldest
view holds the gauge), and every SYNC steps the map's lock is taken to exchange changes with the
worker: its new seeds and the splats it changed or retired come in, the trained splats that moved
or recoloured go out (refined: they publish like any update).

Per-view pose corrections (fitting_pose) only change where views are trained from, never the
published camera: a 1.5-degree pose error per frame is absorbed this way (stream.py on close)."""

import logging
import math
import time

import numpy as np

from .gaussian import AXES, C0

log = logging.getLogger(__name__)

SYNC = 25  # steps between exchanges with the worker
RATES = dict(means=3.2e-4, scales=5e-3, quats=1e-3, opacity=5e-2, color=2.5e-3)
POSE_RATE = 2e-4  # radians, metres per Adam step
POSE_LIMIT = (math.radians(3), 0.05)  # a view corrected further is left where it was
SSIM_WEIGHT = 0.2
RECENT = 3  # the newest views get half of all steps
PRUNE_OPACITY = 0.02
CHANGED = (1e-4, 1 / 255)  # metres, colour: smaller changes are not worth publishing
DUTY = 0.5  # the trainer rests as long as it worked, as ViewFit, for the display's frames


def ssim(a, b, torch):
    x = torch.arange(11, device=a.device, dtype=a.dtype) - 5
    g = torch.exp(-(x**2) / (2 * 1.5**2))
    g = g / g.sum()
    window = (g[:, None] * g[None, :]).expand(3, 1, 11, 11).contiguous()

    def blur(v):
        return torch.nn.functional.conv2d(v[None], window, padding=5, groups=3)[0]

    mu_a, mu_b = blur(a), blur(b)
    var_a, var_b = blur(a * a) - mu_a**2, blur(b * b) - mu_b**2
    cov = blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    return (num / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))).mean()


class StreamFit:
    def __init__(self, backend):
        import torch
        from gsplat import rasterization

        self.torch, self.rasterize = torch, rasterization
        self.backend = backend
        self.active = None  # ViewFit's handle; recover() clears it (nothing to drop here)
        self.params = None
        self.slots = np.zeros(0, np.int64)  # slot of each parameter row
        self.keys = np.zeros(0, np.int64)  # the slot's key when uploaded (a reused slot differs)
        self.written = np.zeros((0, 4, 4), np.float32)  # the records as this trainer last saw them
        self.deltas = {}  # id(view) -> (rotation, translation) parameters
        self.pose_optimizer = None
        self.units = self.grown = self.pruned = self.aligned = self.filled = 0
        self.steps = 0
        self.serial = 0
        self.seconds = dict(sync=0.0)
        self.rng = np.random.default_rng(0)

    def start(self, available, stopped, failed):
        import threading

        torch = self.torch

        def run():
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
                        self.params = None
                        failed(exc)
                        torch.cuda.empty_cache()
                        time.sleep(1.0)
                        continue
                    except Exception:
                        errors += 1
                        self.params = None
                        log.exception("Stream training step failed")
                        if errors >= 5:
                            log.error("Stream training stopped after repeated failures")
                            return
                        time.sleep(0.5)
                        continue
                    if not progressed:
                        time.sleep(0.05)

        thread = threading.Thread(target=run, name="map-training", daemon=True)
        thread.start()
        return thread

    def tensor(self, value, dtype=None):
        return self.torch.as_tensor(value, device="cuda", dtype=dtype or self.torch.float32)

    def upload(self, records):
        torch = self.torch
        r = self.tensor(records)
        return {
            "means": r[:, 0, :3].clone(),
            "scales": r[:, 1, :3].clamp_min(1e-6).log(),
            "quats": r[:, 2].clone(),
            "opacity": torch.logit(r[:, 0, 3].clamp(0.001, 0.999)),
            "color": r[:, 3, :3].clone(),
        }

    def sync(self):
        """Under the map's lock: write trained splats back, take in the worker's changes."""
        torch, b = self.torch, self.backend
        live = np.flatnonzero((b.keys >= 0) & ~b.retiring)
        if self.params is not None and len(self.slots):
            with torch.no_grad():
                p = self.params
                trained = np.zeros((len(self.slots), 4, 4), np.float32)
                trained[:, 0, :3] = p["means"].detach().cpu().numpy()
                trained[:, 0, 3] = p["opacity"].detach().sigmoid().cpu().numpy()
                trained[:, 1, :3] = p["scales"].detach().exp().cpu().numpy()
                q = torch.nn.functional.normalize(p["quats"].detach(), dim=-1)
                trained[:, 2] = q.cpu().numpy()
                trained[:, 3, :3] = p["color"].detach().cpu().numpy()
            slots = self.slots
            ok = (slots < len(b.keys)) & (b.keys[np.minimum(slots, len(b.keys) - 1)] == self.keys)
            ok &= ~b.retiring[np.minimum(slots, len(b.keys) - 1)]
            # The worker changed a splat since this trainer last wrote it (a colour blend, a seed
            # reusing the slot): its version wins and is taken in afresh below.
            current = b.records[np.minimum(slots, len(b.keys) - 1)]
            untouched = ok & np.all(np.isclose(current[:, :4, :], self.written), axis=(1, 2))
            trained[:, 1, 3] = current[:, 1, 3]
            trained[:, 3, 3] = current[:, 3, 3]
            finite = np.isfinite(trained).all(axis=(1, 2))
            moved = np.linalg.norm(trained[:, 0, :3] - current[:, 0, :3], axis=1) > CHANGED[0]
            recoloured = np.abs(trained[:, 3, :3] - current[:, 3, :3]).max(-1) * C0 > CHANGED[1]
            ratio = trained[:, 1, :3] / np.maximum(current[:, 1, :3], 1e-9)
            reshaped = np.abs(np.log(ratio)).max(-1) > 0.02
            refaded = np.abs(trained[:, 0, 3] - current[:, 0, 3]) > 0.02
            write = untouched & finite & (moved | recoloured | reshaped | refaded)
            b.records[slots[write]] = trained[write]
            b.refined[slots[write]] = True
            faded = untouched & finite & (trained[:, 0, 3] < PRUNE_OPACITY)
            if faded.any():
                b.retire(slots[faded])
                self.pruned += int(faded.sum())
            keep_rows = untouched & finite & ~faded
        else:
            keep_rows = np.zeros(0, bool)
        # Rows kept with their Adam state; every other live slot (new, worker-changed) comes fresh.
        kept = self.slots[keep_rows] if len(keep_rows) else np.zeros(0, np.int64)
        fresh = live[~np.isin(live, kept)]
        if self.params is not None and len(keep_rows):
            index = torch.as_tensor(np.flatnonzero(keep_rows), device="cuda")
            old = {k: v.detach()[index] for k, v in self.params.items()}
            states = {
                k: {
                    "exp_avg": self.optimizers[k].state[self.params[k]]["exp_avg"][index],
                    "exp_avg_sq": self.optimizers[k].state[self.params[k]]["exp_avg_sq"][index],
                    "step": self.optimizers[k].state[self.params[k]]["step"],
                }
                if self.params[k] in self.optimizers[k].state
                else None
                for k in self.params
            }
        else:
            old, states = None, None
        new = self.upload(b.records[fresh]) if len(fresh) else None
        slots = np.concatenate((kept, fresh))
        if not len(slots):
            self.params = None
            self.slots = slots
            return
        params = {}
        for k in RATES:
            parts = [x for x in (old[k] if old else None, new[k] if new else None) if x is not None]
            params[k] = torch.nn.Parameter(torch.cat(parts) if len(parts) > 1 else parts[0].clone())
        optimizers = {k: torch.optim.Adam([params[k]], lr=RATES[k], eps=1e-15) for k in RATES}
        if states is not None:
            n_new = len(fresh)
            for k in RATES:
                if states[k] is None:
                    continue
                pad = torch.zeros((n_new, *params[k].shape[1:]), device="cuda")
                optimizers[k].state[params[k]] = {
                    "exp_avg": torch.cat((states[k]["exp_avg"], pad)),
                    "exp_avg_sq": torch.cat((states[k]["exp_avg_sq"], pad)),
                    "step": states[k]["step"],
                }
        self.params, self.optimizers = params, optimizers
        self.slots = slots
        self.keys = b.keys[slots].copy()
        self.written = b.records[slots].copy()
        self.units += 1

    def choose(self):
        views = self.backend.views
        if not views:
            return None
        if self.rng.random() < 0.5:
            return views[-1 - int(self.rng.integers(0, min(RECENT, len(views))))]
        return views[int(self.rng.integers(0, len(views)))]

    def delta(self, view):
        """The view's pose correction parameters (None for the gauge: the oldest view)."""
        torch = self.torch
        if view is self.backend.views[0]:
            return None
        found = self.deltas.get(id(view))
        if found is None:
            found = (
                torch.nn.Parameter(torch.zeros(3, device="cuda")),
                torch.nn.Parameter(torch.zeros(3, device="cuda")),
                view,
                (view.fitting_pose if view.fitting_pose is not None else view.pose).copy(),
            )
            self.deltas[id(view)] = found
            self.pose_optimizer = None
        return found

    def moved(self, rotation, translation):
        torch = self.torch
        theta = rotation.norm().clamp_min(1e-9)
        k = rotation / theta
        zero = torch.zeros((), device="cuda")
        K = torch.stack(
            (
                torch.stack((zero, -k[2], k[1])),
                torch.stack((k[2], zero, -k[0])),
                torch.stack((-k[1], k[0], zero)),
            )
        )
        R = torch.eye(3, device="cuda") + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
        top = torch.cat((R, translation[:, None]), 1)
        return torch.cat((top, torch.tensor([[0.0, 0.0, 0.0, 1.0]], device="cuda")), 0)

    def step(self):
        torch, b = self.torch, self.backend
        if self.params is None or self.steps % SYNC == 0:
            began = time.perf_counter()
            with b.lock:
                self.sync()
                self.apply_poses()
            self.seconds["sync"] += time.perf_counter() - began
        self.steps += 1
        if self.params is None:
            return False
        with b.lock:
            view = self.choose()
            if view is None:
                return False
            delta = self.delta(view)
            base = (
                delta[3]
                if delta is not None
                else (view.fitting_pose if view.fitting_pose is not None else view.pose)
            )
            fine = view.photo is not None
            image = view.photo if fine else view.image
            fx, fy, cx, cy = view.photo_intrinsics if fine else view.intrinsics
        h, w = image.shape[:2]
        camera = self.tensor(np.linalg.inv(base @ AXES))
        if delta is not None:
            camera = self.moved(delta[0], delta[1]) @ camera
        p = self.params
        rendered, alpha, _ = self.rasterize(
            means=p["means"],
            quats=p["quats"],
            scales=p["scales"].exp(),
            opacities=p["opacity"].sigmoid(),
            colors=p["color"][:, None],
            sh_degree=0,
            viewmats=camera[None],
            Ks=self.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])[None],
            width=w,
            height=h,
            near_plane=0.01,
            far_plane=1000,
            packed=False,
        )
        rgb = rendered[0].permute(2, 0, 1)
        target = self.tensor(image).permute(2, 0, 1) / 255
        loss = (1 - SSIM_WEIGHT) * (rgb - target).abs().mean() + SSIM_WEIGHT * (
            1 - ssim(rgb, target, torch)
        )
        loss.backward()
        for opt in self.optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if delta is not None:
            if self.pose_optimizer is None:
                params = [x for d in self.deltas.values() for x in d[:2]]
                self.pose_optimizer = torch.optim.Adam(params, lr=POSE_RATE)
            self.pose_optimizer.step()
            self.pose_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            p["quats"].copy_(torch.nn.functional.normalize(p["quats"], dim=-1))
            p["opacity"].clamp_(-7, 7)
            p["color"].clamp_(-0.5 / C0, 0.5 / C0)
        view.steps += 1
        b.fit_steps += 1
        return True

    def apply_poses(self):
        """Under the map's lock: views' corrected training poses; forget views no longer kept."""
        retained = {id(v) for v in self.backend.views}
        for key in list(self.deltas):
            if key not in retained:
                del self.deltas[key]
                self.pose_optimizer = None
        for rotation, translation, view, base in self.deltas.values():
            with self.torch.no_grad():
                angle, shift = float(rotation.norm()), float(translation.norm())
                if angle > POSE_LIMIT[0] or shift > POSE_LIMIT[1]:
                    continue
                camera = self.moved(rotation, translation) @ self.tensor(np.linalg.inv(base @ AXES))
                view.fitting_pose = np.linalg.inv(camera.double().cpu().numpy()) @ AXES
        self.aligned = len(self.deltas)
