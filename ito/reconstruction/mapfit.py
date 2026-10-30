"""ITO_FIT=map (research): the whole map trained in place on the GPU, one retained view per
step, the views' poses refined alongside, at a cost that does not grow with the map's size.

ITO_FIT=stream (streamfit.py) showed the design is clean (no dark pillar, unlike ViewFit) but
reached only 5-8 steps/s inside Ito's replays: every 25 steps it copied the whole map to the
CPU, compared every record and rebuilt its tensors and optimisers. Here every slot has a fixed
row in GPU tensors (parameters, Adam moments, a step count per row), so nothing is rebuilt:
- the worker flags each slot it writes (fresh: a seed, a colour blend, a retirement, a move); at a
  sync only those rows are uploaded, their Adam state reset, and a changed slot always wins over
  training (the worker saw the newer frame);
- a step renders the live rows (gathered), and Adam updates only the rows that view saw (radii > 0),
  as 3DGS's sparse/selective Adam: unseen rows keep their state and do not drift on old momentum;
- at a sync only rows trained since the last one are compared with what was last written, on the
  GPU, and only those that changed visibly go back to the map (refined: they publish like any
  update); rows trained transparent retire.
ITO_MAPFIT_SCALE: training image scale relative to the view's photo grid (1: Ito's photo grid, twice
the network's; 0.5: the network's grid). ITO_FIT_DUTY as ViewFit (0.5: rest as long as it worked).
ITO_MAPFIT_POSES=0 keeps views where the estimator put them."""

import logging
import math
import os
import time

import numpy as np

from .gaussian import AXES, C0

log = logging.getLogger(__name__)

SYNC = int(os.environ.get("ITO_MAPFIT_SYNC", "20"))  # steps between exchanges with the worker
STALE = 200  # steps after which a sync waits for the worker's lock
SCALE = float(os.environ.get("ITO_MAPFIT_SCALE", "1"))
DUTY = float(os.environ.get("ITO_FIT_DUTY", "0.5"))
POSES = os.environ.get("ITO_MAPFIT_POSES", "1") != "0"
# Columns: means 0-2, log scales 3-5, quaternion 6-9, opacity logit 10, colour (SH0) 11-13.
RATES = [3.2e-4] * 3 + [5e-3] * 3 + [1e-3] * 4 + [5e-2] + [2.5e-3] * 3
BETAS = (0.9, 0.999)
EPS = 1e-15
POSE_RATE = 2e-4  # radians, metres per Adam step
POSE_LIMIT = (math.radians(3), 0.05)  # a view corrected further is left where it was
SSIM_WEIGHT = 0.2
RECENT = 3  # the newest views get half of all steps
PRUNE_OPACITY = 0.02
MOVED, COLOUR, RESHAPED, REFADED = 1e-4, 1 / 255, 0.02, 0.02  # smaller changes are not published


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


class MapFit:
    def __init__(self, backend):
        import torch
        from gsplat import rasterization

        self.torch, self.rasterize = torch, rasterization
        self.backend = backend
        self.active = None  # ViewFit's handle; recover() clears it (nothing to drop here)
        self.capacity = 0
        self.rows = None  # (C, 14) parameters
        self.m = self.v = None  # Adam moments
        self.count = None  # Adam steps per row
        self.written = None  # (C, 14) as last written to / read from the map
        self.touched = None  # rows trained since the last sync
        self.live = None  # indices of live rows
        self.rates = torch.tensor(RATES, device="cuda")
        self.images = {}  # id(view) -> (view, image tensor, intrinsics)
        self.deltas = {}  # id(view) -> (rotation, translation, view, base pose)
        self.pose_optimizer = None
        self.steps = self.syncs = self.written_rows = self.taken_rows = self.pruned = 0
        self.synced = 0  # step of the last sync
        self.views = []
        self.aligned = 0
        self.seconds = dict(sync=0.0, step=0.0)
        self.rng = np.random.default_rng(0)

    # -- thread ---------------------------------------------------------------------------------
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
                        self.reset()
                        failed(exc)
                        torch.cuda.empty_cache()
                        time.sleep(1.0)
                        continue
                    except Exception:
                        errors += 1
                        self.reset()
                        log.exception("Map training step failed")
                        if errors >= 5:
                            log.error("Map training stopped after repeated failures")
                            return
                        time.sleep(0.5)
                        continue
                    if not progressed:
                        time.sleep(0.05)

        thread = threading.Thread(target=run, name="map-training", daemon=True)
        thread.start()
        return thread

    def rescaled(self):
        """The worker rescaled the map (all slots fresh): view pose corrections start over."""
        self.deltas.clear()
        self.pose_optimizer = None

    def reset(self):
        """Forget the GPU copy; the next sync takes the whole map in afresh."""
        self.capacity = 0
        self.rows = None
        b = self.backend
        with b.lock:
            b.fresh[:] = True

    # -- exchange with the worker ---------------------------------------------------------------
    def to_rows(self, records):
        torch = self.torch
        r = torch.as_tensor(records, device="cuda", dtype=torch.float32)
        rows = torch.empty((len(r), 14), device="cuda")
        rows[:, 0:3] = r[:, 0, :3]
        rows[:, 3:6] = r[:, 1, :3].clamp_min(1e-6).log()
        rows[:, 6:10] = torch.nn.functional.normalize(r[:, 2], dim=-1)
        rows[:, 10] = torch.logit(r[:, 0, 3].clamp(0.001, 0.999))
        rows[:, 11:14] = r[:, 3, :3]
        return rows

    def resize(self, n):
        torch = self.torch
        if n == self.capacity and self.rows is not None:
            return
        old = self.capacity if self.rows is not None else 0
        keep = min(old, n)

        def fit(t, width, dtype=torch.float32):
            shape = (n, width) if width else (n,)
            out = torch.zeros(shape, device="cuda", dtype=dtype)
            if t is not None and keep:
                out[:keep] = t[:keep]
            return out

        self.rows = fit(self.rows, 14)
        self.m, self.v = fit(self.m, 14), fit(self.v, 14)
        self.count = fit(self.count, 0)
        self.written = fit(self.written, 14)
        self.touched = fit(self.touched, 0, torch.bool)
        self.capacity = n
        if n > old:
            self.backend.fresh[old:n] = True

    def sync(self):
        """Under the map's lock: write trained rows back, take in what the worker changed."""
        torch, b = self.torch, self.backend
        n = len(b.keys)
        self.resize(n)
        live_mask = (b.keys >= 0) & ~b.retiring
        # 1. Trained rows that changed visibly go to the map, unless the worker changed the slot.
        touched = torch.nonzero(self.touched).flatten()
        if len(touched):
            new, old = self.rows[touched], self.written[touched]
            moved = (new[:, 0:3] - old[:, 0:3]).norm(dim=1) > MOVED
            recoloured = (new[:, 11:14] - old[:, 11:14]).abs().amax(1) * C0 > COLOUR
            reshaped = (new[:, 3:6] - old[:, 3:6]).abs().amax(1) > RESHAPED
            opacity, before = torch.sigmoid(new[:, 10]), torch.sigmoid(old[:, 10])
            refaded = (opacity - before).abs() > REFADED
            faded = opacity < PRUNE_OPACITY
            send = (moved | recoloured | reshaped | refaded | faded) & torch.isfinite(new).all(1)
            index = touched[send]
            if len(index):
                rows = new[send]
                q = torch.nn.functional.normalize(rows[:, 6:10], dim=-1)
                values = torch.cat(
                    (
                        rows[:, 0:3],
                        torch.sigmoid(rows[:, 10:11]),
                        rows[:, 3:6].exp(),
                        q,
                        rows[:, 11:14],
                    ),
                    1,
                )
                slots = index.cpu().numpy()
                values = values.cpu().numpy()
                ok = live_mask[slots] & ~b.fresh[slots]
                slots, values = slots[ok], values[ok]
                fade = values[:, 3] < PRUNE_OPACITY
                writing = slots[~fade]
                v = values[~fade]
                b.records[writing, 0, :3] = v[:, 0:3]
                b.records[writing, 0, 3] = v[:, 3]
                b.records[writing, 1, :3] = v[:, 4:7]
                b.records[writing, 2] = v[:, 7:11]
                b.records[writing, 3, :3] = v[:, 11:14]
                b.refined[writing] = True
                self.written[index] = new[send]
                self.written_rows += len(writing)
                if fade.any():
                    b.retire(slots[fade])
                    live_mask[slots[fade]] = False
                    self.pruned += int(fade.sum())
            self.touched[:] = False
        # 2. Rows the worker wrote since the last sync come in fresh, with no Adam state.
        fresh = np.flatnonzero(b.fresh[:n])
        b.fresh[:n] = False
        if len(fresh):
            take = fresh[live_mask[fresh]]
            if len(take):
                index = torch.as_tensor(take, device="cuda")
                rows = self.to_rows(b.records[take])
                self.rows[index] = rows
                self.written[index] = rows
                self.m[index] = 0
                self.v[index] = 0
                self.count[index] = 0
            self.taken_rows += len(take)
        self.live = torch.as_tensor(np.flatnonzero(live_mask), device="cuda")
        self.syncs += 1

    # -- views ----------------------------------------------------------------------------------
    def choose(self):
        views = self.views = list(self.backend.views)  # one consistent snapshot (lock-free)
        if not views:
            return None
        if self.rng.random() < 0.5:
            return views[-1 - int(self.rng.integers(0, min(RECENT, len(views))))]
        return views[int(self.rng.integers(0, len(views)))]

    def image(self, view):
        """The view's training image on the GPU (float, 3xHxW), its intrinsics; kept as bytes."""
        torch = self.torch
        found = self.images.get(id(view))
        if found is not None and found[0] is view:
            return found[1].float() / 255, found[2]
        fine = view.photo is not None
        image = view.photo if fine else view.image
        fx, fy, cx, cy = view.photo_intrinsics if fine else view.intrinsics
        tensor = torch.as_tensor(image, device="cuda").permute(2, 0, 1).float() / 255
        if SCALE != 1:
            h, w = tensor.shape[1:]
            size = (max(8, round(h * SCALE)), max(8, round(w * SCALE)))
            tensor = torch.nn.functional.interpolate(tensor[None], size=size, mode="area")[0]
            sy, sx = size[0] / h, size[1] / w
            fx, fy, cx, cy = fx * sx, fy * sy, (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5
        stored = (tensor * 255).round().clamp(0, 255).to(torch.uint8)
        self.images[id(view)] = (view, stored, (fx, fy, cx, cy))
        return stored.float() / 255, (fx, fy, cx, cy)

    def delta(self, view):
        """The view's pose correction parameters (None for the gauge: the oldest view)."""
        torch = self.torch
        if not POSES or view is self.views[0]:
            return None
        found = self.deltas.get(id(view))
        if found is None or found[2] is not view:
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

    def forget_views(self):
        retained = {id(v) for v in self.backend.views}
        for key in list(self.images):
            if key not in retained:
                del self.images[key]
        for key in list(self.deltas):
            if key not in retained:
                del self.deltas[key]
                self.pose_optimizer = None

    def apply_poses(self):
        """Under the map's lock: views' corrected training poses."""
        for rotation, translation, view, base in self.deltas.values():
            with self.torch.no_grad():
                angle, shift = float(rotation.norm()), float(translation.norm())
                if angle > POSE_LIMIT[0] or shift > POSE_LIMIT[1]:
                    continue
                camera = self.moved(rotation, translation) @ self.torch.as_tensor(
                    np.linalg.inv(base @ AXES), device="cuda", dtype=self.torch.float32
                )
                view.fitting_pose = np.linalg.inv(camera.double().cpu().numpy()) @ AXES
        self.aligned = len(self.deltas)

    # -- one step -------------------------------------------------------------------------------
    def step(self):
        torch, b = self.torch, self.backend
        # The worker holds the map's lock for the whole of each frame (~200 ms); a step must not
        # wait for it. Syncs are taken when the lock is free (between the worker's iterations), and
        # forced only when none got through for STALE steps.
        due = self.rows is None or self.steps - self.synced >= SYNC
        if due:
            began = time.perf_counter()
            forced = self.rows is None or self.steps - self.synced >= STALE
            if b.lock.acquire(blocking=forced):
                try:
                    self.sync()
                    self.forget_views()
                    self.apply_poses()
                finally:
                    b.lock.release()
                self.synced = self.steps
            self.seconds["sync"] += time.perf_counter() - began
        self.steps += 1
        if self.live is None or not len(self.live):
            return False
        began = time.perf_counter()
        # Lock-free: views are only appended/removed by the worker (atomic list operations), a
        # view's images never change, and a pose is read as one reference.
        view = self.choose()
        if view is None:
            return False
        delta = self.delta(view)
        base = (
            delta[3]
            if delta is not None
            else (view.fitting_pose if view.fitting_pose is not None else view.pose)
        )
        target, (fx, fy, cx, cy) = self.image(view)
        h, w = target.shape[1:]
        camera = torch.as_tensor(np.linalg.inv(base @ AXES), device="cuda", dtype=torch.float32)
        if delta is not None:
            camera = self.moved(delta[0], delta[1]) @ camera
        live = self.live
        leaf = self.rows[live].detach().requires_grad_(True)
        rendered, _, meta = self.rasterize(
            means=leaf[:, 0:3],
            quats=leaf[:, 6:10],
            scales=leaf[:, 3:6].exp(),
            opacities=leaf[:, 10].sigmoid(),
            colors=leaf[:, None, 11:14],
            sh_degree=0,
            viewmats=camera[None],
            Ks=torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device="cuda")[None],
            width=w,
            height=h,
            near_plane=0.01,
            far_plane=1000,
            packed=False,
        )
        rgb = rendered[0].permute(2, 0, 1)
        loss = (1 - SSIM_WEIGHT) * (rgb - target).abs().mean() + SSIM_WEIGHT * (
            1 - ssim(rgb, target, torch)
        )
        loss.backward()
        with torch.no_grad():
            seen = meta["radii"][0] > 0
            if seen.dim() > 1:
                seen = seen.any(-1)
            sub = torch.nonzero(seen).flatten()
            index = live[sub]
            grad = leaf.grad[sub]
            count = self.count[index] + 1
            m = BETAS[0] * self.m[index] + (1 - BETAS[0]) * grad
            v = BETAS[1] * self.v[index] + (1 - BETAS[1]) * grad * grad
            m_hat = m / (1 - BETAS[0] ** count)[:, None]
            v_hat = v / (1 - BETAS[1] ** count)[:, None]
            rows = self.rows[index] - self.rates * m_hat / (v_hat.sqrt() + EPS)
            rows[:, 6:10] = torch.nn.functional.normalize(rows[:, 6:10], dim=-1)
            rows[:, 10].clamp_(-7, 7)
            rows[:, 11:14].clamp_(-0.5 / C0, 0.5 / C0)
            self.rows[index] = rows
            self.m[index], self.v[index], self.count[index] = m, v, count
            self.touched[index] = True
        if delta is not None:
            if self.pose_optimizer is None:
                params = [x for d in self.deltas.values() for x in d[:2]]
                self.pose_optimizer = torch.optim.Adam(params, lr=POSE_RATE)
            self.pose_optimizer.step()
            self.pose_optimizer.zero_grad(set_to_none=True)
        view.steps += 1
        b.fit_steps += 1
        self.seconds["step"] += time.perf_counter() - began
        if self.steps % 1000 == 0:
            log.info(
                "Map training: %d steps, %d syncs, %.1f s in steps, %.1f s in syncs, "
                "%d rows written, %d taken, %d pruned, %d views posed",
                self.steps,
                self.syncs,
                self.seconds["step"],
                self.seconds["sync"],
                self.written_rows,
                self.taken_rows,
                self.pruned,
                self.aligned,
            )
        return True
