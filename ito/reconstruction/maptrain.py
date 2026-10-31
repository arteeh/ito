"""Map trainer (proof of concept, #82; Settings > MASt3R-SLAM > Map trainer, off by default).

MASt3R-SLAM places each keyframe's pointmap as splats (slam.py); this thread then trains those
splats photometrically against the camera's own frames, as 3D Gaussian splatting does: gsplat
renders the map from a training view, an L1 + SSIM loss against the photo, Adam on the splats
that view saw. Training views are the keyframes and extra tracked frames between them, each held
relative to a keyframe, so a global solve or loop closure moves them with it.

From the sprint 2 whole-map trainer (ITO_FIT=map, e60b377): every slot has a fixed row in GPU
tensors (parameters, Adam moments, a step count), so nothing is rebuilt; at a sync, under the
map's lock and never while the worker tracks, the trainer takes in the slots the worker wrote
(fresh: seeded, retired, freed), applies the moves the worker queued, and writes back the rows
it changed visibly, which then publish like any update. A step itself takes no lock.

Moves: with the trainer on, the worker no longer redraws a keyframe the solve moved (that would
throw its training away); it queues the keyframe's pose change as a similarity of Ito's world,
and the trainer moves that keyframe's splats (means, rotations, sizes) by it. Each queued move has
a sequence number and each slot write the number of moves queued before it, so a move applies
exactly to the splats written before it.

GPU budget: the trainer runs on its own stream below the worker's in priority, steps only while
the pilot's renderer leaves the GPU to reconstruction (admission), and rests as long as it worked
(DUTY 0.5: at most half the time). Tracking never waits for it."""

import collections
import logging
import math
import os
import threading
import time

import numpy as np

log = logging.getLogger(__name__)

C0 = 0.28209479177387814  # SH0: colour = 0.5 + C0 * coefficient
AXES = np.array([1.0, -1.0, -1.0])  # OpenCV camera/world axes to Ito's
SYNC = 20  # steps between exchanges with the worker
STALE = 200  # steps after which a sync waits for the map's lock
DUTY = float(os.environ.get("ITO_MAPTRAIN_DUTY", "0.5"))
# Training photos' long side, in MASt3R input long sides (2: 1024 at 100 % input).
SCALE = float(os.environ.get("ITO_MAPTRAIN_SCALE", "2"))
EXTRA = 96  # extra (non-keyframe) training views kept; the most redundant goes first
VIEW_EVERY = 3  # the worker offers every this-many-th tracked frame as an extra view
# An offered view this close to a kept one (distance over the keyframe's depth + angle in
# radians) adds nothing.
SAME_VIEW = 0.08
# Columns: means 0-2, log scales 3-5, quaternion (wxyz) 6-9, opacity logit 10, colour SH0 11-13.
RATES = [3.2e-4] * 3 + [5e-3] * 3 + [1e-3] * 4 + [5e-2] + [2.5e-3] * 3
BETAS = (0.9, 0.999)
EPS = 1e-15
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


def quaternion(rotation):
    """wxyz of a 3x3 rotation matrix (numpy)."""
    m = rotation
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2 * math.sqrt(trace + 1)
        q = (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2 * math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
        q = ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = 2 * math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
        q = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)
    else:
        s = 2 * math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
        q = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)
    q = np.array(q)
    return q / np.linalg.norm(q)


def similarity(delta, scale, world):
    """A keyframe's Sim(3) pose change in upstream's world (delta = new * old^-1, a 4x4 with
    sR), as a 4x4 similarity of Ito's world: splats are placed at Rw (A * scale * p) + tw."""
    s = float(np.cbrt(np.linalg.det(delta[:3, :3])))
    a = np.diag(AXES)
    rw, tw = world[:3, :3], world[:3, 3]
    rotation = rw @ a @ (delta[:3, :3] / s) @ a @ rw.T
    out = np.eye(4)
    out[:3, :3] = s * rotation
    out[:3, 3] = tw - s * rotation @ tw + scale * rw @ (AXES * delta[:3, 3])
    return out


def rescaled(ratio, world):
    """The display scale changed by ratio: splats scale about the map's origin."""
    out = np.eye(4)
    out[:3, :3] *= ratio
    out[:3, 3] = world[:3, 3] * (1 - ratio)
    return out


class Photos:
    """Crop and scale of the driver's frames for training: MASt3R's crop (slam.crop) at SCALE
    times its long side, and the pinhole of that photo."""

    def __init__(self, intrinsics, long):
        from .slam import crop_size

        w1, h1 = intrinsics.width, intrinsics.height
        size = max(w1, h1)
        self.size = (int(round(w1 * long / size)), int(round(h1 * long / size)))
        halfw, halfh = crop_size(*self.size)
        w, h = self.size
        self.x0, self.y0 = w // 2 - halfw, h // 2 - halfh
        self.width, self.height = 2 * halfw, 2 * halfh
        sx, sy = w / w1, h / h1
        self.K = (
            intrinsics.fx * sx,
            intrinsics.fy * sy,
            (intrinsics.cx + 0.5) * sx - 0.5 - self.x0,
            (intrinsics.cy + 0.5) * sy - 0.5 - self.y0,
        )

    def photo(self, rgb, torch):
        image = torch.as_tensor(rgb, device="cuda").permute(2, 0, 1)[None].float()
        w, h = self.size
        image = torch.nn.functional.interpolate(
            image, size=(h, w), mode="bilinear", antialias=True, align_corners=False
        )[0]
        image = image[:, self.y0 : self.y0 + self.height, self.x0 : self.x0 + self.width]
        return image.round().clamp(0, 255).to(torch.uint8).contiguous()


class MapTrainer:
    def __init__(self, backend, intrinsics, long):
        import torch
        from gsplat import rasterization

        self.torch, self.rasterize = torch, rasterization
        self.backend = backend
        self.photos = Photos(intrinsics, round(long * SCALE))
        self.K = torch.tensor(
            [
                [self.photos.K[0], 0, self.photos.K[2]],
                [0, self.photos.K[1], self.photos.K[3]],
                [0, 0, 1],
            ],
            device="cuda",
            dtype=torch.float32,
        )
        self.rates = torch.tensor(RATES, device="cuda")
        self.capacity = 0
        self.rows = self.m = self.v = self.count = self.written = self.touched = None
        self.row_source = self.row_seq = None
        self.live = None
        # Worker -> trainer, appended by the worker (deque appends and pops are atomic).
        self.inbox = collections.deque()  # photos: ("kf", k, gen, rgb) / ("view", ..., rel, event)
        self.moves = collections.deque()  # (source or -1 for all, 4x4 similarity, sequence)
        self.snapshot = None  # the worker's keyframe poses after its last frame (see publish)
        # Trainer's own.
        self.keyframe_photos = {}  # k -> (gen, photo)
        self.extras = []  # dicts: k, gen, rel, photo, event, order
        self.views = []  # (photo, viewmat, order) to train on, rebuilt at each sync
        self.order = 0
        self.steps = self.syncs = self.written_rows = self.taken_rows = self.pruned = 0
        self.moved_rows = 0
        self.synced = 0
        self.error = None
        self.seconds = dict(sync=0.0, step=0.0, rest=0.0)
        self.rng = np.random.default_rng(0)
        self.thread = None

    # -- worker side ------------------------------------------------------------------------------
    def publish(self, keyframes, n, scale, world, gens, depth):
        """The worker's keyframe poses after a frame (under the map's lock): copies, and an
        event the trainer's stream waits on before reading them."""
        torch = self.torch
        poses = keyframes.T_WC[:n].clone()
        depth = depth[:n].clone()
        event = torch.cuda.Event()
        event.record()
        self.snapshot = (poses, n, float(scale), world.copy(), gens[:n].copy(), depth, event)

    # -- thread ---------------------------------------------------------------------------------
    def start(self, available, lock):
        torch = self.torch
        self.lock = lock

        def run():
            # Below the worker's stream (slam.py gives it -1 with the trainer on): when both have
            # work, the GPU schedules tracking's blocks first.
            stream = torch.cuda.Stream(priority=0)
            errors = 0
            with torch.cuda.stream(stream):
                while True:
                    if not available():
                        time.sleep(0.002)
                        continue
                    try:
                        began = time.perf_counter()
                        progressed = self.step()
                        torch.cuda.current_stream().synchronize()
                        if progressed and 0 < DUTY < 1:
                            rest = (time.perf_counter() - began) * (1 / DUTY - 1)
                            self.seconds["rest"] += rest
                            time.sleep(rest)
                        errors = 0
                    except torch.cuda.OutOfMemoryError as exc:
                        self.error = "map trainer out of VRAM: training paused"
                        log.warning("Map trainer: %s", exc)
                        self.reset()
                        torch.cuda.empty_cache()
                        time.sleep(1.0)
                        continue
                    except Exception:
                        errors += 1
                        log.exception("Map trainer step failed")
                        self.reset()
                        if errors >= 5:
                            self.error = "map trainer stopped after repeated failures"
                            log.error("Map trainer stopped after repeated failures")
                            return
                        time.sleep(0.5)
                        continue
                    if not progressed:
                        time.sleep(0.05)

        self.thread = threading.Thread(target=run, name="map-trainer", daemon=True)
        self.thread.start()

    def reset(self):
        """Forget the GPU copy; the next sync takes the whole map in afresh. Moves already
        applied to rows not yet written back are lost with them (a PoC's edge)."""
        self.capacity = 0
        self.rows = None
        with self.lock:
            self.backend.fresh[:] = True

    # -- exchange with the worker (under the map's lock) ----------------------------------------
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
            out = torch.zeros((n, width) if width else (n,), device="cuda", dtype=dtype)
            if t is not None and keep:
                out[:keep] = t[:keep]
            return out

        self.rows = fit(self.rows, 14)
        self.m, self.v = fit(self.m, 14), fit(self.v, 14)
        self.count = fit(self.count, 0)
        self.written = fit(self.written, 14)
        self.touched = fit(self.touched, 0, torch.bool)
        self.row_source = fit(self.row_source, 0, torch.int32)
        self.row_seq = fit(self.row_seq, 0, torch.int64)
        self.capacity = n
        if n > old:
            self.backend.fresh[old:n] = True

    def move(self, source, matrix, sequence):
        """Move the rows written before this move (and owned by source, unless -1)."""
        torch = self.torch
        mask = self.row_seq < sequence
        if source >= 0:
            mask &= self.row_source == source
        index = torch.nonzero(mask).flatten()
        if not len(index):
            return
        s = float(np.cbrt(np.linalg.det(matrix[:3, :3])))
        sr = torch.as_tensor(matrix[:3, :3], device="cuda", dtype=torch.float32)
        t = torch.as_tensor(matrix[:3, 3], device="cuda", dtype=torch.float32)
        rows = self.rows[index]
        rows[:, 0:3] = rows[:, 0:3] @ sr.T + t
        rows[:, 3:6] += math.log(s)
        w, x, y, z = (
            torch.as_tensor(v, device="cuda", dtype=torch.float32)
            for v in quaternion(matrix[:3, :3] / s)
        )
        qw, qx, qy, qz = rows[:, 6], rows[:, 7], rows[:, 8], rows[:, 9]
        rows[:, 6:10] = torch.stack(
            (
                w * qw - x * qx - y * qy - z * qz,
                w * qx + x * qw + y * qz - z * qy,
                w * qy - x * qz + y * qw + z * qx,
                w * qz + x * qy - y * qx + z * qw,
            ),
            1,
        )
        self.rows[index] = rows
        self.touched[index] = True
        self.moved_rows += len(index)

    def sync(self):
        torch, b = self.torch, self.backend
        n = len(b.keys)
        self.resize(n)
        live_mask = (b.keys >= 0) & ~b.retiring
        # 1. Rows the worker wrote since the last sync come in fresh, with no Adam state; the
        # worker saw the newer frame, so its write wins over training.
        fresh = np.flatnonzero(b.fresh[:n])
        b.fresh[:n] = False
        if len(fresh):
            index = torch.as_tensor(fresh, device="cuda")
            self.touched[index] = False
            take = fresh[live_mask[fresh]]
            if len(take):
                index = torch.as_tensor(take, device="cuda")
                rows = self.to_rows(b.records[take])
                self.rows[index] = rows
                self.written[index] = rows
                self.m[index] = 0
                self.v[index] = 0
                self.count[index] = 0
                self.row_source[index] = torch.as_tensor(b.sources[take], device="cuda")
                self.row_seq[index] = torch.as_tensor(b.moves[take], device="cuda")
            self.taken_rows += len(take)
        # 2. The worker's queued moves, in order.
        latest = None
        while self.moves:
            source, matrix, sequence = self.moves.popleft()
            self.move(source, matrix, sequence)
            latest = sequence
        if latest is not None:
            self.row_seq.clamp_(min=latest)
        # 3. Trained (or moved) rows that changed visibly go to the map.
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
                values = torch.cat(
                    (
                        rows[:, 0:3],
                        torch.sigmoid(rows[:, 10:11]),
                        rows[:, 3:6].exp(),
                        torch.nn.functional.normalize(rows[:, 6:10], dim=-1),
                        rows[:, 11:14],
                    ),
                    1,
                )
                slots = index.cpu().numpy()
                values = values.cpu().numpy()
                sequences = self.row_seq[index].cpu().numpy()
                ok = live_mask[slots]
                slots, values, sequences = slots[ok], values[ok], sequences[ok]
                fade = values[:, 3] < PRUNE_OPACITY
                writing, v = slots[~fade], values[~fade]
                b.records[writing, 0, :3] = v[:, 0:3]
                b.records[writing, 0, 3] = v[:, 3]
                b.records[writing, 1, :3] = v[:, 4:7]
                b.records[writing, 2] = v[:, 7:11]
                b.records[writing, 3, :3] = v[:, 11:14]
                b.moves[writing] = sequences[~fade]
                b.dirty[writing] = True
                self.written[index] = new[send]
                self.written_rows += len(writing)
                if fade.any():
                    b.retire(slots[fade])  # marks them fresh: they leave the live rows
                    live_mask[slots[fade]] = False
                    self.pruned += int(fade.sum())
            self.touched[:] = False
        self.live = torch.as_tensor(np.flatnonzero(live_mask), device="cuda")
        self.syncs += 1
        return self.snapshot

    # -- views (lock-free) -----------------------------------------------------------------------
    def take_inbox(self, limit=2):
        torch = self.torch
        for _ in range(limit):
            if not self.inbox:
                return
            item = self.inbox.popleft()
            if item[0] == "kf":
                _, k, gen, rgb = item
                self.keyframe_photos[k] = (gen, self.photos.photo(rgb, torch), self.order)
                # A keyframe slot used again (a full map's spare): views held to it go.
                self.extras = [e for e in self.extras if not (e["k"] == k and e["gen"] != gen)]
            else:
                _, k, gen, rel, event, rgb = item
                torch.cuda.current_stream().wait_event(event)
                photo = self.photos.photo(rgb, torch)
                self.extras.append(dict(k=k, gen=gen, rel=rel, photo=photo, order=self.order))
            self.order += 1

    def viewmats(self, data, scale, world):
        """World-to-camera (OpenCV) 4x4s, Ito's world, of upstream Sim(3) camera poses."""
        torch = self.torch
        import lietorch

        m = lietorch.Sim3(data).matrix().reshape(-1, 4, 4)
        s = data.reshape(-1, 8)[:, 7:8, None]
        rotation = m[:, :3, :3] / s
        a = torch.tensor(AXES, device="cuda", dtype=torch.float32)
        rw = torch.as_tensor(world[:3, :3], device="cuda", dtype=torch.float32)
        tw = torch.as_tensor(world[:3, 3], device="cuda", dtype=torch.float32)
        rc = rw @ (a[:, None] * rotation)
        tc = (rw @ (a * m[:, :3, 3] * scale)[..., None])[..., 0] + tw
        out = torch.zeros_like(m)
        out[:, :3, :3] = rc.transpose(1, 2)
        out[:, :3, 3] = -(rc.transpose(1, 2) @ tc[..., None])[..., 0]
        out[:, 3, 3] = 1
        return out, tc

    def rebuild(self, snapshot):
        """Training views at the poses of the worker's latest snapshot."""
        torch = self.torch
        import lietorch

        poses, n, scale, world, gens, depth, event = snapshot
        torch.cuda.current_stream().wait_event(event)
        with torch.no_grad():
            mats, _ = self.viewmats(poses, scale, world)
            views = []
            for k, (gen, photo, order) in self.keyframe_photos.items():
                if k < n and gens[k] == gen:
                    views.append((photo, mats[k], order))
            extras = [e for e in self.extras if e["k"] < n and gens[e["k"]] == e["gen"]]
            if extras:
                ks = torch.tensor([e["k"] for e in extras], device="cuda")
                rel = torch.stack([e["rel"].reshape(8) for e in extras])
                placed = (lietorch.Sim3(poses.reshape(-1, 8)[ks]) * lietorch.Sim3(rel)).data
                emats, ecentres = self.viewmats(placed, scale, world)
                # Drop what repeats a kept view: the newest of a too-close pair, then, over the
                # cap, the one closest to another.
                reach = (depth[ks] * scale).clamp_min(1e-3)
                close = (ecentres[:, None] - ecentres[None]).norm(dim=-1) / reach[:, None]
                forward = emats[:, 2, :3]
                angle = torch.acos((forward @ forward.T).clamp(-1, 1))
                near = (close + angle).cpu().numpy()
                np.fill_diagonal(near, np.inf)
                keep = np.ones(len(extras), bool)
                for i in range(len(extras) - 1, 0, -1):
                    if near[i, :i][keep[:i]].min() < SAME_VIEW:
                        keep[i] = False
                while keep.sum() > EXTRA:
                    kept = np.flatnonzero(keep)
                    keep[kept[near[np.ix_(kept, kept)].min(1).argmin()]] = False
                self.extras = [e for e, k in zip(extras, keep, strict=True) if k]
                for i, e in enumerate(extras):
                    if keep[i]:
                        views.append((e["photo"], emats[i], e["order"]))
            else:
                self.extras = []
        views.sort(key=lambda v: v[2])
        self.views = views

    def choose(self):
        views = self.views
        if not views:
            return None
        if self.rng.random() < 0.5:
            return views[-1 - int(self.rng.integers(0, min(RECENT, len(views))))]
        return views[int(self.rng.integers(0, len(views)))]

    # -- one step -------------------------------------------------------------------------------
    def step(self):
        torch = self.torch
        self.take_inbox()
        due = self.rows is None or self.steps - self.synced >= SYNC or bool(self.moves)
        if due:
            began = time.perf_counter()
            forced = self.rows is None or self.steps - self.synced >= STALE or bool(self.moves)
            snapshot = None
            if self.lock.acquire(blocking=forced):
                try:
                    snapshot = self.sync()
                finally:
                    self.lock.release()
                self.synced = self.steps
                if snapshot is not None:
                    self.rebuild(snapshot)
            self.seconds["sync"] += time.perf_counter() - began
        self.steps += 1
        if self.live is None or not len(self.live):
            return False
        view = self.choose()
        if view is None:
            return False
        began = time.perf_counter()
        photo, viewmat, _ = view
        target = photo.float() / 255
        h, w = target.shape[1:]
        live = self.live
        leaf = self.rows[live].detach().requires_grad_(True)
        rendered, _, meta = self.rasterize(
            means=leaf[:, 0:3],
            quats=leaf[:, 6:10],
            scales=leaf[:, 3:6].exp(),
            opacities=leaf[:, 10].sigmoid(),
            colors=leaf[:, None, 11:14],
            sh_degree=0,
            viewmats=viewmat[None],
            Ks=self.K[None],
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
        self.seconds["step"] += time.perf_counter() - began
        if self.steps % 1000 == 0:
            log.info(
                "Map trainer: %d steps, %d syncs, %.1f s steps, %.1f s syncs, %.1f s rest, "
                "%d rows written, %d taken, %d moved, %d pruned, %d views",
                self.steps,
                self.syncs,
                self.seconds["step"],
                self.seconds["sync"],
                self.seconds["rest"],
                self.written_rows,
                self.taken_rows,
                self.moved_rows,
                self.pruned,
                len(self.views),
            )
        return True
