"""Full MASt3R-SLAM backend (ITO_SLAM=full): every keyframe in one global Sim(3) graph.

Upstream MASt3R-SLAM (Murai et al., CVPR 2025) keeps every keyframe, links each new one to the one
before it and to up to three earlier ones its retrieval database (ASMK over MASt3R features)
returns, checks every link by symmetric matching, and re-solves all keyframes' poses by
Gauss-Newton on calibrated reprojection and log-depth errors, on a process of its own. A keyframe
made while tracking was lost is relocalised the same way, strictly. This module is that backend,
on a thread of the reconstruction process, with upstream's FactorGraph.add_factors, its
RetrievalDatabase and its calib_proj_kernel; what Ito adds goes into the same normal equations:

- the robot's factors per keyframe: odometry to the keyframe before it, gravity, the camera's
  height over the floor, and the ToF's metres per unit of the keyframe's pointmap (so the Sim(3)
  graph cannot drift its scale away from the sensors);
- a gauge that keeps the keyframe the pilot is looking from where it was: a solve moves the rest
  of the room to it, never the view;
- results go back to the tracking thread as per-keyframe corrections, applied there to the
  keyframes, the pose graph's window and every surface anchored to them (slam.py).

The worker does its GPU work on a stream below tracking's and waits while the renderer asks the
GPU to back off; its native calls release the GIL (native/mast3r_full.cu).
"""

import logging
import math
import queue
import threading
import time

import numpy as np

log = logging.getLogger(__name__)

AXES = np.array([1.0, -1.0, -1.0])  # Ito world/camera axes to MASt3R's (OpenCV) and back.
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])
ITERATIONS = 10  # upstream local_opt max_iters
SCALE_FLOOR = 0.02  # log-scale sigma floor for a keyframe's ToF prior, however many readings
TRACKING_FRAC = (
    0.0  # the tracking edge is kept whatever its match fraction (upstream's consecutive)
)
# Added to the normal equations' diagonal: holds a keyframe no factor fixes in some direction
# (relocalisation failed, no odometry) where it is; negligible beside any real factor.
DAMPING = 1e-3


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def rotvec(rotation):
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(rotation).as_rotvec()


def sim3_matrix(data):
    """lietorch Sim3 data (tx ty tz qx qy qz qw s), any leading shape, to 4x4 similarities."""
    from scipy.spatial.transform import Rotation

    data = np.asarray(data, np.float64).reshape(-1, 8)
    m = np.tile(np.eye(4), (len(data), 1, 1))
    m[:, :3, :3] = Rotation.from_quat(data[:, 3:7]).as_matrix() * data[:, 7, None, None]
    m[:, :3, 3] = data[:, :3]
    return m


def matrix_sim3(m):
    from scipy.spatial.transform import Rotation

    m = np.asarray(m, np.float64).reshape(-1, 4, 4)
    s = np.cbrt(np.linalg.det(m[:, :3, :3]))
    u, _, vh = np.linalg.svd(m[:, :3, :3] / s[:, None, None])
    r = u @ vh
    return np.concatenate((m[:, :3, 3], Rotation.from_matrix(r).as_quat(), s[:, None]), 1)


def split(m):
    s = np.cbrt(np.linalg.det(m[:3, :3]))
    return s, m[:3, :3] / s, m[:3, 3]


def to_ito(m):
    """A world similarity in MASt3R's axes to Ito's."""
    return FLIP @ m @ FLIP


class _Feature:
    def __init__(self, feat):
        self.feat = feat


class GlobalGraph:
    """frontend: the MASt3RFrontend whose keyframes (frames, ids) this graph keeps; available():
    the renderer has not asked the GPU to back off; stopped(): end the worker."""

    def __init__(self, frontend, report, available, stopped, tracking_stream=None):
        import torch

        from .mast3r_runtime import prepare_full

        self.native, checkpoint = prepare_full(report)
        from mast3r_slam.config import config
        from mast3r_slam.global_opt import FactorGraph

        self.torch = torch
        self.config = config
        self.frontend = frontend
        self.available = available
        self.stopped = stopped
        self.tracking_stream = tracking_stream
        self.graph = FactorGraph(frontend.model, frontend.frames, frontend.K, device="cuda")
        self.pairs = set()
        self.loops = []  # (keyframe, earlier keyframe) links retrieval found and matching kept
        self.relocalised = []
        self.lost = set()
        self.tracking = {}  # keyframe -> keyframe it was tracked against when made
        self.gravity = {}  # keyframe -> (robot camera rotation, MASt3R axes; sigma rad)
        self.height = {}  # keyframe -> (camera y, MASt3R axes; sigma m)
        self.odometry = []  # (i, j, rotation, translation, sigma[6]) robot motion i -> j
        self.scales = {}  # keyframe -> list of log metres per pointmap unit (ToF readings)
        self.owned = set()  # keyframes a solve has posed
        self.lock = threading.Lock()  # factors and the retrieval database
        self.tasks = queue.Queue()
        self.results = queue.Queue()
        self.reference = None  # keyframe the tracker last tracked against (the gauge)
        self.solves = 0
        self.solve_seconds = 0.0
        self.last_solve_seconds = 0.0
        self.connect_seconds = 0.0
        self.loop_seconds = []
        self.last_change = (0.0, 0.0)  # largest correction of the last solve: metres, degrees
        self.failed = 0
        with torch.inference_mode():
            self.retrieval = self._retriever(checkpoint, frontend.model)
        self.db = []  # retrieval database image id -> keyframe
        self.thread = threading.Thread(target=self._run, name="mast3r-global", daemon=True)
        self.thread.start()

    @staticmethod
    def _retriever(checkpoint, model):
        from mast3r_slam.retrieval_database import RetrievalDatabase

        return RetrievalDatabase(str(checkpoint), backbone=model, device="cuda")

    # --- called on the tracking thread ---------------------------------------------------

    def add_keyframe(self, index, *, tracked_against, lost, gravity, height, odometry):
        """A keyframe the frontend just kept, with the robot's say on it: gravity (rotation,
        sigma), height (y, sigma), odometry (previous keyframe, rotation, translation, sigma);
        any may be None. Poses are world-from-camera in MASt3R's axes."""
        with self.lock:
            if tracked_against is not None:
                self.tracking[index] = tracked_against
            if lost:
                self.lost.add(index)
            if gravity is not None:
                self.gravity[index] = gravity
            if height is not None:
                self.height[index] = height
            if odometry is not None:
                self.odometry.append((odometry[0], index, *odometry[1:]))
        self.tasks.put(index)

    def add_scale(self, index, log_scale):
        with self.lock:
            self.scales.setdefault(index, []).append(float(log_scale))

    def query(self, frame, k=3):
        """Keyframes the retrieval database finds for a frame, without adding it (relocalisation
        on the tracking thread). Empty until the database has images."""
        if frame.feat is None or not self.db:
            return []
        with self.lock, self.torch.inference_mode():
            found = self.retrieval.update(
                _Feature(frame.feat.float()),
                add_after_query=False,
                k=k,
                min_thresh=self.config["retrieval"]["min_thresh"],
            )
        return [self.db[i] for i in found if i < len(self.db)]

    def poll(self):
        """Solved corrections, oldest first: dicts with 'before' and 'after' (Sim3 data per
        keyframe, MASt3R axes) and 'gauge'."""
        out = []
        while True:
            try:
                out.append(self.results.get_nowait())
            except queue.Empty:
                return out

    def statistics(self):
        return dict(
            keyframes=len(self.frontend.frames),
            edges=int(self.graph.ii.numel()),
            loops=len(self.loops),
            relocalised=len(self.relocalised),
            lost=len(self.lost),
            solves=self.solves,
            solve_ms=round(self.last_solve_seconds * 1000, 1),
            solve_mean_ms=round(self.solve_seconds / max(1, self.solves) * 1000, 1),
            connect_ms=round(self.connect_seconds * 1000, 1),
            loop_ms=[round(s * 1000) for s in self.loop_seconds[-5:]],
            change=[round(v, 4) for v in self.last_change],
            failed=self.failed,
            queued=self.tasks.qsize(),
        )

    # --- worker thread ---------------------------------------------------------------

    def _wait(self):
        while not self.available():
            if self.stopped():
                raise InterruptedError
            time.sleep(0.002)
        if self.tracking_stream is not None:
            # Keyframe tensors are written on tracking's stream; read them after it.
            self.torch.cuda.current_stream().wait_stream(self.tracking_stream)

    def _run(self):
        torch = self.torch
        stream = torch.cuda.Stream(priority=0)
        errors = 0
        with torch.cuda.stream(stream), torch.inference_mode():
            while not self.stopped():
                try:
                    batch = [self.tasks.get(timeout=0.1)]
                except queue.Empty:
                    continue
                while True:
                    try:
                        batch.append(self.tasks.get_nowait())
                    except queue.Empty:
                        break
                try:
                    began = time.perf_counter()
                    loops = 0
                    for index in batch:
                        loops += self._connect(index)
                    torch.cuda.current_stream().synchronize()
                    self.connect_seconds = time.perf_counter() - began
                    solved = time.perf_counter()
                    self._solve()
                    torch.cuda.current_stream().synchronize()
                    self.last_solve_seconds = time.perf_counter() - solved
                    self.solve_seconds += self.last_solve_seconds
                    if loops:
                        self.loop_seconds.append(time.perf_counter() - began)
                    errors = 0
                except InterruptedError:
                    return
                except torch.cuda.OutOfMemoryError:
                    log.exception("Global graph out of GPU memory; it skips this keyframe's solve")
                    self.failed += 1
                    torch.cuda.empty_cache()
                    time.sleep(1.0)
                except Exception:
                    log.exception("Global graph step failed")
                    self.failed += 1
                    errors += 1
                    if errors >= 5:
                        log.error("Global graph stopped after repeated failures")
                        return
                    time.sleep(0.5)

    def _add(self, ii, jj, min_match_frac, is_reloc=False):
        """Upstream add_factors for pairs not linked yet; True if any edge was kept (with
        is_reloc, all or nothing)."""
        pairs = [
            (i, j) for i, j in zip(ii, jj, strict=True) if (min(i, j), max(i, j)) not in self.pairs
        ]
        if not pairs:
            return False
        self._wait()
        before = int(self.graph.ii.numel())
        added = self.graph.add_factors(
            [i for i, _ in pairs], [j for _, j in pairs], min_match_frac, is_reloc=is_reloc
        )
        ii, jj = self.graph.ii[before:].tolist(), self.graph.jj[before:].tolist()
        for i, j in zip(ii, jj, strict=True):
            self.pairs.add((min(i, j), max(i, j)))
        return bool(added)

    def _connect(self, n):
        """Upstream's backend step for keyframe n: retrieval (adding n to the database), the
        edge to the keyframe before it and to what retrieval found; a lost keyframe is
        relocalised strictly instead. Returns how many loop edges were kept."""
        cfg = self.config
        frame = self.frontend.frames[n]
        self._wait()
        with self.lock:
            found = self.retrieval.update(
                _Feature(frame.feat.float()),
                add_after_query=True,
                k=cfg["retrieval"]["k"],
                min_thresh=cfg["retrieval"]["min_thresh"],
            )
            found = [self.db[i] for i in found if i < len(self.db)]
            self.db.append(n)
            lost = n in self.lost
            tracked = self.tracking.get(n)
        candidates = [k for k in dict.fromkeys(found) if k != n]
        if lost:
            if candidates and self._add(
                [n] * len(candidates),
                candidates,
                cfg["reloc"]["min_match_frac"],
                is_reloc=cfg["reloc"]["strict"],
            ):
                self.relocalised.append((n, candidates))
                log.info("Relocalised keyframe %d against %s", n, candidates)
                return len(candidates)
            return 0
        if tracked is not None and tracked != n:
            # Upstream's consecutive edge: the keyframe tracking linked it to, always kept.
            self._add([tracked], [n], TRACKING_FRAC)
        others = [k for k in candidates if k != tracked]
        if n > 0 and n - 1 != tracked and n - 1 not in others:
            others.insert(0, n - 1)
        kept = 0
        if others:
            before = set(self.pairs)
            # (n, k), never (k, n): upstream keeps an edge (n - 1, n) unchecked as consecutive;
            # here the keyframe before need not overlap (tracking chose another), so every
            # edge but the tracking one passes upstream's two-way match-fraction check.
            self._add([n] * len(others), others, cfg["local_opt"]["min_match_frac"])
            for k in others:
                pair = (min(k, n), max(k, n))
                if pair in self.pairs and pair not in before and k != n - 1:
                    self.loops.append((n, k))
                    kept += 1
        return kept

    def _priors(self, poses, local):
        """The robot's factors at poses (n, 8 Sim3 data): 7x7 blocks (A, i, j) and gradients
        (b, k) of 0.5 r^T W r for left perturbations Exp(tau, phi, sigma) * T, as upstream's."""
        A, ai, aj, b, bk = [], [], [], [], []
        m = sim3_matrix(poses)

        def add(rows, jacobians, residual, weights):
            for r, J in zip(rows, jacobians, strict=True):
                b.append(J.T @ (weights * residual))
                bk.append(r)
                for c, K in zip(rows, jacobians, strict=True):
                    A.append(J.T @ (weights[:, None] * K))
                    ai.append(r)
                    aj.append(c)

        with self.lock:
            gravity = dict(self.gravity)
            height = dict(self.height)
            odometry = list(self.odometry)
            scales = {k: list(v) for k, v in self.scales.items()}
        up = np.array([0.0, -1.0, 0.0])  # MASt3R's world: +y down
        for k, (robot, sigma) in gravity.items():
            if k not in local:
                continue
            _, rotation, _ = split(m[local[k]])
            # Up as the camera sees it, against the robot's: roll and pitch whatever the yaw.
            error = rotation.T @ up - robot.T @ up
            J = np.zeros((3, 7))
            J[:, 3:6] = rotation.T @ skew(up)
            add([local[k]], [J], error, np.full(3, sigma**-2))
        for k, (y, sigma) in height.items():
            if k not in local:
                continue
            _, _, t = split(m[local[k]])
            J = np.zeros((1, 7))
            J[0, 0:3] = (0, 1, 0)
            J[0, 3:6] = -skew(t)[1]
            J[0, 6] = t[1]
            add([local[k]], [J], np.array([t[1] - y]), np.array([sigma**-2]))
        for i, j, rotation, translation, sigma in odometry:
            if i not in local or j not in local:
                continue
            _, ri, ti = split(m[local[i]])
            _, rj, tj = split(m[local[j]])
            r_rot = rotvec(rotation.T @ ri.T @ rj)
            r_t = ri.T @ (tj - ti) - translation
            Ji, Jj = np.zeros((6, 7)), np.zeros((6, 7))
            Ji[0:3, 3:6] = -rj.T
            Jj[0:3, 3:6] = rj.T
            Ji[3:6, 0:3] = -ri.T
            Ji[3:6, 3:6] = ri.T @ skew(tj)
            Ji[3:6, 6] = -ri.T @ ti
            Jj[3:6, 0:3] = ri.T
            Jj[3:6, 3:6] = -ri.T @ skew(tj)
            Jj[3:6, 6] = ri.T @ tj
            add(
                [local[i], local[j]],
                [Ji, Jj],
                np.concatenate((r_rot, r_t)),
                np.asarray(sigma, float) ** -2,
            )
        for k, readings in scales.items():
            if k not in local or not readings:
                continue
            values = np.array(readings)
            centre = float(np.median(values))
            spread = 1.4826 * float(np.median(np.abs(values - centre))) if len(values) > 2 else 0.05
            sigma = max(SCALE_FLOOR, math.hypot(spread, 0.05) / math.sqrt(len(values)))
            s, _, _ = split(m[local[k]])
            J = np.zeros((1, 7))
            J[0, 6] = 1.0
            add([local[k]], [J], np.array([math.log(s) - centre]), np.array([sigma**-2]))
        return A, ai, aj, b, bk

    def _solve(self):
        """All keyframes with a factor, Gauss-Newton as upstream's solve_GN_calib (the oldest
        pinned, points constrained to their rays, ITERATIONS steps), robot factors added."""
        torch = self.torch
        from mast3r_slam.geometry import constrain_points_to_ray

        cfg = self.config["local_opt"]
        frames = self.frontend.frames
        with self.lock:
            involved = set(self.graph.ii.tolist()) | set(self.graph.jj.tolist())
            involved |= {k for k in self.gravity} | {j for _, j, *_ in self.odometry}
            involved |= {i for i, *_ in self.odometry}
        involved = sorted(k for k in involved if k < len(frames))
        if len(involved) < 2:
            return
        local = {k: i for i, k in enumerate(involved)}
        keyframes = [frames[k] for k in involved]
        self._wait()
        before = torch.stack([kf.T_WC.data.reshape(8) for kf in keyframes]).float().contiguous()
        img_size = keyframes[0].img.shape[-2:]
        height, width = img_size
        Xs = constrain_points_to_ray(
            img_size, torch.stack([kf.X_canon for kf in keyframes]).float(), self.frontend.K
        ).contiguous()
        Cs = torch.stack([kf.get_average_conf() for kf in keyframes]).float().contiguous()
        ii, jj, idx, valid, Q = self.graph.prep_two_way_edges()
        lookup = torch.full((len(frames),), -1, dtype=torch.long, device=ii.device)
        lookup[torch.as_tensor(involved, device=ii.device)] = torch.arange(
            len(involved), device=ii.device
        )
        li, lj = lookup[ii].contiguous(), lookup[jj].contiguous()
        idx, valid, Q = idx.contiguous(), valid.contiguous(), Q.float().contiguous()
        K = self.frontend.K.float().contiguous()
        import lietorch

        poses = before.clone()
        n = len(involved)
        for _ in range(ITERATIONS):
            self._wait()
            Hs, gs = self.native.calib_linearize(
                poses,
                Xs,
                Cs,
                K,
                li,
                lj,
                idx,
                valid,
                Q,
                height,
                width,
                int(cfg["pixel_border"]),
                float(cfg["depth_eps"]),
                float(cfg["sigma_pixel"]),
                float(cfg["sigma_depth"]),
                float(cfg["C_conf"]),
                float(cfg["Q_conf"]),
            )
            A, ai, aj, b, bk = self._priors(poses.double().cpu().numpy(), local)
            blocks = [Hs.reshape(-1, 7, 7).double().cpu()]
            rows = [torch.cat((li, li, lj, lj)).cpu()]
            cols = [torch.cat((li, lj, li, lj)).cpu()]
            gradients = [gs.reshape(-1, 7).double().cpu()]
            which = [torch.cat((li, lj)).cpu()]
            if A:
                blocks.append(torch.as_tensor(np.array(A)))
                rows.append(torch.as_tensor(ai))
                cols.append(torch.as_tensor(aj))
                gradients.append(torch.as_tensor(np.array(b)))
                which.append(torch.as_tensor(bk))
            # The oldest keyframe is pinned: index -1 drops its blocks (upstream's num_fix=1).
            dx, ok = self.native.block_solve(
                torch.cat(blocks).float(),
                torch.cat(rows) - 1,
                torch.cat(cols) - 1,
                torch.cat(gradients).float(),
                torch.cat(which) - 1,
                n - 1,
                0.0,
                DAMPING,
            )
            if not int(ok):
                self.failed += 1
                break
            step = -dx.to(device=poses.device, dtype=torch.float32)
            moved = lietorch.Sim3.exp(step) * lietorch.Sim3(poses[1:])
            poses = torch.cat((poses[:1], moved.data.reshape(-1, 8)))
            if float(step.norm()) < float(cfg["delta_norm"]):
                break
        after = poses.double().cpu().numpy()
        start = before.double().cpu().numpy()
        if not np.isfinite(after).all():
            self.failed += 1
            return
        gauge = self.reference if self.reference in local else involved[-1]
        self.solves += 1
        self.results.put(
            dict(
                keyframes=involved,
                before=start,
                after=after,
                gauge=gauge,
                solve=self.solves,
            )
        )


def gauge_correction(before, after):
    """The rigid D (MASt3R axes) with D @ after == before at the gauge keyframe, up to its
    scale: the camera the pilot looks from keeps its place and orientation; the ToF's say on
    its scale still applies."""
    _, rb, tb = split(before)
    _, ra, ta = split(after)
    d = np.eye(4)
    d[:3, :3] = rb @ ra.T
    d[:3, 3] = tb - d[:3, :3] @ ta
    return d


def corrections(result, current):
    """Per keyframe: the world similarity (MASt3R axes) taking its pose now to its solved pose
    under the gauge. current: keyframe -> Sim3 data now (the tracking thread may have moved a
    keyframe since the solve began; the solve's increment is applied on top of that)."""
    keyframes = result["keyframes"]
    before = sim3_matrix(result["before"])
    after = sim3_matrix(result["after"])
    g = keyframes.index(result["gauge"])
    d = gauge_correction(before[g], after[g])
    out = {}
    for i, k in enumerate(keyframes):
        increment = d @ after[i] @ np.linalg.inv(before[i])
        now = sim3_matrix(current[k])[0]
        out[k] = (increment, increment @ now)
    return out
