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

import json
import logging
import math
import os
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
# Retrieval links to a keyframe this many before count as loop closures (the rest overlap
# recent views, as most of a room's keyframes do).
LOOP_GAP = 10
EDGE_CHUNK = 32  # edge directions linearised per kernel launch (widened from compact storage)
STEP = (0.1, math.radians(5), 0.05)  # per keyframe per iteration: metres, radians, log-scale
# A solve that would move a keyframe some solve already placed further than this (metres,
# degrees) diverged; it is dropped. Loop closures in a room move keyframes centimetres.
LIMIT = (0.5, 20.0)
# Every new edge is solved alone first (its two keyframes, one held): where its matches put the
# other keyframe against where the graph has it. A link that disagrees by more than CHECK
# (metres, degrees) while the new keyframe's tracking edge agrees within AGREE is a wrong match
# (a look-alike shelf); ITO_SLAM_EDGE_CHECK=drop (default) leaves it out, =log only records it.
EDGE_CHECK = os.environ.get("ITO_SLAM_EDGE_CHECK", "drop")
CHECK = (0.3, 15.0)
AGREE = (0.1, 5.0)
PAIR_ITERATIONS = 10
# The gauge: ITO_SLAM_GAUGE=settled (default) holds the keyframe the pilot tracks against if a
# solve has placed it before, else the settled keyframe it was tracked from, so a solve moves the
# new keyframe into the room rather than the room about it; =reference holds the pilot's
# keyframe always (before 18 Oct).
GAUGE = os.environ.get("ITO_SLAM_GAUGE", "settled")
# ITO_SLAM_LOG=<file>: one JSON line per solve and per checked edge (a diagnostic).
SOLVE_LOG = os.environ.get("ITO_SLAM_LOG")


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
        self.FactorGraph = FactorGraph
        # Edges, each as upstream's FactorGraph holds them but compact: (i, j, idx i->j int32,
        # idx j->i int32, valid j, valid i, Q i->j fp16, Q j->i fp16); widened per chunk to solve.
        self.edges = []
        self.pairs = set()
        self.loops = []  # (keyframe, earlier keyframe) links retrieval found and matching kept
        self.relocalised = []
        self.lost = set()
        self.tracking = {}  # keyframe -> keyframe it was tracked against when made
        self.gravity = {}  # keyframe -> (up as the camera sees it, MASt3R axes; sigma rad)
        self.height = {}  # keyframe -> (camera y, MASt3R axes; sigma m)
        self.odometry = []  # (i, j, rotation, translation, sigma[6]) robot motion i -> j
        self.turns = []  # (i, j, rotation, sigma) the head gyro's turn i -> j (ITO_HEAD_IMU)
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
        self.rejected = 0
        self.edge_bytes = 0
        self.error = None
        self.steps = []
        self.worst = ((0.0, 0.0), None)
        self.kinds = {}  # (min, max) keyframe pair -> 'track', 'prev', 'retrieval' or 'reloc'
        self.checked = []  # (n, k, kind, metres, degrees, matches, kept) per checked edge
        self.dropped = 0  # links left out by the edge check
        self.unsettled = set()  # keyframes whose own tracking edge disagreed with their pose
        self.unsettled_view = ()  # its copy for statistics() on the tracking thread
        self.gauges = []  # the gauge keyframe of the last solves
        self.log_file = open(SOLVE_LOG, "a", buffering=1) if SOLVE_LOG else None
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

    def add_keyframe(self, index, *, tracked_against, lost, gravity, height, odometry, turn=None):
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
            if turn is not None:
                self.turns.append((turn[0], index, *turn[1:]))
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
            edges=len(self.edges),
            links=len(self.loops),
            loops=sum(n - k >= LOOP_GAP for n, k in self.loops),
            relocalised=len(self.relocalised),
            lost=len(self.lost),
            solves=self.solves,
            solve_ms=round(self.last_solve_seconds * 1000, 1),
            solve_mean_ms=round(self.solve_seconds / max(1, self.solves) * 1000, 1),
            connect_ms=round(self.connect_seconds * 1000, 1),
            loop_ms=[round(s * 1000) for s in self.loop_seconds[-5:]],
            change=[round(v, 4) for v in self.last_change],
            failed=self.failed,
            rejected=self.rejected,
            queued=self.tasks.qsize(),
            edge_mb=round(self.edge_bytes / 2**20),
            error=self.error,
            steps=self.steps,
            worst=[round(self.worst[0][0], 3), round(self.worst[0][1], 1), self.worst[1]],
            dropped=self.dropped,
            unsettled=list(self.unsettled_view),
            gauges=self.gauges[-5:],
            # the largest disagreement any edge checked so far had: metres, degrees
            check_worst=[
                round(max((c[3] for c in self.checked), default=0.0), 3),
                round(max((c[4] for c in self.checked), default=0.0), 1),
            ],
        )

    def _log(self, **row):
        if self.log_file is not None:
            self.log_file.write(json.dumps(row) + "\n")

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
                except torch.cuda.OutOfMemoryError as exc:
                    log.exception("Global graph out of GPU memory; it skips this keyframe's solve")
                    self.failed += 1
                    self.error = f"OOM: {exc}"[:300]
                    torch.cuda.empty_cache()
                    time.sleep(1.0)
                except Exception as exc:
                    import traceback

                    log.exception("Global graph step failed")
                    self.failed += 1
                    self.error = "".join(traceback.format_exception(exc)[-3:])[-600:]
                    errors += 1
                    if errors >= 5:
                        log.error("Global graph stopped after repeated failures")
                        return
                    time.sleep(0.5)

    def _add(self, ii, jj, min_match_frac, is_reloc=False, kind="retrieval"):
        """Upstream add_factors for pairs not linked yet; True if any edge was kept (with
        is_reloc, all or nothing). kind: what made the link, one per pair or for all."""
        kinds = kind if isinstance(kind, list) else [kind] * len(ii)
        kinds = {(min(i, j), max(i, j)): c for i, j, c in zip(ii, jj, kinds, strict=True)}
        pairs = [
            (i, j) for i, j in zip(ii, jj, strict=True) if (min(i, j), max(i, j)) not in self.pairs
        ]
        if not pairs:
            return False
        self._wait()
        graph = self.FactorGraph(
            self.frontend.model, self.frontend.frames, self.frontend.K, device="cuda"
        )
        added = graph.add_factors(
            [i for i, _ in pairs], [j for _, j in pairs], min_match_frac, is_reloc=is_reloc
        )
        if not added:
            return False
        for e, (i, j) in enumerate(zip(graph.ii.tolist(), graph.jj.tolist(), strict=True)):
            edge = (
                i,
                j,
                graph.idx_ii2jj[e].int(),
                graph.idx_jj2ii[e].int(),
                graph.valid_match_j[e],
                graph.valid_match_i[e],
                graph.Q_ii2jj[e].half(),
                graph.Q_jj2ii[e].half(),
            )
            self.edges.append(edge)
            self.edge_bytes += sum(t.numel() * t.element_size() for t in edge[2:])
            self.pairs.add((min(i, j), max(i, j)))
            self.kinds[(min(i, j), max(i, j))] = kinds.get((min(i, j), max(i, j)), "retrieval")
        return True

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
        first = len(self.edges)
        if lost:
            if candidates and self._add(
                [n] * len(candidates),
                candidates,
                cfg["reloc"]["min_match_frac"],
                is_reloc=cfg["reloc"]["strict"],
                kind="reloc",
            ):
                self.relocalised.append((n, candidates))
                log.info("Relocalised keyframe %d against %s", n, candidates)
                self._check(n, first)
                return len(candidates)
            return 0
        if tracked is not None and tracked != n:
            # Upstream's consecutive edge: the keyframe tracking linked it to, always kept.
            self._add([tracked], [n], TRACKING_FRAC, kind="track")
        others = [k for k in candidates if k != tracked]
        if n > 0 and n - 1 != tracked and n - 1 not in others:
            others.insert(0, n - 1)
        kept = 0
        if others:
            # (n, k), never (k, n): upstream keeps an edge (n - 1, n) unchecked as consecutive;
            # here the keyframe before need not overlap (tracking chose another), so every
            # edge but the tracking one passes upstream's two-way match-fraction check.
            self._add(
                [n] * len(others),
                others,
                cfg["local_opt"]["min_match_frac"],
                kind=["prev" if k == n - 1 else "retrieval" for k in others],
            )
        before = {(min(e[0], e[1]), max(e[0], e[1])) for e in self.edges[:first]}
        self._check(n, first)
        kept_pairs = {(min(e[0], e[1]), max(e[0], e[1])) for e in self.edges[first:]}
        for k in others:
            pair = (min(k, n), max(k, n))
            if pair in kept_pairs and pair not in before and k != n - 1:
                self.loops.append((n, k))
                kept += n - k >= LOOP_GAP
        return kept

    def _check(self, n, first):
        """The edge check on keyframe n's new edges (self.edges[first:]): each solved alone
        (n moved, the other held). Links that put n far from where its tracking edge and the
        graph agree it is are dropped (ITO_SLAM_EDGE_CHECK=drop); if n's own tracking edge
        disagrees with its pose, n's pose is off, not the links: it is marked unsettled and
        never holds the gauge until a solve has placed it."""
        if EDGE_CHECK == "off" or len(self.edges) <= first:
            return
        results = []
        for e in range(first, len(self.edges)):
            edge = self.edges[e]
            other = edge[1] if edge[0] == n else edge[0]
            kind = self.kinds.get((min(edge[0], edge[1]), max(edge[0], edge[1])), "retrieval")
            try:
                metres, degrees, matches = self._pair(edge, n)
            except Exception:
                log.exception("Edge check failed")
                continue
            results.append((e, other, kind, metres, degrees, matches))
        # Without a tracking edge (a lost keyframe, relocalised) its pose is the robot's guess:
        # nothing says the links are wrong rather than it.
        tracking = [r for r in results if r[2] == "track"]
        trusted = bool(tracking) and tracking[0][3] <= AGREE[0] and tracking[0][4] <= AGREE[1]
        if tracking and not trusted or not tracking and n in self.lost:
            self.unsettled.add(n)
            self.unsettled_view = tuple(sorted(self.unsettled))
        drop = []
        for e, other, kind, metres, degrees, matches in results:
            off = metres > CHECK[0] or degrees > CHECK[1]
            keep = not (EDGE_CHECK == "drop" and trusted and off and kind != "track")
            self.checked.append((n, other, kind, metres, degrees, matches, keep))
            self._log(
                event="edge",
                keyframe=n,
                other=other,
                kind=kind,
                metres=round(metres, 4),
                degrees=round(degrees, 2),
                matches=matches,
                trusted=trusted,
                kept=keep,
            )
            if not keep:
                drop.append(e)
        for e in sorted(drop, reverse=True):
            edge = self.edges.pop(e)
            self.edge_bytes -= sum(t.numel() * t.element_size() for t in edge[2:])
            self.dropped += 1
            log.info("Edge check: link %d-%d left out", edge[0], edge[1])

    def _pair(self, edge, moving):
        """Gauss-Newton on one edge alone, the keyframe `moving` free and the other held, from
        where the graph has them now: how far (metres of its centre, degrees) the edge's
        matches alone would move it, and how many matches count."""
        torch = self.torch
        import lietorch
        from mast3r_slam.geometry import constrain_points_to_ray

        cfg = self.config["local_opt"]
        frames = self.frontend.frames
        i, j = edge[0], edge[1]
        # Local 0 is held: the keyframe that is not moving.
        order = (j, i) if moving == i else (i, j)
        held = [(frames[k].T_WC.data, frames[k].X_canon, frames[k].C, frames[k].N) for k in order]
        self._wait()
        poses = torch.stack([h[0].reshape(8) for h in held]).float().contiguous()
        img_size = frames[i].img.shape[-2:]
        height, width = img_size
        Xs = constrain_points_to_ray(
            img_size, torch.stack([h[1] for h in held]).float(), self.frontend.K
        ).contiguous()
        Cs = torch.stack([h[2] / h[3] for h in held]).float().contiguous()
        torch.cuda.current_stream().synchronize()
        del held
        Xs = torch.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        Cs = torch.where(torch.isfinite(Cs), Cs, 0)
        a, b = (0, 1) if order[0] == i else (1, 0)  # local index of edge[0], edge[1]
        li = torch.tensor([a, b], device="cuda")
        lj = torch.tensor([b, a], device="cuda")
        idx = torch.stack((edge[2], edge[3])).long().contiguous()
        valid = torch.stack((edge[4], edge[5])).contiguous()
        Q = torch.stack((edge[6], edge[7])).float().contiguous()
        counted = valid.reshape(2, -1) & (Q.reshape(2, -1) > float(cfg["Q_conf"]))
        matches = int(counted.sum())
        K = self.frontend.K.float().contiguous()
        start = poses.clone()
        for _ in range(PAIR_ITERATIONS):
            self._wait()
            Hs, gs = self.native.calib_linearize(
                poses, Xs, Cs, K, li, lj, idx, valid, Q, height, width,
                int(cfg["pixel_border"]), float(cfg["depth_eps"]), float(cfg["sigma_pixel"]),
                float(cfg["sigma_depth"]), float(cfg["C_conf"]), float(cfg["Q_conf"]),
            )  # fmt: skip
            arguments = (
                Hs.reshape(-1, 7, 7).cpu(),
                torch.cat((li, li, lj, lj)).cpu() - 1,
                torch.cat((li, lj, li, lj)).cpu() - 1,
                gs.reshape(-1, 7).cpu(),
                torch.cat((li, lj)).cpu() - 1,
                1,
            )
            for lm in (0.0, 1e-4, 1e-2):
                dx, ok = self.native.block_solve(*arguments, lm, DAMPING)
                if int(ok):
                    break
            if not int(ok):
                break
            step = -dx.to(device=poses.device, dtype=torch.float32)
            size = max(
                float(step[:, 0:3].norm()) / STEP[0],
                float(step[:, 3:6].norm()) / STEP[1],
                float(step[:, 6].abs().max()) / STEP[2],
                1.0,
            )
            step = step / size
            moved = lietorch.Sim3.exp(step) * lietorch.Sim3(poses[1:])
            poses = torch.cat((poses[:1], moved.data.reshape(-1, 8)))
            if float(step.norm()) < 1e-6:
                break
        before = sim3_matrix(start[1].double().cpu().numpy())[0]
        after = sim3_matrix(poses[1].double().cpu().numpy())[0]
        if not np.isfinite(after).all():
            return math.inf, 180.0, matches
        _, rb, tb = split(before)
        _, ra, ta = split(after)
        angle = math.degrees(math.acos(np.clip((np.trace(rb.T @ ra) - 1) / 2, -1, 1)))
        return float(np.linalg.norm(ta - tb)), angle, matches

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
            turns = list(self.turns)
            scales = {k: list(v) for k, v in self.scales.items()}
        up = np.array([0.0, -1.0, 0.0])  # MASt3R's world: +y down
        for k, (seen, sigma) in gravity.items():
            if k not in local:
                continue
            _, rotation, _ = split(m[local[k]])
            # Up as the camera sees it, against the robot's: roll and pitch whatever the yaw.
            error = rotation.T @ up - seen
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
        for i, j, rotation, sigma in turns:
            if i not in local or j not in local:
                continue
            _, ri, _ = split(m[local[i]])
            _, rj, _ = split(m[local[j]])
            Ji, Jj = np.zeros((3, 7)), np.zeros((3, 7))
            Ji[:, 3:6] = -rj.T
            Jj[:, 3:6] = rj.T
            add(
                [local[i], local[j]],
                [Ji, Jj],
                rotvec(rotation.T @ ri.T @ rj),
                np.full(3, sigma**-2),
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
        import lietorch
        from mast3r_slam.geometry import constrain_points_to_ray

        cfg = self.config["local_opt"]
        frames = self.frontend.frames
        edges = list(self.edges)
        with self.lock:
            # Only keyframes with a visual edge: one whose edges are not made yet (queued while
            # the worker connected another) was posed by the robot's factors alone and pulled
            # back by its edges one solve later, turning the room about it.
            involved = {e[0] for e in edges} | {e[1] for e in edges}
        involved = sorted(k for k in involved if k < len(frames))
        if len(involved) < 2:
            return
        local = {k: i for i, k in enumerate(involved)}
        keyframes = [frames[k] for k in involved]
        # The tracking thread replaces these tensors as it goes; hold them until the copies
        # are done, or the allocator may hand their memory to its stream meanwhile.
        held = [(kf.T_WC.data, kf.X_canon, kf.C, kf.N) for kf in keyframes]
        self._wait()
        before = torch.stack([h[0].reshape(8) for h in held]).float().contiguous()
        img_size = keyframes[0].img.shape[-2:]
        height, width = img_size
        Xs = constrain_points_to_ray(
            img_size, torch.stack([h[1] for h in held]).float(), self.frontend.K
        ).contiguous()
        Cs = torch.stack([h[2] / h[3] for h in held]).float().contiguous()
        finite = (torch.isfinite(Xs).all(-1) & torch.isfinite(Cs[..., 0])).all(-1)
        finite &= torch.isfinite(before).all(-1)
        torch.cuda.current_stream().synchronize()
        del held
        if not bool(finite.all()):
            # Guarded at the source (mast3r_frontend._finite_pointmaps); never solve through it.
            bad = [involved[i] for i in torch.nonzero(~finite)[:, 0].tolist()]
            self.error = f"non-finite keyframes {bad}"
            Xs = torch.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
            Cs = torch.where(torch.isfinite(Cs), Cs, 0)
        K = self.frontend.K.float().contiguous()
        directions = [(local[e[0]], local[e[1]], e[2], e[4], e[6]) for e in edges] + [
            (local[e[1]], local[e[0]], e[3], e[5], e[7]) for e in edges
        ]
        poses = before.clone()
        n = len(involved)
        steps = self.steps = []
        for _ in range(ITERATIONS):
            blocks, rows, cols, gradients, which = [], [], [], [], []
            for start in range(0, len(directions), EDGE_CHUNK):
                self._wait()
                chunk = directions[start : start + EDGE_CHUNK]
                li = torch.as_tensor([c[0] for c in chunk], device="cuda")
                lj = torch.as_tensor([c[1] for c in chunk], device="cuda")
                Hs, gs = self.native.calib_linearize(
                    poses,
                    Xs,
                    Cs,
                    K,
                    li,
                    lj,
                    torch.stack([c[2] for c in chunk]).long().contiguous(),
                    torch.stack([c[3] for c in chunk]).contiguous(),
                    torch.stack([c[4] for c in chunk]).float().contiguous(),
                    height,
                    width,
                    int(cfg["pixel_border"]),
                    float(cfg["depth_eps"]),
                    float(cfg["sigma_pixel"]),
                    float(cfg["sigma_depth"]),
                    float(cfg["C_conf"]),
                    float(cfg["Q_conf"]),
                )
                blocks.append(Hs.reshape(-1, 7, 7).cpu())
                rows.append(torch.cat((li, li, lj, lj)).cpu())
                cols.append(torch.cat((li, lj, li, lj)).cpu())
                gradients.append(gs.reshape(-1, 7).cpu())
                which.append(torch.cat((li, lj)).cpu())
            A, ai, aj, b, bk = self._priors(poses.double().cpu().numpy(), local)
            if A:
                blocks.append(torch.as_tensor(np.array(A), dtype=torch.float32))
                rows.append(torch.as_tensor(ai))
                cols.append(torch.as_tensor(aj))
                gradients.append(torch.as_tensor(np.array(b), dtype=torch.float32))
                which.append(torch.as_tensor(bk))
            # The oldest keyframe is pinned: index -1 drops its blocks (upstream's num_fix=1).
            arguments = (
                torch.cat(blocks),
                torch.cat(rows) - 1,
                torch.cat(cols) - 1,
                torch.cat(gradients),
                torch.cat(which) - 1,
                n - 1,
            )
            # Upstream's plain Gauss-Newton step; where the normal equations do not factor
            # (a keyframe far from where its edges put it) Levenberg-Marquardt damping.
            for lm in (0.0, 1e-4, 1e-2):
                dx, ok = self.native.block_solve(*arguments, lm, DAMPING)
                if int(ok):
                    break
            if not int(ok):
                self.failed += 1
                self.error = "normal equations not positive definite"
                if not steps:
                    return
                break  # keep the last iterate that factored
            step = -dx.to(device=poses.device, dtype=torch.float32)
            # A trust region per keyframe: Gauss-Newton has no line search, and a keyframe
            # made mid-turn can start far off; large steps are shortened, not taken whole.
            size = torch.maximum(
                step[:, 0:3].norm(dim=1) / STEP[0],
                torch.maximum(step[:, 3:6].norm(dim=1) / STEP[1], step[:, 6].abs() / STEP[2]),
            )
            step = step / size.clamp_min(1.0)[:, None]
            steps.append(round(float(step.norm()), 4))
            moved = lietorch.Sim3.exp(step) * lietorch.Sim3(poses[1:])
            poses = torch.cat((poses[:1], moved.data.reshape(-1, 8)))
            if float(step.norm()) < float(cfg["delta_norm"]):
                break
        after = poses.double().cpu().numpy()
        start = before.double().cpu().numpy()
        if not np.isfinite(after).all():
            self.failed += 1
            self.error = "non-finite poses"
            return
        gauge = self.reference if self.reference in local else involved[-1]
        if GAUGE == "settled" and (gauge not in self.owned or gauge in self.unsettled):
            parent = self.tracking.get(gauge)
            if parent in local and parent in self.owned and parent not in self.unsettled:
                gauge = parent
        self.gauges.append(gauge)
        del self.gauges[:-20]
        moves = corrections(
            dict(keyframes=involved, before=start, after=after, gauge=gauge),
            dict(zip(involved, start, strict=True)),
        )
        self.worst = max(
            (
                (_size(increment, sim3_matrix(start[local[k]])[0]), k)
                for k, (increment, _) in moves.items()
            ),
            default=((0.0, 0.0), None),
        )
        worst = max(
            (
                _size(increment, sim3_matrix(start[local[k]])[0])
                for k, (increment, _) in moves.items()
                if k in self.owned
            ),
            default=(0.0, 0.0),
        )
        rejected = worst[0] > LIMIT[0] or worst[1] > LIMIT[1]
        if self.log_file is not None:
            self._log(
                event="solve",
                solve=self.solves + 1,
                keyframes=len(involved),
                edges=len(edges),
                reference=self.reference,
                gauge=gauge,
                steps=steps,
                rejected=rejected,
                moves={
                    str(k): [round(v, 4) for v in _size(inc, sim3_matrix(start[local[k]])[0])]
                    for k, (inc, _) in moves.items()
                },
                worst_edges=[
                    [e[0], e[1], self.kinds.get((min(e[0], e[1]), max(e[0], e[1])))]
                    for e in edges
                    if self.worst[1] in (e[0], e[1])
                ],
            )
        if rejected:
            # Gauss-Newton has no line search: a step this large moved settled keyframes
            # through the room; keep them, the next keyframe's solve starts afresh.
            self.rejected += 1
            self.error = (
                f"rejected: a settled keyframe would move {worst[0]:.2f} m, {worst[1]:.0f} deg"
            )
            return
        self.unsettled -= set(involved)
        self.unsettled_view = tuple(sorted(self.unsettled))
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


def _size(increment, pose):
    """How far a world similarity carries a camera: metres of its centre, degrees of turn."""
    s, rotation, _ = split(increment)
    centre = pose[:3, 3]
    shift = float(np.linalg.norm(increment[:3, :3] @ centre + increment[:3, 3] - centre))
    angle = math.degrees(math.acos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
    return shift, angle


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
