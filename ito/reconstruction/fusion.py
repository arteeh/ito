"""Sliding-window pose graph: the robot's camera motion corrected by vision.

The world is the robot's odometry frame at startup; nothing restarts. Factors: robot relative
motion, gravity, a prior on the oldest node, visual relative poses (Cauchy loss), and where
present head-IMU turns and gravity and map alignments. The frame delay td (a frame stamped t
was exposed at t - td) and the front end's log-scale (metres per unit, from floor and ToF
observations) are estimated apart from the poses."""

import math
import os
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

WINDOW = 30
ANCHOR_ROTATION = math.radians(2)
ANCHOR_POSITION = 0.05
CAUCHY = 2.0
SCALE_SIGMA = 0.2
VISUAL_SCALE = float(os.environ.get("ITO_VISUAL_SCALE", "0.3"))
# ITO_ROBOT_SIGMA=F (research): robot relative-motion, gravity and height sigmas times F.
ROBOT_SIGMA = float(os.environ.get("ITO_ROBOT_SIGMA", "1"))
SCALE_MEMORY = 20
DELAY_MEMORY = 60
DELAY_TURN = math.radians(20)
DELAY_REACH = 0.05
DELAY_STEP = 0.001
UP = np.array([0.0, 1.0, 0.0])
MAP_FLOOR = np.repeat((math.radians(0.05), 0.005), 3)
PUBLISHED = 600
# The robot's kinematic camera height over the floor holds absolutely, not only from one sample
# to the next: with relative motion alone, a degree of pitch error turned a 1.5 m walk into a
# 3 cm drop of the camera (in the sim), and everything mapped after it sat 3 cm low. Robots
# whose world's Y is height over the floor (the wire contract) can say so this well.
HEIGHT_SIGMA = 0.01


@dataclass
class Node:
    time: float  # Frame stamp on the pilot clock; exposed at time - td.
    pose: np.ndarray  # World-from-camera, 4x4.


@dataclass
class Visual:
    reference: int
    frame: int
    rotation: np.ndarray  # 3x3, reference-from-frame
    translation: np.ndarray  # front-end units, in the reference camera frame
    sigma: np.ndarray  # rotation (rad) and translation (m) sigmas, 6


def _cauchy(whitened):
    squared = np.sum(whitened * whitened, axis=1, keepdims=True)
    cost = CAUCHY**2 * np.log1p(squared / CAUCHY**2)
    factor = np.sqrt(np.divide(cost, squared, out=np.ones_like(squared), where=squared > 1e-12))
    return whitened * factor


class PoseGraph:
    def __init__(self, motion, td=0.0, td_bounds=(0.0, 0.3), estimate_td=False, inertial=None):
        self.motion = motion
        self.inertial = inertial
        self.td = float(td)
        self.td_bounds = td_bounds
        self.estimate_td = estimate_td
        self.delays = deque(maxlen=DELAY_MEMORY)
        self.log_scale = 0.0
        self.scales = deque(maxlen=SCALE_MEMORY)
        self.scale_known = False
        self.nodes: dict[int, Node] = {}
        self.keyframes: list[int] = []
        self.current: int | None = None
        self.visual: list[Visual] = []
        self.mapped: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.anchor: np.ndarray | None = None
        self.next_id = 0
        self.solves = 0
        self.published = deque(maxlen=PUBLISHED)

    def latest(self):
        if self.current is not None:
            return self.current
        return self.keyframes[-1] if self.keyframes else None

    def latest_keyframe(self):
        return self.keyframes[-1] if self.keyframes else None

    def fold_scale(self):
        factor = math.exp(self.log_scale)
        for v in self.visual:
            v.translation = v.translation * factor
        self.scales = deque(
            ((value - self.log_scale, sigma) for value, sigma in self.scales), SCALE_MEMORY
        )
        self.log_scale = 0.0
        return factor

    def add_scale(self, log_scale, sigma):
        self.scales.append((float(log_scale), float(sigma)))

    def predict(self, t):
        last = self.latest()
        if last is None:
            return self.motion.at(t - self.td)
        node = self.nodes[last]
        relative, _ = self.motion.between(node.time - self.td, t - self.td)
        return node.pose @ relative

    def add_frame(self, t, pose=None):
        if self.current is not None:
            self._drop(self.current)
        node = self.next_id
        self.next_id += 1
        self.nodes[node] = Node(t, self.predict(t) if pose is None else np.array(pose, float))
        self.current = node
        if self.anchor is None:
            self.anchor = self.nodes[node].pose.copy()
        return node

    def add_visual(self, reference, frame, relative, sigma):
        if reference not in self.nodes or frame not in self.nodes or reference == frame:
            return
        translation = np.array(relative[:3, 3], float)
        sigma = np.array(sigma, float)
        sigma[3:] = np.hypot(sigma[3:], VISUAL_SCALE * np.linalg.norm(translation))
        self.visual.append(
            Visual(reference, frame, np.array(relative[:3, :3], float), translation, sigma)
        )
        if self.estimate_td:
            self._observe_delay(self.visual[-1])

    def add_map(self, node, pose, sigma):
        sigma = np.asarray(sigma, float)
        if node not in self.nodes or sigma.shape != (6,) or not np.isfinite(sigma).all():
            return False
        if not np.isfinite(pose).all():
            return False
        self.mapped[node] = (np.array(pose, float), np.maximum(sigma, MAP_FLOOR))
        return True

    def _observe_delay(self, visual):
        a, b = self.nodes[visual.reference].time, self.nodes[visual.frame].time
        low, high = self.td_bounds
        delays = np.arange(
            max(low, self.td - DELAY_REACH), min(high, self.td + DELAY_REACH) + 1e-9, DELAY_STEP
        )
        if len(delays) < 3:
            return
        rotations, _ = self.motion.poses(np.concatenate((a - delays, b - delays)))
        robot = rotations[: len(delays)].inv() * rotations[len(delays) :]
        error = (Rotation.from_matrix(visual.rotation).inv() * robot).magnitude()
        best = int(np.argmin(error))
        if best in (0, len(delays) - 1):
            return
        turn = (robot[best - 1].inv() * robot[best + 1]).magnitude() / (2 * DELAY_STEP)
        if turn < DELAY_TURN:
            return
        left, middle, right = error[best - 1 : best + 2]
        curve = left - 2 * middle + right
        shift = 0.5 * (left - right) / curve if curve > 1e-12 else 0.0
        delay = float(delays[best] + np.clip(shift, -1, 1) * DELAY_STEP)
        self.delays.append((delay, float(visual.sigma[0]) / turn))

    def solve_delay(self):
        if not self.estimate_td or not self.delays:
            return
        values, sigmas = np.array(self.delays).T
        delay = float(np.median(values))
        for _ in range(8):
            weights = 1 / (sigmas**2 + ((delay - values) / CAUCHY) ** 2)
            delay = float(weights @ values / weights.sum())
        self.td = float(np.clip(delay, *self.td_bounds))

    def promote(self):
        if self.current is None:
            return []
        self.keyframes.append(self.current)
        self.current = None
        dropped = []
        while len(self.keyframes) > WINDOW:
            oldest = self.keyframes.pop(0)
            self._drop(oldest)
            dropped.append(oldest)
            self.anchor = self.nodes[self.keyframes[0]].pose.copy()
        return dropped

    def expire(self, oldest):
        dropped = []
        while self.keyframes and self.nodes[self.keyframes[0]].time - self.td < oldest:
            node = self.keyframes.pop(0)
            self._drop(node)
            dropped.append(node)
        if dropped and self.keyframes:
            self.anchor = self.nodes[self.keyframes[0]].pose.copy()
        return dropped

    def _drop(self, node):
        del self.nodes[node]
        self.mapped.pop(node, None)
        self.visual = [v for v in self.visual if node not in (v.reference, v.frame)]
        if self.current == node:
            self.current = None

    def pose(self, node):
        return self.nodes[node].pose

    def correction(self, node):
        n = self.nodes[node]
        return n.pose @ np.linalg.inv(self.motion.at(n.time - self.td))

    def publish(self, node):
        n = self.nodes[node]
        self.published.append((n.time, node, n.pose.copy()))

    def pose_at(self, t):
        if not self.published:
            return self.predict(t)
        times = [entry[0] for entry in self.published]
        after = int(np.searchsorted(times, t))
        if after == 0 or after == len(times):
            correction = self._published_correction(self.published[min(after, len(times) - 1)])
        else:
            (t0, *_), (t1, *_) = self.published[after - 1], self.published[after]
            c0 = self._published_correction(self.published[after - 1])
            c1 = self._published_correction(self.published[after])
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            ends = Rotation.from_matrix(np.array([c0[:3, :3], c1[:3, :3]]))
            turn = (ends[0].inv() * ends[1]).as_rotvec()
            correction = np.eye(4)
            correction[:3, :3] = (ends[0] * Rotation.from_rotvec(turn * f)).as_matrix()
            correction[:3, 3] = c0[:3, 3] * (1 - f) + c1[:3, 3] * f
        return correction @ self.motion.at(t - self.td)

    def _published_correction(self, entry):
        time, _, pose = entry
        return pose @ np.linalg.inv(self.motion.at(time - self.td))

    def _refresh_published(self):
        for i, (time, node, _) in enumerate(self.published):
            if node in self.nodes:
                self.published[i] = (time, node, self.nodes[node].pose.copy())

    def solve_frame(self):
        if self.current is None:
            return None
        self._solve([self.current], frame_only=True)
        return self.nodes[self.current].pose

    def solve_window(self):
        order = self.keyframes + ([self.current] if self.current is not None else [])
        if order:
            self._solve(order, frame_only=False)
            self._refresh_published()
        return {k: self.nodes[k].pose for k in self.keyframes}

    def solve_scale(self):
        if not self.scales:
            return
        values, sigmas = np.array(self.scales).T
        prior = 1 / SCALE_SIGMA**2 if self.scale_known else 0.0
        scale = float(np.median(values)) if not self.scale_known else self.log_scale
        for _ in range(8):
            weights = 1 / (sigmas**2 + ((scale - values) / CAUCHY) ** 2)
            scale = (weights @ values + prior * self.log_scale) / (weights.sum() + prior)
        self.log_scale = float(scale)
        self.scale_known = True

    def _solve(self, free, frame_only):
        window = self.keyframes + ([self.current] if self.current is not None else [])
        visual = [v for v in self.visual if v.reference in self.nodes and v.frame in self.nodes]
        if frame_only:
            visual = [v for v in visual if self.current in (v.reference, v.frame)]
            involved = {self.current, *(self.keyframes[-1:])}
            involved.update(n for v in visual for n in (v.reference, v.frame))
            order = sorted(involved, key=lambda n: self.nodes[n].time)
            chain_nodes = [(self.keyframes[-1], self.current)] if self.keyframes else []
        else:
            order = window
            chain_nodes = list(zip(window, window[1:], strict=False))
        index = {node: i for i, node in enumerate(order)}
        free_index = {node: i for i, node in enumerate(free)}
        chain = [(index[a], index[b]) for a, b in chain_nodes]
        anchored = not self.keyframes or window[0] in free_index
        start = {node: self.nodes[node].pose.copy() for node in order}
        start_r = Rotation.from_matrix(np.array([start[n][:3, :3] for n in order]))
        start_p = np.array([start[n][:3, 3] for n in order])
        times = np.array([self.nodes[n].time for n in order])
        estimate_scale = not frame_only and bool(self.visual)
        size = 6 * len(free) + int(estimate_scale)
        td, scale0 = self.td, self.log_scale

        rows = np.array([index[n] for n in free], dtype=int)
        start_q = start_r.as_quat()
        ci = np.array([i for i, _ in chain], dtype=int)
        cj = np.array([j for _, j in chain], dtype=int)
        robot_sigma = np.array(
            [
                np.sqrt(np.diag(self.motion.between(times[i] - td, times[j] - td)[1]))
                for i, j in chain
            ]
        ).reshape(-1, 6)
        gravity_sigma = np.array([self.motion.gravity_sigma(times[i] - td) for i in rows])
        height_sigma = np.array(
            [self.motion.height_sigma(times[i] - td, HEIGHT_SIGMA) for i in rows]
        )
        turns, ups = self._inertial(times - td, chain, rows)
        if turns:
            ti, tj = (np.array([t[k] for t in turns], dtype=int) for k in (0, 1))
            turn_inv = Rotation.concatenate([t[2] for t in turns])
            turn_sigma = np.array([t[3] for t in turns])[:, None]
        if ups:
            ui = np.array([u[0] for u in ups], dtype=int)
            up_seen = np.array([u[1] for u in ups])
            up_sigma = np.array([u[2] for u in ups])[:, None]
        robot_r, robot_p = self.motion.poses(times - td)
        vi = np.array([index[v.reference] for v in visual], dtype=int)
        vj = np.array([index[v.frame] for v in visual], dtype=int)
        if visual:
            v_rotation_inv = Rotation.from_matrix(np.array([v.rotation for v in visual])).inv()
            v_translation = np.array([v.translation for v in visual])
            v_sigma = np.array([v.sigma for v in visual])
        anchor_r = Rotation.from_matrix(self.anchor[:3, :3]).inv()
        mapped = [n for n in free if n in self.mapped]
        mi = np.array([index[n] for n in mapped], dtype=int)
        if mapped:
            m_rotation_inv = Rotation.from_matrix(
                np.array([self.mapped[n][0][:3, :3] for n in mapped])
            ).inv()
            m_position = np.array([self.mapped[n][0][:3, 3] for n in mapped])
            m_sigma = np.array([self.mapped[n][1] for n in mapped])

        def unpack(x):
            positions = start_p.copy()
            quats = start_q.copy()
            if len(rows):
                delta = x[: 6 * len(free)].reshape(-1, 6)
                quats[rows] = (start_r[rows] * Rotation.from_rotvec(delta[:, :3])).as_quat()
                positions[rows] += delta[:, 3:]
            scale = x[6 * len(free)] if estimate_scale else scale0
            return Rotation.from_quat(quats), positions, scale

        def residuals(x):
            rotations, positions, scale = unpack(x)
            blocks = []
            if len(ci):
                ri_inv = rotations[ci].inv()
                measured_inv = robot_r[ci].inv()
                relative_r = measured_inv * robot_r[cj]
                relative_p = measured_inv.apply(robot_p[cj] - robot_p[ci])
                rotation = (relative_r.inv() * ri_inv * rotations[cj]).as_rotvec()
                moved = ri_inv.apply(positions[cj] - positions[ci]) - relative_p
                blocks.append((np.hstack((rotation, moved)) / (ROBOT_SIGMA * robot_sigma)).ravel())
            if len(rows):
                error = (rotations[rows] * robot_r[rows].inv()).as_rotvec()
                blocks.append((error[:, [0, 2]] / (ROBOT_SIGMA * gravity_sigma[:, None])).ravel())
                blocks.append(
                    (positions[rows, 1] - robot_p[rows, 1]) / (ROBOT_SIGMA * height_sigma)
                )
            if turns:
                error = (turn_inv * rotations[ti].inv() * rotations[tj]).as_rotvec()
                blocks.append(_cauchy(error / turn_sigma).ravel())
            if ups:
                error = np.cross(rotations[ui].inv().apply(UP), up_seen)
                blocks.append(_cauchy(error / up_sigma).ravel())
            if anchored:
                error = (anchor_r * rotations[0]).as_rotvec()
                blocks.append(error / ANCHOR_ROTATION)
                blocks.append((positions[0] - self.anchor[:3, 3]) / ANCHOR_POSITION)
            if visual:
                ri_inv = rotations[vi].inv()
                rotation = (v_rotation_inv * ri_inv * rotations[vj]).as_rotvec()
                moved = (
                    ri_inv.apply(positions[vj] - positions[vi]) - math.exp(scale) * v_translation
                )
                blocks.append(_cauchy(np.hstack((rotation, moved)) / v_sigma).ravel())
            if mapped:
                rotation = (m_rotation_inv * rotations[mi]).as_rotvec()
                moved = positions[mi] - m_position
                blocks.append(_cauchy(np.hstack((rotation, moved)) / m_sigma).ravel())
            if estimate_scale:
                blocks.append(np.array([(scale - scale0) / SCALE_SIGMA]))
            return np.concatenate(blocks) if blocks else np.zeros(1)

        x0 = np.zeros(size)
        if estimate_scale:
            x0[6 * len(free)] = scale0
        sparsity = (
            None
            if frame_only
            else self._sparsity(
                order,
                free_index,
                chain_nodes,
                visual,
                mapped,
                anchored,
                estimate_scale,
                [(order[i], order[j]) for i, j, _, _ in turns],
                [order[i] for i, _, _ in ups],
            )
        )
        result = least_squares(
            residuals,
            x0,
            jac_sparsity=sparsity,
            method="trf",
            x_scale="jac",
            max_nfev=20 if frame_only else 40,
        )
        if not np.isfinite(result.x).all():
            return
        rotations, positions, scale = unpack(result.x)
        matrices = rotations.as_matrix()
        for node, i in zip(free, rows, strict=True):
            pose = np.eye(4)
            pose[:3, :3] = matrices[i]
            pose[:3, 3] = positions[i]
            self.nodes[node].pose = pose
        self.solves += 1

    def _inertial(self, exposures, chain, rows):
        if self.inertial is None:
            return [], []
        turns = []
        for i, j in chain:
            found = self.inertial.turn(exposures[i], exposures[j])
            if found is not None:
                turns.append((i, j, found[0].inv(), found[1]))
        ups = []
        for i in rows:
            found = self.inertial.up(exposures[i])
            if found is not None:
                ups.append((i, *found))
        return turns, ups

    def _sparsity(
        self, order, free_index, chain, visual, mapped, anchored, scale, turns=(), ups=()
    ):
        rows = []
        size = 6 * len(free_index) + int(scale)

        def block(count, nodes, uses_scale=False):
            columns = set()
            for node in nodes:
                if node in free_index:
                    columns.update(range(6 * free_index[node], 6 * free_index[node] + 6))
            if uses_scale and scale:
                columns.add(6 * len(free_index))
            rows.extend([columns] * count)

        for a, b in chain:
            block(6, (a, b))
        for node in free_index:
            block(2, (node,))
        for node in free_index:
            block(1, (node,))
        for a, b in turns:
            block(3, (a, b))
        for node in ups:
            block(3, (node,))
        if anchored:
            block(6, (order[0],))
        for v in visual:
            block(6, (v.reference, v.frame), uses_scale=True)
        for node in mapped:
            block(6, (node,))
        if scale:
            block(1, (), uses_scale=True)
        matrix = lil_matrix((max(len(rows), 1), size), dtype=int)
        for r, columns in enumerate(rows):
            for c in columns:
                matrix[r, c] = 1
        return matrix
