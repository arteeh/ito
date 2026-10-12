"""Fuse the robot's own camera motion with what vision sees, in a small sliding window.

The robot measures where its camera points better than monocular vision does (encoders, IMU,
50 Hz); vision corrects what the robot measures badly (yaw drift over minutes, slipping feet).
The world is the robot's odometry frame where it started, so there is no map to restart: a
frame vision cannot use simply adds no visual factor, and its pose comes from the robot.

Variables: one pose per node (the last WINDOW keyframes and the current frame), the frame
delay td (a frame stamped t was exposed at t - td) and one log-scale between the visual
front end's lengths and metres. Factors: the robot's relative motion between consecutive
nodes, gravity (pitch and roll) per node, a weak prior holding the oldest node where the
window last put it, visual relative poses and direct observations of the scale (the floor
under a frame, slam.py), both with a Cauchy loss. Solved with scipy's least squares on a
sparse Jacobian: per frame the current node alone, per keyframe the whole window.
"""

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

WINDOW = 12  # keyframes
# The oldest node stays near where the window last placed it: the only thing fixing the
# window's heading and position, since every other factor is relative or about gravity.
ANCHOR_ROTATION = math.radians(2)
ANCHOR_POSITION = 0.05
# A visual factor whose whitened error is past this many sigmas counts less and less.
CAUCHY = 2.0
SCALE_SIGMA = 0.2  # Log-scale prior per keyframe solve, around the last estimate.
# Scale observations kept: a few seconds of frames, as the front end's own scale drifts.
SCALE_MEMORY = 60
TD_STEP = 0.01  # How far the frame delay may wander between keyframe solves, one sigma.
UP = np.array([0.0, 1.0, 0.0])


@dataclass
class Node:
    time: float  # Frame stamp on the pilot clock; exposed at time - td.
    pose: np.ndarray  # World-from-camera, 4x4.


@dataclass
class Visual:
    reference: int  # node id
    frame: int  # node id
    rotation: np.ndarray  # 3x3, reference-from-frame
    translation: np.ndarray  # front-end units, in the reference camera frame
    sigma: np.ndarray  # rotation (rad) and translation (m) sigmas, 6


def _cauchy(whitened):
    """Rescale residual blocks (rows) so their squared norm is the Cauchy cost."""
    squared = np.sum(whitened * whitened, axis=1, keepdims=True)
    cost = CAUCHY**2 * np.log1p(squared / CAUCHY**2)
    factor = np.sqrt(np.divide(cost, squared, out=np.ones_like(squared), where=squared > 1e-12))
    return whitened * factor


class PoseGraph:
    def __init__(self, motion, td=0.0, td_bounds=(0.0, 0.3), estimate_td=False):
        self.motion = motion
        self.td = float(td)
        self.td_bounds = td_bounds
        self.estimate_td = estimate_td
        self.log_scale = 0.0
        # (log metres per front-end unit, sigma), newest last; until the first is solved the
        # front end's scale is unknown, and nothing holds it near its start.
        self.scales = deque(maxlen=SCALE_MEMORY)
        self.scale_known = False
        self.nodes: dict[int, Node] = {}
        self.keyframes: list[int] = []  # oldest first
        self.current: int | None = None
        self.visual: list[Visual] = []
        self.anchor: np.ndarray | None = None  # pose prior on the oldest keyframe
        self.next_id = 0
        self.solves = 0

    # -- the window ---------------------------------------------------------------------

    def latest(self):
        """The id of the newest node, if any."""
        if self.current is not None:
            return self.current
        return self.keyframes[-1] if self.keyframes else None

    def latest_keyframe(self):
        return self.keyframes[-1] if self.keyframes else None

    def fold_scale(self):
        """Hand the visual scale estimate to the front end: returns the factor its lengths
        take, and restates the factors already held in the new units."""
        factor = math.exp(self.log_scale)
        for v in self.visual:
            v.translation = v.translation * factor
        self.scales = deque(
            ((value - self.log_scale, sigma) for value, sigma in self.scales), SCALE_MEMORY
        )
        self.log_scale = 0.0
        return factor

    def add_scale(self, log_scale, sigma):
        """An observation of the front end's scale: log metres per front-end unit."""
        self.scales.append((float(log_scale), float(sigma)))

    def predict(self, t):
        """World-from-camera for a frame stamped t: the newest node moved on by the robot."""
        last = self.latest()
        if last is None:
            return self.motion.at(t - self.td)
        node = self.nodes[last]
        relative, _ = self.motion.between(node.time - self.td, t - self.td)
        return node.pose @ relative

    def add_frame(self, t, pose=None):
        """Make a frame stamped t the current node, replacing a current one not promoted."""
        if self.current is not None:
            self._drop(self.current)
        node = self.next_id
        self.next_id += 1
        self.nodes[node] = Node(t, self.predict(t) if pose is None else np.array(pose, float))
        self.current = node
        if self.anchor is None:
            # The world is the robot's odometry where tracking began.
            self.anchor = self.nodes[node].pose.copy()
        return node

    def add_visual(self, reference, frame, relative, sigma):
        """A front end's reference-from-frame pose (translation in its own units)."""
        if reference not in self.nodes or frame not in self.nodes or reference == frame:
            return
        self.visual.append(
            Visual(
                reference,
                frame,
                np.array(relative[:3, :3], float),
                np.array(relative[:3, 3], float),
                np.asarray(sigma, float),
            )
        )

    def promote(self):
        """The current frame becomes a keyframe; the oldest leaves a full window."""
        if self.current is None:
            return []
        self.keyframes.append(self.current)
        self.current = None
        dropped = []
        while len(self.keyframes) > WINDOW:
            oldest = self.keyframes.pop(0)
            self._drop(oldest)
            dropped.append(oldest)
            # Where the window put the new oldest node holds the world's heading from now on.
            self.anchor = self.nodes[self.keyframes[0]].pose.copy()
        return dropped

    def expire(self, oldest):
        """Drop keyframes exposed before `oldest`, where the robot's motion no longer reaches."""
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
        self.visual = [v for v in self.visual if node not in (v.reference, v.frame)]
        if self.current == node:
            self.current = None

    def pose(self, node):
        return self.nodes[node].pose

    def correction(self, node):
        """World-from-odometry: how far fusion moved this node from the robot's own pose."""
        n = self.nodes[node]
        return n.pose @ np.linalg.inv(self.motion.at(n.time - self.td))

    # -- solving ------------------------------------------------------------------------

    def solve_frame(self):
        """Refine the current node alone, keyframes held."""
        if self.current is None:
            return None
        self._solve([self.current], frame_only=True)
        return self.nodes[self.current].pose

    def solve_window(self):
        """Refine every node, and the frame delay and scale with them."""
        order = self.keyframes + ([self.current] if self.current is not None else [])
        if order:
            self._solve(order, frame_only=False)
        return {k: self.nodes[k].pose for k in self.keyframes}

    def solve_scale(self):
        """Refine the scale alone, every pose and the frame delay held (known poses)."""
        if self.keyframes:
            self._solve([], frame_only=False, estimate_td=False)

    def _solve(self, free, frame_only, estimate_td=True):
        window = self.keyframes + ([self.current] if self.current is not None else [])
        visual = [v for v in self.visual if v.reference in self.nodes and v.frame in self.nodes]
        if frame_only:
            # Only what touches the current node: the newest keyframe and its references.
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
        estimate_td = estimate_td and self.estimate_td and not frame_only
        estimate_scale = not frame_only and bool(self.visual or self.scales)
        observed = np.array(self.scales if estimate_scale else np.empty((0, 2))).reshape(-1, 2)
        # The first observations set the scale freely; after that it drifts slowly.
        scale_prior = estimate_scale and (self.scale_known or not len(observed))
        extra = int(estimate_td) + int(estimate_scale)
        size = 6 * len(free) + extra
        td0, scale0 = self.td, self.log_scale

        rows = np.array([index[n] for n in free], dtype=int)
        start_q = start_r.as_quat()
        ci = np.array([i for i, _ in chain], dtype=int)
        cj = np.array([j for _, j in chain], dtype=int)
        # Noise grows with time and travel, not with td's few milliseconds: fixed per solve.
        robot_sigma = np.array(
            [
                np.sqrt(np.diag(self.motion.between(times[i] - td0, times[j] - td0)[1]))
                for i, j in chain
            ]
        ).reshape(-1, 6)
        gravity_sigma = np.array([self.motion.gravity_sigma(times[i] - td0) for i in rows])
        vi = np.array([index[v.reference] for v in visual], dtype=int)
        vj = np.array([index[v.frame] for v in visual], dtype=int)
        if visual:
            v_rotation_inv = Rotation.from_matrix(np.array([v.rotation for v in visual])).inv()
            v_translation = np.array([v.translation for v in visual])
            v_sigma = np.array([v.sigma for v in visual])
        anchor_r = Rotation.from_matrix(self.anchor[:3, :3]).inv()

        def unpack(x):
            positions = start_p.copy()
            quats = start_q.copy()
            if len(rows):
                delta = x[: 6 * len(free)].reshape(-1, 6)
                quats[rows] = (start_r[rows] * Rotation.from_rotvec(delta[:, :3])).as_quat()
                positions[rows] += delta[:, 3:]
            at = 6 * len(free)
            td = x[at] if estimate_td else td0
            at += int(estimate_td)
            scale = x[at] if estimate_scale else scale0
            return Rotation.from_quat(quats), positions, td, scale

        def residuals(x):
            rotations, positions, td, scale = unpack(x)
            robot_r, robot_p = self.motion.poses(times - td)
            blocks = []
            if len(ci):
                # Robot relative motion between consecutive nodes, in the earlier one's frame.
                ri_inv = rotations[ci].inv()
                measured_inv = robot_r[ci].inv()
                relative_r = measured_inv * robot_r[cj]
                relative_p = measured_inv.apply(robot_p[cj] - robot_p[ci])
                rotation = (relative_r.inv() * ri_inv * rotations[cj]).as_rotvec()
                moved = ri_inv.apply(positions[cj] - positions[ci]) - relative_p
                blocks.append((np.hstack((rotation, moved)) / robot_sigma).ravel())
            if len(rows):
                # Gravity: pitch and roll (world x and z) of each free node against the robot's.
                error = (rotations[rows] * robot_r[rows].inv()).as_rotvec()
                blocks.append((error[:, [0, 2]] / gravity_sigma[:, None]).ravel())
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
            if estimate_td:
                blocks.append(np.array([(td - td0) / TD_STEP]))
            if len(observed):
                error = (scale - observed[:, :1]) / observed[:, 1:]
                blocks.append(_cauchy(error).ravel())
            if scale_prior:
                blocks.append(np.array([(scale - scale0) / SCALE_SIGMA]))
            return np.concatenate(blocks) if blocks else np.zeros(1)

        x0 = np.zeros(size)
        if estimate_td:
            x0[6 * len(free)] = td0
        if estimate_scale:
            x0[6 * len(free) + int(estimate_td)] = scale0
        scale_rows = len(observed) + int(scale_prior) if estimate_scale else 0
        sparsity = (
            None
            if frame_only or not free
            else self._sparsity(
                order,
                free_index,
                chain_nodes,
                visual,
                anchored,
                estimate_td,
                estimate_scale,
                scale_rows,
                size,
            )
        )
        lower = np.full(size, -np.inf)
        upper = np.full(size, np.inf)
        if estimate_td:
            lower[6 * len(free)], upper[6 * len(free)] = self.td_bounds
            x0[6 * len(free)] = float(np.clip(td0, *self.td_bounds))
        result = least_squares(
            residuals,
            x0,
            # A frame's six unknowns, or the scale alone, differentiate faster densely than
            # through sparse groups.
            jac_sparsity=sparsity,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            max_nfev=20 if frame_only else 40,
        )
        if not np.isfinite(result.x).all():
            return
        rotations, positions, td, scale = unpack(result.x)
        matrices = rotations.as_matrix()
        for node, i in zip(free, rows, strict=True):
            pose = np.eye(4)
            pose[:3, :3] = matrices[i]
            pose[:3, 3] = positions[i]
            self.nodes[node].pose = pose
        self.td, self.log_scale = float(td), float(scale)
        self.scale_known = self.scale_known or bool(len(observed))
        self.solves += 1

    def _sparsity(
        self, order, free_index, chain, visual, anchored, estimate_td, scale, scale_rows, size
    ):
        rows = []

        def block(count, nodes, uses_td=False, uses_scale=False):
            columns = set()
            for node in nodes:
                if node in free_index:
                    columns.update(range(6 * free_index[node], 6 * free_index[node] + 6))
            if uses_td and estimate_td:
                columns.add(6 * len(free_index))
            if uses_scale and scale:
                columns.add(6 * len(free_index) + int(estimate_td))
            rows.extend([columns] * count)

        for a, b in chain:
            block(6, (a, b), uses_td=True)
        for node in free_index:
            block(2, (node,), uses_td=True)
        if anchored:
            block(6, (order[0],))
        for v in visual:
            block(6, (v.reference, v.frame), uses_scale=True)
        if estimate_td:
            block(1, (), uses_td=True)
        block(scale_rows, (), uses_scale=True)
        matrix = lil_matrix((max(len(rows), 1), size), dtype=int)
        for r, columns in enumerate(rows):
            for c in columns:
                matrix[r, c] = 1
        return matrix
