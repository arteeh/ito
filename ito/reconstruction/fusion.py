"""Fuse the robot's own camera motion with what vision sees, in a small sliding window.

The robot measures where its camera points better than monocular vision does (encoders, IMU,
50 Hz); vision corrects what the robot measures badly (yaw drift over minutes, slipping feet).
The world is the robot's odometry frame where it started, so there is no map to restart: a
frame vision cannot use simply adds no visual factor, and its pose comes from the robot.

Variables: one pose per node (the last WINDOW keyframes and the current frame) and a
log-scale for the visual translations. Factors: the robot's relative motion between
consecutive nodes, gravity (pitch and roll) per node, a weak prior holding the oldest node
where the window last put it, and visual relative poses with a Cauchy loss. Solved with
scipy's least squares on a sparse Jacobian: per frame the current node alone, per keyframe
the whole window.

The frame delay td (a frame stamped t was exposed at t - td) is estimated apart from the
poses too, from every visual rotation measured while the camera turned: the robot's
rotation between the same two frames matches it only at the right delay. A delay a few
milliseconds off turns every frame of a fast turn by the turn rate times the error, so it
has to settle within the first turn, not over a window's slow keyframes.

A head IMU on the camera's own link (sensors.HeadSensors), where the robot has one, adds two
more: the gyro's turn between consecutive nodes and the chip's gravity per node, both much
tighter than the robot's, which reaches the camera through every geared joint on the way.

A map aligner (the Gaussian map rendered at a frame's pose and compared with its image)
adds a map factor: the frame's pose in the world the map was built in, with its sigma,
under the same Cauchy loss as a visual factor.

The front end's lengths become metres through one log-scale estimated apart from the poses,
from direct observations of it (the floor under a frame, slam.py), not from the visual
translations: a front end seeded at the robot's prediction mostly echoes the robot's travel
in them, in whatever units its map has.
"""

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

# Keyframes. Coming back to a view after a turn too fast to track, a frame should find the
# keyframes vision placed before the turn still in the window, not only those the robot's
# motion placed through it: on the sweep180 replay 30 halves the rotation p95 of 12.
WINDOW = 30
# The oldest node stays near where the window last placed it: the only thing fixing the
# window's heading and position, since every other factor is relative or about gravity.
ANCHOR_ROTATION = math.radians(2)
ANCHOR_POSITION = 0.05
# A visual factor whose whitened error is past this many sigmas counts less and less.
CAUCHY = 2.0
SCALE_SIGMA = 0.2  # Log-scale prior per solve, around the last estimate.
# A visual translation's length is only as good as its keyframe's scale, which wanders
# between keyframes: against the robot's travel, tour replay factors spread p10-p90 over
# 1.1-1.7x. Its sigma grows by this fraction of its length.
VISUAL_SCALE = 0.3
# Scale observations kept: about two seconds of frames, as the front end's own scale drifts.
SCALE_MEMORY = 20
# Frame-delay observations kept, and the camera's turn between a visual rotation's two
# frames, per second of delay, below which one says too little to keep.
DELAY_MEMORY = 60
DELAY_TURN = math.radians(20)
DELAY_REACH = 0.05  # seconds either side of the current delay an observation may land
DELAY_STEP = 0.001
UP = np.array([0.0, 1.0, 0.0])
# A map alignment's sigma is the photometric fit's own, linearised: it knows nothing of the
# map's errors. Never tighter than this (rotation vector rad x3, translation m x3).
MAP_FLOOR = np.repeat((math.radians(0.05), 0.005), 3)
# Published poses kept for pose_at: at 10 Hz about as far back as a minute of motion reaches.
PUBLISHED = 600


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
    def __init__(self, motion, td=0.0, td_bounds=(0.0, 0.3), estimate_td=False, inertial=None):
        self.motion = motion
        self.inertial = inertial  # sensors.HeadSensors, or None
        self.td = float(td)
        self.td_bounds = td_bounds
        self.estimate_td = estimate_td
        self.delays = deque(maxlen=DELAY_MEMORY)  # (seconds, sigma), newest last
        self.log_scale = 0.0
        # (log metres per front-end unit, sigma), newest last; until the first is solved the
        # front end's scale is unknown, and nothing holds it near its start.
        self.scales = deque(maxlen=SCALE_MEMORY)
        self.scale_known = False
        self.nodes: dict[int, Node] = {}
        self.keyframes: list[int] = []  # oldest first
        self.current: int | None = None
        self.visual: list[Visual] = []
        self.mapped: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # node: (pose, sigma 6)
        self.anchor: np.ndarray | None = None  # pose prior on the oldest keyframe
        self.next_id = 0
        self.solves = 0
        # (time, node id, pose) of every published node, oldest first; keyframes' follow
        # their window solves while they are in the window.
        self.published = deque(maxlen=PUBLISHED)

    # -- the window ---------------------------------------------------------------------

    def latest(self):
        """The id of the newest node, if any."""
        if self.current is not None:
            return self.current
        return self.keyframes[-1] if self.keyframes else None

    def latest_keyframe(self):
        return self.keyframes[-1] if self.keyframes else None

    def fold_scale(self):
        """Hand the scale estimate to the front end: returns the factor its lengths take,
        and restates the factors and observations already held in the new units."""
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
        translation = np.array(relative[:3, 3], float)
        sigma = np.array(sigma, float)
        sigma[3:] = np.hypot(sigma[3:], VISUAL_SCALE * np.linalg.norm(translation))
        self.visual.append(
            Visual(reference, frame, np.array(relative[:3, :3], float), translation, sigma)
        )
        if self.estimate_td:
            self._observe_delay(self.visual[-1])

    def add_map(self, node, pose, sigma):
        """Where a map aligner placed a node: world-from-camera, and the one-sigma rotation
        vector (rad) and translation (m) of it, no tighter than MAP_FLOOR. False if the
        answer says nothing."""
        sigma = np.asarray(sigma, float)
        if node not in self.nodes or sigma.shape != (6,) or not np.isfinite(sigma).all():
            return False
        if not np.isfinite(pose).all():
            return False
        self.mapped[node] = (np.array(pose, float), np.maximum(sigma, MAP_FLOOR))
        return True

    def _observe_delay(self, visual):
        """The delay at which the robot's rotation between a visual factor's two frames
        matches the one vision measured, if the camera turned enough to tell."""
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
        # How fast the robot's relative rotation moves with the delay, around the best one.
        turn = (robot[best - 1].inv() * robot[best + 1]).magnitude() / (2 * DELAY_STEP)
        if turn < DELAY_TURN:
            return
        # Sub-step minimum of the parabola through the three around it.
        left, middle, right = error[best - 1 : best + 2]
        curve = left - 2 * middle + right
        shift = 0.5 * (left - right) / curve if curve > 1e-12 else 0.0
        delay = float(delays[best] + np.clip(shift, -1, 1) * DELAY_STEP)
        self.delays.append((delay, float(visual.sigma[0]) / turn))

    def solve_delay(self):
        """The frame delay from its observations, robustly, as solve_scale the scale."""
        if not self.estimate_td or not self.delays:
            return
        values, sigmas = np.array(self.delays).T
        delay = float(np.median(values))
        for _ in range(8):
            weights = 1 / (sigmas**2 + ((delay - values) / CAUCHY) ** 2)
            delay = float(weights @ values / weights.sum())
        self.td = float(np.clip(delay, *self.td_bounds))

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
        self.mapped.pop(node, None)
        self.visual = [v for v in self.visual if node not in (v.reference, v.frame)]
        if self.current == node:
            self.current = None

    def pose(self, node):
        return self.nodes[node].pose

    def correction(self, node):
        """World-from-odometry: how far fusion moved this node from the robot's own pose."""
        n = self.nodes[node]
        return n.pose @ np.linalg.inv(self.motion.at(n.time - self.td))

    def publish(self, node):
        """Remember a node's pose as the one a frame stamped at its time was given."""
        n = self.nodes[node]
        self.published.append((n.time, node, n.pose.copy()))

    def pose_at(self, t):
        """World-from-camera for any frame stamped t, tracked or not: the robot's motion at
        its exposure, moved by fusion's correction interpolated between the published
        poses around it (held past either end)."""
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

    # -- solving ------------------------------------------------------------------------

    def solve_frame(self):
        """Refine the current node alone, keyframes held."""
        if self.current is None:
            return None
        self._solve([self.current], frame_only=True)
        return self.nodes[self.current].pose

    def solve_window(self):
        """Refine every node, and the frame delay with them."""
        order = self.keyframes + ([self.current] if self.current is not None else [])
        if order:
            self._solve(order, frame_only=False)
            self._refresh_published()
        return {k: self.nodes[k].pose for k in self.keyframes}

    def solve_scale(self):
        """The scale from its direct observations, each a little apart from the last."""
        if not self.scales:
            return
        values, sigmas = np.array(self.scales).T
        prior = 1 / SCALE_SIGMA**2 if self.scale_known else 0.0
        scale = float(np.median(values)) if not self.scale_known else self.log_scale
        # One unknown: iteratively reweighted least squares under the Cauchy loss.
        for _ in range(8):
            weights = 1 / (sigmas**2 + ((scale - values) / CAUCHY) ** 2)
            scale = (weights @ values + prior * self.log_scale) / (weights.sum() + prior)
        self.log_scale = float(scale)
        self.scale_known = True

    def _solve(self, free, frame_only):
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
        # The visual translations' units float a little per solve, around the scale from the
        # floor: they need not match it exactly, and are not handed to the front end.
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
            if turns:
                # The IMU's turn between consecutive nodes, in the earlier one's frame.
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
            # A frame's six unknowns differentiate faster densely than through sparse groups.
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
        """The head IMU's factors for this solve: (i, j, inverse turn, sigma) per chain pair
        and (i, up in the camera, sigma) per free node, by solve index."""
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
