"""Keyframe anchors (A4, ITO_ANCHORED in reconstruction/gaussian.py).

The worker sends a surface that follows a keyframe in that keyframe's frame as it was placed, and
one world similarity per keyframe (a table, SplatUpdate.anchor_table). When a global solve moves a
keyframe only its similarity changes; the pilot moves what it draws there toward it at a limited
rate (AnchorEase), all of a keyframe's surfaces together, so a correction glides in and never
jumps or tears a keyframe's surfaces apart."""

import math
import os

import numpy as np

# ITO_ANCHOR_EASE="deg/s,cm/s,%/s,tau": the fastest an anchor turns, slides (its keyframe's camera
# centre) and rescales toward its target, and the time constant of the approach (seconds).
_rates = [float(v) for v in os.environ.get("ITO_ANCHOR_EASE", "2,3,2,0.3").split(",")]
TURN_RATE = math.radians(_rates[0])
MOVE_RATE = _rates[1] / 100
SCALE_RATE = _rates[2] / 100
TAU = max(1e-3, _rates[3])
SETTLED = (math.radians(0.005), 2e-4, 1e-5)  # rad, m, log-scale: snap to the target


def split_table(table):
    """(n, 4, 4) similarities with the centres row reset, and the (n, 3) centres."""
    table = np.asarray(table, np.float64)
    matrices = table.copy()
    matrices[:, 3] = (0.0, 0.0, 0.0, 1.0)
    return matrices, table[:, 3, :3].copy()


def world_records(records, anchor_ids, table):
    """Splat records (n, 4, 4) in the world: each carried by its anchor's similarity (for tools
    mirroring the pilot's scene; the renderer does this on the GPU)."""
    records = np.array(records, np.float32, copy=True)
    if table is None or anchor_ids is None or not len(records):
        return records
    from scipy.spatial.transform import Rotation

    matrices, _ = split_table(table)
    ids = np.asarray(anchor_ids, np.int64)
    for entry in np.unique(ids):
        if entry <= 0 or entry >= len(matrices):
            continue
        m = matrices[entry]
        mask = (ids == entry) & (records[:, 0, 3] > 0)
        scale = float(np.cbrt(np.linalg.det(m[:3, :3])))
        records[mask, 0, :3] = records[mask, 0, :3] @ m[:3, :3].T + m[:3, 3]
        records[mask, 1, :3] *= scale
        turn = Rotation.from_matrix(m[:3, :3] / scale)
        quats = Rotation.from_quat(records[mask, 2][:, [1, 2, 3, 0]])
        records[mask, 2] = (turn * quats).as_quat()[:, [3, 0, 1, 2]]
    return records


class AnchorEase:
    """The similarities the renderer draws each anchor with, moved toward the worker's at a limited
    rate. Entry 0 is the world (identity)."""

    def __init__(self, capacity):
        self.current = np.tile(np.eye(4), (capacity, 1, 1))
        self.target = self.current.copy()
        self.centres = np.zeros((capacity, 3))
        self.seen = np.zeros(capacity, bool)  # some drawn surface uses it
        self.count = 1
        self.changed = True
        self.last = None

    def drawn(self, anchor_ids):
        """These anchors now have surfaces drawn by them."""
        self.seen[np.asarray(anchor_ids, np.int64)] = True

    def set(self, table):
        matrices, centres = split_table(table)
        n = min(len(matrices), len(self.current))
        self.target[1:n] = matrices[1:n]
        self.centres[:n] = centres[:n]
        # An anchor nothing is drawn by yet takes its similarity at once: its surfaces will
        # arrive in its frame as it is now. One already drawn by eases there.
        fresh = np.flatnonzero(~self.seen[:n])
        self.current[fresh] = self.target[fresh]
        self.count = max(self.count, n)
        self.changed = True

    def advance(self, now):
        """Move every anchor toward its target for the time since the last call; returns the
        (n, 4, 4) float32 matrices if any changed, else None."""
        dt = 0.0 if self.last is None else min(0.1, max(0.0, now - self.last))
        self.last = now
        n = self.count
        moving = np.flatnonzero(np.abs(self.target[:n] - self.current[:n]).max(axis=(1, 2)) > 1e-9)
        if len(moving):
            self.current[moving] = self._step(
                self.current[moving], self.target[moving], self.centres[moving], dt
            )
            self.changed = True
        if not self.changed:
            return None
        self.changed = False
        return self.current[:n].astype(np.float32)

    @staticmethod
    def _step(current, target, centres, dt):
        from scipy.spatial.transform import Rotation

        delta = target @ np.linalg.inv(current)  # world similarity, current -> target
        scale = np.cbrt(np.linalg.det(delta[:, :3, :3]))
        rotation = delta[:, :3, :3] / scale[:, None, None]
        pivot = np.einsum("nij,nj->ni", current[:, :3, :3], centres) + current[:, :3, 3]
        moved = np.einsum("nij,nj->ni", delta[:, :3, :3], pivot) + delta[:, :3, 3]
        slide = moved - pivot
        rotvec = Rotation.from_matrix(rotation).as_rotvec()
        angle = np.linalg.norm(rotvec, axis=1)
        distance = np.linalg.norm(slide, axis=1)
        log_scale = np.abs(np.log(scale))
        fraction = np.full(len(current), 1 - math.exp(-dt / TAU))
        with np.errstate(divide="ignore", invalid="ignore"):
            for amount, rate in (
                (angle, TURN_RATE),
                (distance, MOVE_RATE),
                (log_scale, SCALE_RATE),
            ):
                fraction = np.minimum(fraction, np.where(amount > 0, rate * dt / amount, 1.0))
        settled = (angle < SETTLED[0]) & (distance < SETTLED[1]) & (log_scale < SETTLED[2])
        fraction = np.where(settled, 1.0, np.clip(fraction, 0.0, 1.0))
        part = Rotation.from_rotvec(rotvec * fraction[:, None]).as_matrix()
        part *= np.exp(np.log(scale) * fraction)[:, None, None]
        step = np.tile(np.eye(4), (len(current), 1, 1))
        step[:, :3, :3] = part
        step[:, :3, 3] = pivot + slide * fraction[:, None] - np.einsum("nij,nj->ni", part, pivot)
        out = step @ current
        out[settled] = target[settled]
        return out
