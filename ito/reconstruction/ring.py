"""Single producer/consumer rings. Full rings retain dirty slots at the producer.

Each packet replaces only the listed stable slots; zero opacity evicts a slot.
Locks are always tried, never waited on, including publication and consumption.
The robot's motion has a ring of its own (MotionRing): the worker may wait for it.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

MAX_ANCHORS = 8192  # gaussian.MAX_ANCHORS: table entries, 0 the world, keyframe k at k + 1


@dataclass(frozen=True)
class SplatUpdate:
    indices: np.ndarray
    records: np.ndarray
    capacity: int
    count: int
    revision: int
    captured_at: float
    epoch: float
    fade_seconds: float
    acknowledge: Callable[[], None] | None = None
    observations: tuple[tuple[int, float], ...] = ()
    observed_at: np.ndarray | None = None
    coordinate_frame: str = "world"
    # ITO_EXPOSURE: per-channel log gain to draw the map at the camera's usual exposure.
    color_gain: tuple = (0.0, 0.0, 0.0)
    # Records are baked into the persistent root anchor before entering the ring.
    anchors: tuple = (
        (0, (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    )
    # ITO_ANCHORED (A4): each record's anchor (an anchor_table entry; 0 = the world), and the
    # newest table if it changed since the last packet: (n, 4, 4) world-from-anchor similarities
    # whose last row holds the anchor's centre (x, y, z, 1) instead of (0, 0, 0, 1). A record's
    # world form is its table entry applied to it (anchors.world_records).
    anchor_ids: np.ndarray | None = None
    anchor_table: np.ndarray | None = None


class UpdateRing:
    def __init__(
        self,
        context,
        epoch: float,
        fade_seconds: float,
        *,
        slots=4,
        batch=4096,
        coordinate_frame="world",
    ):
        self.slots, self.batch = slots, batch
        self.epoch, self.fade_seconds = epoch, fade_seconds
        self.coordinate_frame = coordinate_frame
        self.data = context.RawArray("f", slots * batch * 16)
        self.indices = context.RawArray("I", slots * batch)
        self.sequences = context.RawArray("Q", slots * batch)
        self.captures = context.RawArray("d", slots * batch)
        self.headers = context.RawArray("d", slots * 6)
        self.ready = context.RawArray("b", slots)
        self.locks = [context.Lock() for _ in range(slots)]
        self.read_position = self.write_position = 0
        self.color_gain = context.RawArray("d", 3)
        self.anchor_ids = context.RawArray("I", slots * batch)
        self.anchor_table = context.RawArray("f", MAX_ANCHORS * 16)
        self.anchor_header = context.RawArray("q", 2)  # revision, entries
        self.anchor_lock = context.Lock()
        self.anchors_seen = 0
        self.acknowledged = context.RawValue("q", -1)
        self.ack_lock = context.Lock()

    def publish(
        self,
        indices,
        records,
        capacity,
        count,
        revision,
        captured_at,
        sequences=None,
        captures=None,
        anchors=None,
    ):
        slot = self.write_position % self.slots
        if not self.locks[slot].acquire(False):
            return False
        try:
            if self.ready[slot]:
                return False
            n = len(indices)
            if n > self.batch:
                raise ValueError("Update exceeds ring packet size")
            np.frombuffer(self.data, np.float32).reshape(self.slots, self.batch, 4, 4)[slot, :n] = (
                records
            )
            np.frombuffer(self.indices, np.uint32).reshape(self.slots, self.batch)[slot, :n] = (
                indices
            )
            np.frombuffer(self.sequences, np.uint64).reshape(self.slots, self.batch)[slot, :n] = (
                sequences if sequences is not None else 0
            )
            np.frombuffer(self.captures).reshape(self.slots, self.batch)[slot, :n] = (
                captures if captures is not None else captured_at
            )
            np.frombuffer(self.anchor_ids, np.uint32).reshape(self.slots, self.batch)[slot, :n] = (
                anchors if anchors is not None else 0
            )
            np.frombuffer(self.headers).reshape(self.slots, 6)[slot] = (
                n,
                capacity,
                count,
                revision,
                captured_at,
                0,
            )
            self.ready[slot] = 1
            self.write_position += 1
            return True
        finally:
            self.locks[slot].release()

    def publish_anchors(self, table, centres):
        """ITO_ANCHORED: the whole anchor table (n, 4, 4 world similarities; centres n, 3), for
        the next packet the pilot polls. False if the pilot holds it (try again)."""
        n = min(len(table), MAX_ANCHORS)
        if not self.anchor_lock.acquire(False):
            return False
        try:
            out = np.frombuffer(self.anchor_table, np.float32).reshape(MAX_ANCHORS, 4, 4)
            out[:n] = table[:n]
            out[:n, 3, :3] = centres[:n]
            out[:n, 3, 3] = 1
            header = np.frombuffer(self.anchor_header, np.int64)
            header[1] = n
            header[0] += 1
            return True
        finally:
            self.anchor_lock.release()

    def _anchor_table(self):
        if not self.anchor_lock.acquire(False):
            return None
        try:
            revision, n = np.frombuffer(self.anchor_header, np.int64)
            if revision == self.anchors_seen:
                return None
            self.anchors_seen = int(revision)
            table = np.frombuffer(self.anchor_table, np.float32).reshape(MAX_ANCHORS, 4, 4)
            return table[: int(n)].copy()
        finally:
            self.anchor_lock.release()

    def acknowledge(self, revision):
        if self.ack_lock.acquire(False):
            try:
                self.acknowledged.value = max(self.acknowledged.value, revision)
            finally:
                self.ack_lock.release()

    def last_acknowledged(self):
        if not self.ack_lock.acquire(False):
            return -1
        try:
            return self.acknowledged.value
        finally:
            self.ack_lock.release()

    def poll(self):
        slot = self.read_position % self.slots
        if not self.locks[slot].acquire(False):
            return None
        try:
            if not self.ready[slot]:
                return None
            n, capacity, count, revision, captured_at, _ = np.frombuffer(self.headers).reshape(
                self.slots, 6
            )[slot]
            n = int(n)
            records = (
                np.frombuffer(self.data, np.float32)
                .reshape(self.slots, self.batch, 4, 4)[slot, :n]
                .copy()
            )
            indices = (
                np.frombuffer(self.indices, np.uint32)
                .reshape(self.slots, self.batch)[slot, :n]
                .copy()
            )
            sequences = np.frombuffer(self.sequences, np.uint64).reshape(self.slots, self.batch)[
                slot, :n
            ]
            captures = np.frombuffer(self.captures).reshape(self.slots, self.batch)[slot, :n]
            anchor_ids = (
                np.frombuffer(self.anchor_ids, np.uint32)
                .reshape(self.slots, self.batch)[slot, :n]
                .copy()
            )
            unique, first = np.unique(sequences, return_index=True)
            observations = tuple(
                (int(seq), float(captures[i])) for seq, i in zip(unique, first, strict=True) if seq
            )
            self.ready[slot] = 0
            self.read_position += 1
            return SplatUpdate(
                indices,
                records,
                int(capacity),
                int(count),
                int(revision),
                captured_at,
                self.epoch,
                self.fade_seconds,
                lambda: self.acknowledge(int(revision)),
                observations,
                captures.copy(),
                self.coordinate_frame,
                color_gain=tuple(self.color_gain),
                anchor_ids=anchor_ids,
                anchor_table=self._anchor_table(),
            )
        finally:
            self.locks[slot].release()


MOTION_STATES = ("standing", "walking", "turning", "held", "fallen")
MOTION_CAPACITY = 512  # 10 s at robotd's 50 Hz
MOTION_FIELDS = 1 + 16 + 3 + 1


class MotionRing:
    def __init__(self, context, capacity=MOTION_CAPACITY):
        self.capacity = capacity
        self.data = context.RawArray("d", capacity * MOTION_FIELDS)
        self.written = context.RawValue("Q", 0)
        self.lock = context.Lock()
        self.pending = []

    def put(self, samples):
        self.pending.extend(samples)
        del self.pending[: -self.capacity]
        if not self.pending or not self.lock.acquire(False):
            return
        try:
            data = np.frombuffer(self.data).reshape(self.capacity, MOTION_FIELDS)
            written = self.written.value
            for time, pose, spin, state in self.pending:
                row = data[written % self.capacity]
                row[0] = time
                row[1:17] = np.asarray(pose, dtype=np.float64).ravel()
                row[17:20] = spin
                row[20] = MOTION_STATES.index(state)
                written += 1
            self.written.value = written
            self.pending.clear()
        finally:
            self.lock.release()

    def read(self, cursor):
        with self.lock:
            written = self.written.value
            first = max(cursor, written - self.capacity)
            data = np.frombuffer(self.data).reshape(self.capacity, MOTION_FIELDS)
            rows = data[np.arange(first, written) % self.capacity].copy()
        return rows, written
