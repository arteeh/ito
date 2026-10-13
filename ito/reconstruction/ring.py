"""Single producer/consumer rings. Full rings retain dirty slots at the producer.

Each packet replaces only the listed stable slots; zero opacity evicts a slot.
Locks are always tried, never waited on, including publication and consumption.
The robot's motion has a ring of its own (MotionRing): the worker may wait for it.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


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
    # Actual contributing frames, not the most recent backend call. Empty for evictions.
    observations: tuple[tuple[int, float], ...] = ()


class UpdateRing:
    def __init__(self, context, epoch: float, fade_seconds: float, *, slots=4, batch=4096):
        self.slots, self.batch = slots, batch
        self.epoch, self.fade_seconds = epoch, fade_seconds
        self.data = context.RawArray("f", slots * batch * 16)
        self.indices = context.RawArray("I", slots * batch)
        self.sequences = context.RawArray("Q", slots * batch)
        self.captures = context.RawArray("d", slots * batch)
        self.headers = context.RawArray("d", slots * 6)
        self.ready = context.RawArray("b", slots)
        self.locks = [context.Lock() for _ in range(slots)]
        self.read_position = self.write_position = 0
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
            )
        finally:
            self.locks[slot].release()


# The robot's motion samples cross to the worker as rows of time, a 4x4 world-from-camera
# pose, camera-frame angular velocity and a state index.
MOTION_STATES = ("standing", "walking", "turning", "held", "fallen")
MOTION_CAPACITY = 512  # 10 s at robotd's 50 Hz
MOTION_FIELDS = 1 + 16 + 3 + 1


class MotionRing:
    """Single-writer shared ring of motion samples from the pilot to the worker.

    The writer never waits: a busy lock keeps the samples for its next call. The worker
    reads everything written since its last read.
    """

    def __init__(self, context, capacity=MOTION_CAPACITY):
        self.capacity = capacity
        self.data = context.RawArray("d", capacity * MOTION_FIELDS)
        self.written = context.RawValue("Q", 0)
        self.lock = context.Lock()
        self.pending = []

    def put(self, samples):
        """samples: (time, 4x4 world-from-camera, angular velocity, state name), oldest first."""
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
        """(rows since cursor, new cursor); rows lost to a lapped reader are skipped."""
        with self.lock:
            written = self.written.value
            first = max(cursor, written - self.capacity)
            data = np.frombuffer(self.data).reshape(self.capacity, MOTION_FIELDS)
            rows = data[np.arange(first, written) % self.capacity].copy()
        return rows, written
