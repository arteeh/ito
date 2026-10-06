"""Single producer/consumer ring. Full rings retain dirty slots at the producer.

Each packet replaces only the listed stable slots; zero opacity evicts a slot.
Locks are always tried, never waited on, including publication and consumption.
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


class UpdateRing:
    def __init__(self, context, epoch: float, fade_seconds: float, *, slots=4, batch=4096):
        self.slots, self.batch = slots, batch
        self.epoch, self.fade_seconds = epoch, fade_seconds
        self.data = context.RawArray("f", slots * batch * 16)
        self.indices = context.RawArray("I", slots * batch)
        self.headers = context.RawArray("d", slots * 6)
        self.ready = context.RawArray("b", slots)
        self.locks = [context.Lock() for _ in range(slots)]
        self.read_position = self.write_position = 0
        self.acknowledged = context.RawValue("q", -1)
        self.ack_lock = context.Lock()

    def publish(self, indices, records, capacity, count, revision, captured_at):
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
            )
        finally:
            self.locks[slot].release()
