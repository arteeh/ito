"""Stable Gaussian slots and acknowledged retirement shared by all live inputs."""

import sys
from contextlib import nullcontext

import numpy as np

KEEP = 1e30
FIRST_SLOTS = 65_536
MOVED = 1 << 62
SLOT_FIELDS = {
    "records": (np.float32, 0, (4, 4)),
    "keys": (np.int64, -1, ()),
    "seen": (np.float64, -np.inf, ()),
    "retiring": (bool, False, ()),
    "pressure_victims": (bool, False, ()),
    "dirty": (bool, False, ()),
    "refined": (bool, False, ()),  # changed by training: publishes after dirty slots
    "retire_revision": (np.float64, np.inf, ()),
    "release_after": (np.float64, np.inf, ()),
    "sequences": (np.uint64, 0, ()),
    "captures": (np.float64, 0, ()),
    "support": (np.uint32, 0, ()),
    "uncertainty": (np.float32, 1, ()),
    "contribution": (np.float32, 0, ()),
    "contradictions": (np.uint8, 0, ()),
    "owners": (np.int64, -1, ()),
}


class GaussianSlots:
    def __init__(
        self,
        budget,
        intrinsics,
        *,
        voxel_size=0.04,
        fade_seconds=0.5,
        device="auto",
    ):
        self.admission = None
        self.xp = np
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("Reconstruction device must be auto, cpu or cuda")
        if device != "cpu":
            try:
                import cupy

                if cupy.cuda.runtime.getDeviceCount():
                    float((cupy.arange(4, dtype=cupy.float32) * 2).sum())
                    self.xp = cupy
            except Exception as exc:
                if device == "cuda":
                    raise RuntimeError(
                        "CUDA reconstruction needs CuPy and an NVIDIA CUDA device"
                    ) from exc
            if device == "cuda" and self.xp is np:
                raise RuntimeError("CUDA reconstruction needs an NVIDIA CUDA device")
        self.intrinsics = intrinsics
        self.voxel_size, self.fade = voxel_size, fade_seconds
        for name, (dtype, _, shape) in SLOT_FIELDS.items():
            setattr(self, name, np.zeros((0, *shape), dtype))
        self.grow(min(budget, FIRST_SLOTS))
        self.source_sequence = 0
        self.source_captured = 0.0
        self.budget = budget
        self.refreshed = 0

    def bind_admission(self, admission):
        self.admission = admission
        if self.xp is not np:
            admission.memory_used = self.cuda_memory_bytes
            admission.bind_stream(self.xp)

    def cuda_memory_bytes(self):
        torch = sys.modules.get("torch")
        return self.xp.get_default_memory_pool().total_bytes() + (
            torch.cuda.memory_reserved() if torch is not None else 0
        )

    def gpu_chunk(self, kind="map"):
        if self.admission is None or self.xp is np:
            return nullcontext()
        return self.admission.cuda_chunk(
            (self.xp.cuda.Event(), self.xp.cuda.Event(), self.xp.cuda.get_elapsed_time),
            kind=kind,
        )

    @property
    def count(self):
        return int(np.count_nonzero(self.keys >= 0))

    def grow(self, slots):
        added = slots - len(self.keys)
        for name, (dtype, empty, shape) in SLOT_FIELDS.items():
            extra = np.full((added, *shape), empty, dtype)
            setattr(self, name, np.concatenate((getattr(self, name), extra)))

    def resize(self, budget, now):
        self.budget = budget
        excess = max(0, self.count - budget - int(self.retiring.sum()))
        candidates = np.flatnonzero((self.keys >= 0) & ~self.retiring)
        if excess:
            self.retire(self.victims(candidates, excess), pressure=True)
        return excess

    def compact(self):
        high = np.arange(self.budget, len(self.keys))
        movers = high[(self.keys[high] >= 0) & ~self.retiring[high]]
        free = np.flatnonzero(self.keys[: self.budget] < 0)
        moved = min(len(movers), len(free))
        if moved:
            source, target = movers[:moved], free[:moved]
            for name in SLOT_FIELDS:
                array = getattr(self, name)
                array[target] = array[source]
            self.sequences[target] = 0
            self.dirty[target] = True
            self.keys[source] = MOVED + source
            self.retire(source, pressure=True)
        if (self.keys[self.budget :] >= 0).any():
            return False
        for name in SLOT_FIELDS:
            setattr(self, name, getattr(self, name)[: self.budget].copy())
        return True

    def oldest(self, candidates, n):
        if n <= 0:
            return candidates[:0]
        if n >= len(candidates):
            return candidates
        return candidates[np.argpartition(self.seen[candidates], n - 1)[:n]]

    def retire(self, indices, *, pressure=False):
        self.records[indices, 3, 3] = 1
        self.retiring[indices] = True
        self.pressure_victims[indices] = pressure
        self.sequences[indices] = 0
        self.dirty[indices] = True

    def expire(self, now, acknowledged):
        accepted = self.retiring & (self.retire_revision <= acknowledged)
        self.release_after[accepted] = np.minimum(
            self.release_after[accepted], now + self.fade + 0.05
        )
        expired = (self.keys >= 0) & self.retiring & (self.release_after <= now)
        self.refined[expired] = False
        self.keys[expired] = -1
        self.sequences[expired] = 0
        self.records[expired] = 0
        self.retiring[expired] = False
        self.pressure_victims[expired] = False
        self.retire_revision[expired] = np.inf
        self.release_after[expired] = np.inf
        self.dirty[expired] = True
