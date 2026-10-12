"""Bounded background mapper and stable-slot publication for the existing scene ring."""

import logging
import threading
import time
from collections import deque
from dataclasses import replace

import numpy as np

from .mapper import MAX_GAUSSIANS, GaussianMap

log = logging.getLogger(__name__)

PUBLISH_EVERY = 0.1  # seconds
PUBLISH_RECORDS = 65536  # per publication, unless the map's structure changed
DUTY = 0.25  # share of the time the optimiser may hold the GPU; tracking has the rest


class OnlineMapper:
    antialiased = True

    def __init__(self, budget, report, *, duty=DUTY):
        # Initialize before taking ownership of the scene, so missing CUDA/kernels can
        # select the point fallback without losing an existing map.
        self.engine = GaussianMap(budget)
        self.budget = min(budget, MAX_GAUSSIANS)
        self.duty = duty
        self.report = report
        self.records = np.zeros((self.budget, 4, 4), np.float32)
        self.keys = np.full(self.budget, -1, np.int64)
        self.dirty = np.zeros(self.budget, bool)
        self.retiring = np.zeros(self.budget, bool)
        self.retire_revision = np.full(self.budget, np.inf)
        self.lock = threading.Lock()
        self.engine_lock = threading.Lock()
        self.keyframes = deque()  # never dropped: they carry the map's anchors
        self.views = deque(maxlen=2)  # training-only frames: the newest matter
        self.corrections = {}
        self.forgotten = set()
        self.output = {}
        self.node_slots = {}
        self.requested_budget = self.budget
        self.failure = None
        self.captured_at = 0.0
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, name="ito-mapper", daemon=True)
        self.thread.start()

    @property
    def count(self):
        return int(np.count_nonzero(self.keys >= 0))

    def submit(self, frame):
        torch = self.engine.torch
        # Inputs may be reused by the frontend immediately after this call. Own the
        # tensors and record their producing stream before the low-priority stream reads.
        fields = {
            k: None
            if getattr(frame, k) is None
            else torch.as_tensor(getattr(frame, k), device="cuda").detach().clone()
            for k in ("points", "colors", "confidence")
        }
        frame = replace(
            frame,
            rgb=frame.rgb.copy(),
            pose=frame.pose.copy(),
            depth=None if frame.depth is None else frame.depth.copy(),
            **fields,
        )
        ready = torch.cuda.Event()
        ready.record()
        keyframe = frame.anchor is None or frame.anchor == frame.node
        with self.lock:
            (self.keyframes if keyframe else self.views).append((frame, ready))

    def correct(self, poses):
        with self.lock:
            self.corrections.update({k: v.copy() for k, v in poses.items()})

    def forget(self, nodes):
        with self.lock:
            self.forgotten.update(nodes)
            for name in ("keyframes", "views"):
                queue = getattr(self, name)
                kept = [(f, e) for f, e in queue if f.node not in nodes and f.anchor not in nodes]
                queue.clear()
                queue.extend(kept)
            for node in nodes:
                self.corrections.pop(node, None)

    def resize(self, budget, now):
        self.requested_budget = min(budget, MAX_GAUSSIANS)
        self.budget = self.requested_budget
        # Existing stable indices remain valid until their zero-opacity deltas arrive.
        if self.budget > len(self.keys):
            n = self.budget - len(self.keys)
            self.records = np.pad(self.records, ((0, n), (0, 0), (0, 0)))
            self.keys = np.pad(self.keys, (0, n), constant_values=-1)
            self.dirty = np.pad(self.dirty, (0, n))
            self.retiring = np.pad(self.retiring, (0, n))
            self.retire_revision = np.pad(self.retire_revision, (0, n), constant_values=np.inf)

    def expire(self, now, acknowledged):
        if self.failure:
            raise RuntimeError(self.failure)
        if not self.lock.acquire(False):
            return
        try:
            output, self.output = self.output, {}
        finally:
            self.lock.release()
        if output:
            self.captured_at = max(self.captured_at, max(t for _, t in output.values()))
        output = {n: r for n, (r, _) in output.items()}
        # Release all replaced ranges before allocating, so a full map can exchange
        # anchors in one turn. Coalescing is per node, including deletion tombstones.
        old = {node: self.node_slots.pop(node, np.empty(0, np.int64)) for node in output}
        for slots in old.values():
            self.keys[slots] = -1
            self.records[slots] = 0
            self.dirty[slots] = True
        reserved = {node: old[node][: len(records)] for node, records in output.items()}
        for node, slots in reserved.items():
            self.keys[slots] = node
        free = np.flatnonzero(self.keys < 0)
        cursor = 0
        for node, records in output.items():
            n = len(records)
            if not n:
                continue
            retained = reserved[node]
            extra = max(0, n - len(retained))
            slots = np.concatenate((retained, free[cursor : cursor + extra]))
            cursor += extra
            self.node_slots[node] = slots
            if len(slots) != n:
                raise RuntimeError("Gaussian publication exceeded its slot budget")
            self.keys[slots] = node
            self.records[slots] = records
            self.dirty[slots] = True

    def _publish(self, newest_capture, cursor):
        """Urgent nodes (corrected, forgotten), then round robin; everything after a
        structural change, since splits and prunes touch every anchor."""
        engine = self.engine
        keys = engine.keys()
        urgent = [n for n in engine.urgent]
        rotation = keys[cursor:] + keys[:cursor] if keys else []
        selected, count = [], 0
        sizes = engine.sizes()
        for node in dict.fromkeys(urgent + rotation):
            if node not in engine.changed:
                continue
            selected.append(node)
            count += sizes.get(node, 0)
            if count >= PUBLISH_RECORDS and not engine.structure_changed:
                break
        if not selected:
            return cursor
        records = engine.records_for(selected)
        for node in selected:
            engine.changed.discard(node)
            engine.urgent.discard(node)
        engine.structure_changed = False
        with self.lock:
            self.output.update({n: (r, newest_capture) for n, r in records.items()})
        last = selected[-1]
        return (keys.index(last) + 1) % len(keys) if last in keys else cursor

    def _run(self):
        torch = self.engine.torch
        try:
            stream = torch.cuda.Stream(priority=0)
            with torch.cuda.stream(stream):
                last_publish = 0.0
                newest_capture = 0.0
                cursor = 0
                while not self.stopped.is_set():
                    began = time.monotonic()
                    with self.lock:
                        frames = list(self.keyframes) + list(self.views)
                        self.keyframes.clear()
                        self.views.clear()
                        poses, self.corrections = self.corrections, {}
                        forgotten, self.forgotten = self.forgotten, set()
                    with self.engine_lock:
                        self.engine.resize(self.requested_budget)
                        self.engine.forget(forgotten)
                        for frame, ready in sorted(frames, key=lambda f: f[0].captured_at):
                            stream.wait_event(ready)
                            self.engine.add(frame)
                            newest_capture = max(newest_capture, frame.captured_at)
                        self.engine.correct(poses)
                        self.engine.step()
                        # Synchronize only this stream: keep at most one training step
                        # queued. Sleeping after it bounds the optimizer's duty cycle.
                        stream.synchronize()
                        now = time.monotonic()
                        if now - last_publish >= PUBLISH_EVERY:
                            cursor = self._publish(newest_capture, cursor)
                            last_publish = now
                    elapsed = time.monotonic() - began
                    self.stopped.wait(max(0.005, elapsed * (1 / self.duty - 1)))
        except Exception as exc:
            log.exception("Gaussian mapper stopped")
            self.failure = f"Gaussian mapper stopped: {exc}"

    def align(self, rgb, pose0, intrinsics, **options):
        # Tracking is free to decline alignment while optimization owns the map.
        if not self.engine_lock.acquire(False):
            return np.array(pose0, copy=True), np.full(6, np.inf), 0
        try:
            return self.engine.align(rgb, pose0, intrinsics, **options)
        finally:
            self.engine_lock.release()

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=0.4)
