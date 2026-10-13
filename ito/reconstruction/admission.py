"""Bound queued CUDA work, not display gaps. Only reconstruction waits.

Torch, CuPy and the shipped native kernels share a nonblocking, normal-priority
stream. Completed display frames feed a short cooldown when rendering runs late;
stream priority is a scheduling hint, never a reservation or kernel preemption.
"""

from collections import deque
from contextlib import contextmanager

from ito import clock

MAX_SPLATS = 524_288
MAX_CUDA_BYTES = 6_000_000_000
PUBLICATION_PACKETS = 2
MATCH_QUERIES = 4096
MAX_PENDING_CHUNKS = 4


class Wake:
    """One pending notification, including when a receiver dies while waiting."""

    def __init__(self, context):
        self.semaphore = context.BoundedSemaphore(1)
        self.semaphore.acquire(False)

    def notify(self):
        try:
            self.semaphore.release()
        except ValueError:
            pass

    def wait(self, timeout):
        return self.semaphore.acquire(timeout=timeout)


class Admission:
    def __init__(self, context, stopped):
        self.stopped = stopped
        self.enabled = context.RawValue("b", False)
        self.pause_until = context.RawValue("d", 0)
        self.last_progress = context.RawValue("d", 0)
        self.memory_peak = context.RawValue("Q", 0)
        self.active_seconds = context.RawValue("d", 0)
        self.chunks = context.RawValue("Q", 0)
        self.overruns = context.RawValue("Q", 0)
        self.wake = Wake(context)
        self.memory_used = None
        self.pending = deque()
        self.cooldown = 0.0
        self.recovered_at = 0.0
        self.stream = self.torch_stream = None

    def completed(self, began, period):
        """After both eyes and UI complete on GL; no render-side lock or CUDA wait."""
        now = clock.now()
        self.enabled.value = True
        if now - began > period:
            self.cooldown = min(0.020, max(0.002, self.cooldown * 2))
            self.recovered_at = now + 0.25
            self.overruns.value += 1
        elif now >= self.recovered_at:
            self.cooldown = max(0.0, self.cooldown - 0.001)
        self.pause_until.value = now + self.cooldown
        self.wake.notify()

    def available(self, seconds=None):
        return clock.now() >= self.pause_until.value

    def acquire(self, seconds=None):
        import multiprocessing as mp

        parent = mp.parent_process()
        while not self.stopped.value and (parent is None or parent.is_alive()):
            if self.available():
                return True
            self.wake.wait(min(0.005, max(0, self.pause_until.value - clock.now())))
        return False

    def bind_stream(self, xp):
        """Startup transfers finish once before changing streams; never during display."""
        import sys

        xp.cuda.get_current_stream().synchronize()
        torch = sys.modules.get("torch")
        if torch is not None:
            torch.cuda.synchronize()
            self.torch_stream = torch.cuda.Stream(priority=0)
            self.stream = xp.cuda.ExternalStream(self.torch_stream.cuda_stream)
        else:
            self.stream = xp.cuda.Stream(non_blocking=True)

    def check_memory(self):
        if self.memory_used is not None:
            used = self.memory_used()
            self.memory_peak.value = max(used, self.memory_peak.value)
            if used > MAX_CUDA_BYTES:
                raise RuntimeError("Reconstruction exceeds its 6 GB CUDA budget")

    def collect(self):
        while self.pending and self.pending[0][1].query():
            start, end, elapsed = self.pending.popleft()
            self.active_seconds.value += elapsed(start, end) / 1000
            self.chunks.value += 1
            self.last_progress.value = clock.now()

    def checkpoint(self):
        self.collect()
        # Queue a few units ahead to hide launch overhead, without an entire model
        # already enqueued when a render miss asks us to back off. Query, don't fence.
        while len(self.pending) >= MAX_PENDING_CHUNKS:
            if not self.acquire():
                raise InterruptedError("Reconstruction stopped")
            self.wake.wait(0.001)
            self.collect()
        if not self.acquire():
            raise InterruptedError("Reconstruction stopped")
        self.check_memory()

    @contextmanager
    def cuda_chunk(self, synchronize=None, events=None, *, kind="map"):
        self.checkpoint()
        if events is not None:
            start, end, elapsed = events
            start.record()
        try:
            yield
        finally:
            if events is not None:
                end.record()
                self.pending.append((start, end, elapsed))
            self.check_memory()

    @contextmanager
    def torch_chunks(self):
        """One stream for an integration; fence only at its CPU publication boundary."""
        from contextlib import nullcontext

        if self.stream is None:
            yield
            return
        if self.torch_stream is not None:
            import torch

            scope = torch.cuda.stream(self.torch_stream)
        else:
            scope = nullcontext()
        with scope, self.stream:
            self.checkpoint()
            try:
                yield
            finally:
                self.stream.synchronize()
                self.collect()
                self.check_memory()
                self.last_progress.value = clock.now()


def bind_model(model, admission):
    """Neural blocks bound launch-ahead; events are collected without per-block waits."""
    import torch

    blocks = [
        block
        for name in ("enc_blocks", "dec_blocks", "dec_blocks2")
        for block in getattr(model, name)
    ]
    inside = {id(module) for block in blocks for module in block.modules()}
    blocks.extend(
        module
        for module in model.modules()
        if isinstance(module, torch.nn.Conv2d) and id(module) not in inside
    )
    handles = []
    for block in blocks:
        scope = []

        def before(module, inputs, *, scope=scope):
            chunk = admission.cuda_chunk(
                events=(
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                    lambda start, end: start.elapsed_time(end),
                )
            )
            chunk.__enter__()
            scope.append(chunk)

        def after(module, inputs, result, *, scope=scope):
            if scope:
                scope.pop().__exit__(None, None, None)

        handles.extend(
            (
                block.register_forward_pre_hook(before),
                block.register_forward_hook(after, always_call=True),
            )
        )
    return handles


def bind_native(admission):
    """Keep short matching kernels and exact query outputs, without fencing each tile."""
    import mast3r_slam_backends as native
    import torch

    for name in ("iter_proj", "refine_matches"):
        original = getattr(native, name)

        def bounded(image, queries, initial, *options, original=original):
            if not admission.enabled.value:
                return original(image, queries, initial, *options)
            outputs = None
            for member in range(initial.shape[0]):
                for start in range(0, initial.shape[1], MATCH_QUERIES):
                    end = min(start + MATCH_QUERIES, initial.shape[1])
                    with admission.cuda_chunk(
                        events=(
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                            lambda start, end: start.elapsed_time(end),
                        )
                    ):
                        # Upstream's 16-thread blocks have no tail guard.
                        q = queries[member : member + 1, start:end]
                        p = initial[member : member + 1, start:end]
                        padding = (-len(p[0])) % 16
                        if padding:
                            q = torch.cat((q, q[:, -1:].expand(-1, padding, -1)), dim=1)
                            p = torch.cat((p, p[:, -1:].expand(-1, padding, -1)), dim=1)
                        result = original(image[member : member + 1], q, p, *options)
                        if outputs is None:
                            outputs = [
                                torch.empty(
                                    (*initial.shape[:2], *r.shape[2:]),
                                    dtype=r.dtype,
                                    device=r.device,
                                )
                                for r in result
                            ]
                        for output, part in zip(outputs, result, strict=True):
                            output[member : member + 1, start:end].copy_(part[:, : end - start])
            return outputs

        setattr(native, name, bounded)
    # Solvers already read their normal equations and convergence back to CPU.
    # Their original iteration loop retains those necessary dependencies.
