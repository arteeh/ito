import time
from collections import deque
from contextlib import contextmanager

from ito import clock

PUBLICATION_PACKETS = 2
MATCH_QUERIES = 4096
MAX_PENDING_CHUNKS = 4
MAX_BACKOFF_FRAME_SHARE = 0.25
GPU_STAGES = (
    "inference",
    "iter_proj",
    "refine_matches",
    "project",
    "voxelize",
    "match",
    "readback",
    "carve",
    "graph",
    "fit_forward",
    "fit_backward",
    "fit_readback",
)


class Wake:
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
        self.gpu_busy = context.RawValue("d", -1)
        self.memory_pressure = context.RawValue("b", False)
        self.active_seconds = context.RawValue("d", 0)
        self.chunks = context.RawValue("Q", 0)
        self.stage_seconds = context.RawArray("d", len(GPU_STAGES))
        self.stage_calls = context.RawArray("Q", len(GPU_STAGES))
        self.overruns = context.RawValue("Q", 0)
        self.wake = Wake(context)
        self.memory_used = None
        self.pending = deque()
        self.cooldown = 0.0
        self.recovered_at = 0.0
        self.stream = self.torch_stream = None

    def completed(self, began, period):
        now = clock.now()
        self.enabled.value = True
        if now - began > period:
            self.cooldown = min(period * MAX_BACKOFF_FRAME_SHARE, max(0.002, self.cooldown * 2))
            self.recovered_at = now + 0.25
            self.overruns.value += 1
        elif now >= self.recovered_at:
            self.cooldown = max(0.0, self.cooldown - 0.001)
        self.pause_until.value = now + self.cooldown
        self.wake.notify()

    def available(self):
        return clock.now() >= self.pause_until.value

    def acquire(self):
        import multiprocessing as mp

        parent = mp.parent_process()
        while not self.stopped.value and (parent is None or parent.is_alive()):
            if self.available():
                return True
            # Sleep, not a semaphore timeout: on Windows a 1 ms semaphore wait
            # takes ~10 ms (timer tick), which turns a 2 ms render backoff into 10.
            time.sleep(min(0.002, max(0.0005, self.pause_until.value - clock.now())))
        return False

    def bind_stream(self, xp):
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

    def collect(self):
        while self.pending and event_done(self.pending[0][1]):
            start, end, elapsed, kind = self.pending.popleft()
            seconds = elapsed(start, end) / 1000
            self.active_seconds.value += seconds
            if kind in GPU_STAGES:
                index = GPU_STAGES.index(kind)
                self.stage_seconds[index] += seconds
                self.stage_calls[index] += 1
            self.chunks.value += 1
            self.last_progress.value = clock.now()

    def checkpoint(self):
        self.collect()
        # Queue a few units ahead to hide launch overhead, without an entire model
        # already enqueued when a render miss asks us to back off. Query, don't fence.
        # When full, wait on the oldest chunk itself (about a millisecond of GPU work)
        # rather than polling: a polled 1 ms wait sleeps ~10 ms on Windows, leaving
        # the GPU idle for reconstruction most of the time.
        while len(self.pending) >= MAX_PENDING_CHUNKS:
            if not self.acquire():
                raise InterruptedError("Reconstruction stopped")
            self.pending[0][1].synchronize()
            self.collect()
        if not self.acquire():
            raise InterruptedError("Reconstruction stopped")
        self.check_memory()

    @contextmanager
    def cuda_chunk(self, events, *, kind="map"):
        self.checkpoint()
        start, end, elapsed = events
        start.record()
        try:
            yield
        finally:
            end.record()
            self.pending.append((start, end, elapsed, kind))
            self.check_memory()

    @contextmanager
    def integration(self):
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


def event_done(event):
    query = getattr(event, "query", None)
    return query() if query is not None else event.done


def bind_model(model, admission):
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
                ),
                kind="inference",
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
    import mast3r_slam_backends as native
    import torch

    for name in ("iter_proj", "refine_matches"):
        original = getattr(native, name)

        def bounded(image, queries, initial, *options, original=original, name=name):
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
                        ),
                        kind=name,
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
    for name in ("gauss_newton_points", "gauss_newton_rays", "gauss_newton_calib"):
        original = getattr(native, name)

        def solve(*args, original=original):
            with admission.cuda_chunk(
                events=(
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                    lambda start, end: start.elapsed_time(end),
                ),
                kind="graph",
            ):
                return original(*args)

        setattr(native, name, solve)
