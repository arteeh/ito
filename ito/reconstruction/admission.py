"""Display-issued, expiring GPU grants. Waiting happens only in reconstruction.

A grant never queues future GPU work. CUDA callers fence a bounded chunk before
requesting another; elapsed chunk cost increases the next reservation.
"""

from contextlib import contextmanager

from ito import clock

MAX_SPLATS = 524_288
MAX_CUDA_BYTES = 6_000_000_000
GUARD_SECONDS = 0.002
PUBLICATION_PACKETS = 2
CHUNK_OPERATIONS = 64
CHUNK_ENQUEUE_SECONDS = 0.001
MATCH_QUERIES = 4096


class Wake:
    """One pending notification, including when a receiver dies while waiting."""

    def __init__(self, context):
        self.semaphore = context.BoundedSemaphore(1)
        self.semaphore.acquire(False)

    def notify(self):
        try:
            self.semaphore.release()
        except ValueError:
            pass  # The latest notification is already pending.

    def wait(self, timeout):
        return self.semaphore.acquire(timeout=timeout)


class Admission:
    def __init__(self, context, stopped):
        self.stopped = stopped
        self.lock = context.Lock()
        self.lease = context.Lock()
        self.in_cuda_chunk = False
        self.checkpoint = None
        self.memory_used = None
        self.memory_peak = context.RawValue("Q", 0)
        self.last_progress = context.RawValue("d", 0)
        self.deadline = context.RawValue("d", 0)
        self.enabled = context.RawValue("b", False)
        self.wake = Wake(context)
        self.chunk_seconds = 0.001
        self.estimates = {}
        self.overruns = context.RawValue("Q", 0)
        self.chunks = context.RawValue("Q", 0)

    def completed(self, began, period):
        """Called after the display's GL fence, once for the complete stereo frame."""
        self.enabled.value = True
        if not self.lock.acquire(False):
            return
        try:
            # The display has finished. Only this interval's remaining time is
            # available; reserve a driver/wakeup margin before the next draw.
            self.deadline.value = began + period - GUARD_SECONDS
        finally:
            self.lock.release()
        self.wake.notify()

    def available(self, seconds=None):
        if not self.enabled.value:
            return True
        seconds = self.chunk_seconds if seconds is None else seconds
        if not self.lock.acquire(False):
            return False
        try:
            return clock.now() + seconds < self.deadline.value
        finally:
            self.lock.release()

    def acquire(self, seconds=None):
        import multiprocessing as mp

        parent = mp.parent_process()
        while not self.stopped.value and (parent is None or parent.is_alive()):
            if self.available(seconds):
                return True
            self.wake.wait(0.05)
        return False

    def begin(self, seconds=None):
        while self.acquire(seconds):
            if self.lease.acquire(False):
                if self.available(seconds):
                    return True
                self.lease.release()
            self.wake.wait(0.001)
        return False

    def observe(self, began, service_seconds=None, kind="torch"):
        elapsed = clock.now() - began if service_seconds is None else service_seconds
        if self.memory_used is not None:
            used = self.memory_used()
            self.memory_peak.value = max(used, self.memory_peak.value)
            if used > MAX_CUDA_BYTES:
                raise RuntimeError("Reconstruction exceeds its 6 GB CUDA budget")
        self.last_progress.value = clock.now()
        self.chunks.value += 1
        if self.enabled.value and clock.now() > self.deadline.value:
            self.overruns.value += 1
        # CUDA event intervals include host launch gaps as well as execution.
        # The ceiling is the traced submission budget, not a kernel guarantee;
        # chunk overruns and display misses are reported separately.
        previous = self.estimates.get(kind, 0.001)
        self.estimates[kind] = max(0.0003, min(0.003, max(elapsed * 1.25, previous * 0.9)))
        if kind == "torch":
            self.chunk_seconds = self.estimates[kind]

    @contextmanager
    def cuda_chunk(self, synchronize, events=None, *, kind="map"):
        if self.checkpoint is not None:
            self.checkpoint()
        if not self.begin(self.estimates.get(kind, 0.001)):
            raise InterruptedError("Reconstruction stopped")
        self.in_cuda_chunk = True
        try:
            synchronize()
            began = clock.now()
            if events is not None:
                start, end, elapsed = events
                start.record()
            try:
                yield
            finally:
                if events is not None:
                    end.record()
                synchronize()
                self.observe(
                    began, elapsed(start, end) / 1000 if events is not None else None, kind
                )
        finally:
            self.in_cuda_chunk = False
            self.lease.release()
            self.wake.notify()

    @contextmanager
    def torch_chunks(self):
        """Bound the CUDA submission queue, including inference and optimizer operations.

        This is cooperative admission, not kernel preemption: a single long CUDA
        operation may overrun its grant and is visible in the overrun counter.
        """
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_leaves

        admission = self

        class Chunks(TorchDispatchMode):
            operations = 0
            began = 0.0

            def __init__(self):
                super().__init__()
                self.start = torch.cuda.Event(enable_timing=True)
                self.end = torch.cuda.Event(enable_timing=True)

            def finish(self):
                if self.operations:
                    try:
                        if self.has_end:
                            self.end.synchronize()
                            service = self.start.elapsed_time(self.end) / 1000
                        else:
                            torch.cuda.synchronize()
                            service = 0
                        admission.observe(self.began, service)
                    finally:
                        self.operations = 0
                        admission.lease.release()
                        admission.wake.notify()

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                if admission.in_cuda_chunk or func._schema.name in {
                    "aten::view",
                    "aten::_unsafe_view",
                    "aten::as_strided",
                    "aten::slice",
                    "aten::select",
                    "aten::transpose",
                    "aten::t",
                    "aten::permute",
                    "aten::expand",
                    "aten::detach",
                    "aten::alias",
                    "aten::unsqueeze",
                    "aten::squeeze",
                    "aten::split",
                    "aten::split_with_sizes",
                    "aten::unbind",
                }:
                    return func(*args, **kwargs)
                cuda = any(
                    isinstance(v, torch.Tensor) and v.is_cuda for v in tree_leaves((args, kwargs))
                )
                device = kwargs.get("device")
                cuda = cuda or (device is not None and torch.device(device).type == "cuda")
                if cuda:
                    if self.operations and (
                        self.operations >= CHUNK_OPERATIONS
                        or clock.now() - self.began > CHUNK_ENQUEUE_SECONDS
                    ):
                        self.finish()
                    if not self.operations:
                        if not admission.begin():
                            raise InterruptedError("Reconstruction stopped")
                        self.began = clock.now()
                        self.has_end = False
                        self.start.record()
                    self.operations += 1
                result = func(*args, **kwargs)
                if cuda:
                    self.end.record()
                    self.has_end = True
                return result

        if not self.enabled.value:
            yield
            return
        chunks = Chunks()
        previous = self.checkpoint
        self.checkpoint = chunks.finish
        try:
            with chunks:
                yield
        finally:
            chunks.finish()
            self.checkpoint = previous


def bind_model(model, admission):
    """Use neural blocks and head convolutions as inference submission units."""
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
    for index, block in enumerate(blocks):
        scope = []

        def before(module, inputs, *, scope=scope, index=index):
            chunk = admission.cuda_chunk(
                torch.cuda.synchronize,
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                    lambda start, end: start.elapsed_time(end),
                ),
                kind=f"model-{index}",
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


def bind_matching(admission):
    """Bound upstream's native kernels, which bypass Torch's dispatch hooks.

    Each query is independent. Preserve the full descriptor/ray image, process
    one batch member and 4096 queries at a time, and retain identical outputs.
    The binding is process-local: Ito owns one reconstruction backend per worker.
    """
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
                    with admission.cuda_chunk(torch.cuda.synchronize, kind=name):
                        # Upstream launches 16-thread blocks without a tail guard.
                        # Image-sized queries are divisible by 16; pad any shorter tail.
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
