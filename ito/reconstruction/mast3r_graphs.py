"""MASt3R's network replayed as CUDA graphs (ITO_MAST3R_GRAPHS=0 turns it off).

Eager, the encoder, decoder and heads launch a few thousand small kernels per frame from Python,
and the GPU waits on the launches: a live profile had CUDA busy only 23-43% of the time, in
hundreds of thousands of gaps under a millisecond. A graph launches all of a unit's kernels in
one call. Each unit (the encoder, the decoder, each head) is captured once per input shape and
then replayed: inputs are copied into its fixed buffers, outputs cloned out of them.

All graphs share one memory pool, so only one may run at a time: replays are serialised across
threads (the worker and full SLAM's graph thread both run MASt3R) and each waits on the GPU for
the one before. A unit that cannot be captured (a host sync inside it) runs eagerly from then on.
"""

import logging
import os
import threading

log = logging.getLogger(__name__)

ENABLED = os.environ.get("ITO_MAST3R_GRAPHS", "1") != "0"
MAX_SHAPES = 6  # per unit; beyond that (an unusual caller) the unit runs eagerly


def tensors(value, function):
    import torch

    if isinstance(value, torch.Tensor):
        return function(value)
    if isinstance(value, dict):
        return {key: tensors(item, function) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(tensors(item, function) for item in value)
    return value


def flat(value, into):
    import torch

    if isinstance(value, torch.Tensor):
        into.append(value)
    elif isinstance(value, (tuple, list)):
        for item in value:
            flat(item, into)
    return into


class Graphs:
    """The shared pool, lock and ordering of every captured unit of one model."""

    def __init__(self, admission):
        import torch

        self.torch = torch
        self.admission = admission
        self.lock = threading.Lock()
        self.pool = torch.cuda.graph_pool_handle()
        self.done = torch.cuda.Event()
        self.units = []
        self.captured = 0
        self.failed = 0

    def unit(self, function, name):
        unit = Unit(self, function, name)
        self.units.append(unit)
        return unit

    def chunk(self):
        from contextlib import nullcontext

        torch = self.torch
        if self.admission is None:
            return nullcontext()
        return self.admission.cuda_chunk(
            events=(
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
                lambda start, end: start.elapsed_time(end),
            ),
            kind="inference",
        )


class Unit:
    def __init__(self, graphs, function, name):
        self.graphs, self.function, self.name = graphs, function, name
        self.entries = {}
        self.eager = False

    def __call__(self, *arguments, **static):
        """arguments: tensors or (nested) lists of tensors, the graph's inputs; static: values
        the graph is captured for (part of the key, passed to every call as they are)."""
        g = self.graphs
        torch = g.torch
        inputs = flat(arguments, [])
        if self.eager or torch.cuda.is_current_stream_capturing():
            return self.function(*arguments, **static)
        key = (
            tuple((tuple(t.shape), t.dtype, t.device) for t in inputs),
            tuple(sorted(static.items())),
            torch.is_autocast_enabled("cuda"),
            torch.is_inference_mode_enabled(),
        )
        with g.lock:
            entry = self.entries.get(key)
            if entry is None:
                if len(self.entries) >= MAX_SHAPES:
                    return self.function(*arguments, **static)
                entry = self.capture(key, arguments, static)
                if entry is None:
                    return self.function(*arguments, **static)
            buffers, graph, outputs = entry
            stream = torch.cuda.current_stream()
            with g.chunk():
                stream.wait_event(g.done)
                for buffer, value in zip(buffers, inputs, strict=True):
                    buffer.copy_(value)
                graph.replay()
                result = tensors(outputs, lambda t: t.clone())
                g.done.record(stream)
            return result

    def capture(self, key, arguments, static):
        g = self.graphs
        torch = g.torch
        copies = tensors(arguments, lambda t: t.clone())
        buffers = flat(copies, [])
        admission = g.admission
        caller = torch.cuda.current_stream()
        side = torch.cuda.Stream()
        side.wait_stream(caller)
        if admission is not None:
            admission.capturing = threading.get_ident()
        try:
            # Warm-up outside the capture, as CUDA graphs need (cuBLAS/cuDNN pick their kernels).
            with torch.cuda.stream(side):
                for _ in range(2):
                    self.function(*copies, **static)
            caller.wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=g.pool, capture_error_mode="thread_local"):
                outputs = self.function(*copies, **static)
        except RuntimeError as exc:
            log.warning("MASt3R %s runs without a CUDA graph: %s", self.name, str(exc)[:300])
            self.eager = True
            g.failed += 1
            torch.cuda.synchronize()
            return None
        finally:
            if admission is not None:
                admission.capturing = None
        entry = (buffers, graph, outputs)
        self.entries[key] = entry
        g.captured += 1
        log.info("MASt3R %s captured as a CUDA graph (%d)", self.name, len(self.entries))
        return entry


def graph_model(model, admission=None):
    """Replace model._encode_image, _decoder and _downstream_head (as precision may have
    wrapped them) with graph replays. Returns the Graphs, or None when disabled."""
    if not ENABLED:
        return None
    import torch

    graphs = Graphs(admission)
    encode = graphs.unit(model._encode_image, "encoder")
    decode = graphs.unit(lambda *a: tuple(tuple(side) for side in model_decoder(*a)), "decoder")
    model_decoder = model._decoder
    head_function = model._downstream_head
    heads = graphs.unit(
        lambda decout, head, shape: head_function(head, decout, torch.tensor([shape])), "head"
    )
    shapes = {}  # tokens per image -> (height, width): one host read per size, not per call

    def encode_image(image, true_shape):
        return encode(image, true_shape)

    def decoder(f1, pos1, f2, pos2):
        return decode(f1, pos1, f2, pos2)

    def downstream_head(head, decout, img_shape):
        tokens = decout[-1].shape[1]
        shape = shapes.get(tokens)
        if shape is None:
            shape = shapes[tokens] = tuple(int(v) for v in img_shape[0].tolist())
        return heads(list(decout), head=head, shape=shape)

    model._encode_image = encode_image
    model._decoder = decoder
    model._downstream_head = downstream_head
    return graphs
