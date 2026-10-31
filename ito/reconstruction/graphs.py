"""MASt3R's blocks replayed as CUDA graphs, one graph per admission chunk.

Eager, a tracked frame launches a few thousand small kernels from Python, and on Windows each
launch costs the CPU ~8 us (WDDM): the worker's thread spent about as long launching MASt3R as
the GPU spent running it, and the GPU idled between launches. A graph launches a block's
kernels in one call. The units are the ones admission already splits MASt3R into (encoder and
decoder blocks, the heads' stages), so the renderer can still win the GPU between them.

Each unit is captured once per input shape (in practice one: every frame has Ito's input size),
then replayed: inputs are copied into its fixed buffers, outputs cloned out of them. The
kernels are the ones the eager forward launches, so the outputs are the same. All units share
one memory pool, so only one replay may run at a time: replays are serialised across threads
(the frontend and upstream's backend thread both run MASt3R) and each one's stream waits for
the one before on the GPU. A unit that cannot be captured runs eagerly from then on.
"""

import logging
import threading

log = logging.getLogger(__name__)


def units(model):
    """MASt3R's units: the admission chunks, each also a graph: its encoder and decoder blocks,
    and the heads' stages (DPT's per-layer projections, its four fusion blocks and output
    convs, the local-feature MLP)."""
    blocks = [
        block
        for name in ("enc_blocks", "dec_blocks", "dec_blocks2")
        for block in getattr(model, name)
    ]
    for head in (model.downstream_head1, model.downstream_head2):
        dpt = head.dpt
        blocks.extend(dpt.act_postprocess)
        blocks.extend(getattr(dpt.scratch, f"refinenet{i}") for i in (4, 3, 2, 1))
        blocks.extend((dpt.head, head.head_local_features))
    return blocks


def graph_model(model):
    """Replay every unit of the model as CUDA graphs (captured at first use)."""
    model.graphs = Graphs()
    for unit in units(model):
        model.graphs.install(unit)
    return model.graphs


class Graphs:
    def __init__(self):
        import torch

        self.torch = torch
        self.lock = threading.Lock()
        self.pool = torch.cuda.graph_pool_handle()
        self.done = torch.cuda.Event()
        self.side = torch.cuda.Stream()  # warm-ups and captures, one stream for all units
        self.captured = 0
        self.failed = 0
        self.off = False  # set when a replay's fence failed: every unit runs eagerly

    def install(self, module):
        """Replace module.forward (tensor arguments only) with graph replays."""
        module.forward = Unit(self, module.forward, type(module).__name__)


class Unit:
    def __init__(self, graphs, forward, name):
        self.graphs, self.forward, self.name = graphs, forward, name
        self.entries = {}
        self.eager = False

    def __call__(self, *inputs):
        g = self.graphs
        torch = g.torch
        if (
            self.eager
            or g.off
            or torch.is_grad_enabled()
            or torch.cuda.is_current_stream_capturing()
            or not all(isinstance(t, torch.Tensor) and t.is_cuda for t in inputs)
        ):
            return self.forward(*inputs)
        key = (
            tuple((tuple(t.shape), t.dtype) for t in inputs),
            torch.is_inference_mode_enabled(),
            torch.is_autocast_enabled("cuda"),
        )
        with g.lock:
            if g.off:
                return self.forward(*inputs)
            entry = self.entries.get(key)
            if entry is None:
                entry = self.capture(key, inputs)
                if entry is None:
                    return self.forward(*inputs)
            buffers, graph, outputs, aliases, single = entry
            stream = torch.cuda.current_stream()
            stream.wait_event(g.done)
            # From here this stream may use the shared pool: whatever happens (an OOM cloning an
            # output), done is recorded after it, or the next replay on the other thread's
            # stream could run over this one's buffers.
            try:
                for buffer, value in zip(buffers, inputs, strict=True):
                    buffer.copy_(value)
                graph.replay()
                # An output that is one of the inputs (a decoder block returns its memory as it
                # got it) is handed back as the caller's own tensor.
                result = tuple(
                    inputs[alias] if alias is not None else output.clone()
                    for output, alias in zip(outputs, aliases, strict=True)
                )
            finally:
                try:
                    g.done.record(stream)
                except BaseException:
                    # No fence for the pool: no replay is safe after this one; eager from now.
                    g.off = True
                    log.error("MASt3R's CUDA graphs are off: their fence failed")
                    raise
        return result[0] if single else result

    def capture(self, key, inputs):
        g = self.graphs
        torch = g.torch
        buffers = [t.clone() for t in inputs]
        caller = torch.cuda.current_stream()
        side = g.side
        side.wait_stream(caller)
        graph = outputs = None
        try:
            # Warm-up outside the capture, as CUDA graphs need (cuBLAS/cuDNN pick their kernels).
            with torch.cuda.stream(side):
                for _ in range(2):
                    self.forward(*buffers)
            caller.wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=g.pool, capture_error_mode="thread_local"):
                outputs = self.forward(*buffers)
        except torch.cuda.OutOfMemoryError:
            # Not a unit that cannot be captured: the VRAM is short, and the caller reports it
            # (at setup the clear "Out of VRAM ... lower ..." error). Same precision either way.
            del graph, outputs, buffers
            self.eager = True
            g.failed += 1
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            raise
        except RuntimeError as exc:
            log.warning("MASt3R %s runs without a CUDA graph: %s", self.name, str(exc)[:300])
            self.eager = True
            g.failed += 1
            torch.cuda.synchronize()
            return None
        finally:
            # torch.cuda.graph restores the stream only after capture_end(), which raises when
            # the capture failed.
            torch.cuda.set_stream(caller)
        single = isinstance(outputs, torch.Tensor)
        outputs = (outputs,) if single else tuple(outputs)
        ids = {id(b): i for i, b in enumerate(buffers)}
        aliases = tuple(ids.get(id(o)) for o in outputs)
        entry = (buffers, graph, outputs, aliases, single)
        self.entries[key] = entry
        g.captured += 1
        return entry
