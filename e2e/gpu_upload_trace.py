"""Diagnose GL upload stalls with per-call host timing under the shared-load runner.

ITO_GPU_ENTRY=e2e/gpu_upload_trace.py bash e2e/gpu_budget.sh OUTPUT --load both \
    --admit --incremental --updates --frames 6000
Diagnostic timings are excluded from reference acceptance.
"""

import json
import os
import time
from copy import copy
from pathlib import Path

import gpu_budget
import moderngl
from OpenGL import GL

from ito.render.renderer import GaussianRenderer

packets = []
display_calls = []
current = None
original_apply = GaussianRenderer.apply


def apply(renderer, update):
    global current
    current = dict(start=time.perf_counter(), calls=[])
    try:
        return original_apply(renderer, update)
    finally:
        current["total_ms"] = (time.perf_counter() - current["start"]) * 1000
        if current["total_ms"] > 1:
            packets.append(current)
        current = None


def timed(owner, name):
    original = getattr(owner, name)

    def call(*args, **kwargs):
        if current is None:
            return original(*args, **kwargs)
        began = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            current["calls"].append(
                dict(call=name, start=began, ms=(time.perf_counter() - began) * 1000)
            )

    setattr(owner, name, call)


GaussianRenderer.apply = apply
for owner, names in (
    (moderngl.Buffer, ("write", "bind_to_storage_buffer")),
    (moderngl.ComputeShader, ("run",)),
    (GL, ("glClientWaitSync", "glMemoryBarrier", "glFenceSync")),
):
    for name in names:
        timed(owner, name)


def display_call(owner, name):
    original = getattr(owner, name)

    def call(*args, **kwargs):
        began, cpu = time.perf_counter(), time.thread_time()
        try:
            return original(*args, **kwargs)
        finally:
            elapsed = (time.perf_counter() - began) * 1000
            if elapsed > 2:
                display_calls.append(
                    dict(
                        call=name, start=began, ms=elapsed, cpu_ms=(time.thread_time() - cpu) * 1000
                    )
                )

    setattr(owner, name, call)


for owner, name in (
    (GaussianRenderer, "sort"),
    (GaussianRenderer, "draw"),
    (moderngl.Context, "finish"),
):
    display_call(owner, name)


original_worker = gpu_budget.load_worker


def profiled_worker(args, *events):
    from gpu_cupti import Cupti

    kernels = bool(os.environ.get("ITO_GPU_TRACE_KERNELS"))
    trace = Cupti(apis=not kernels, allocations_only=not kernels)
    args = copy(args)
    args.profile = False
    with trace:
        original_worker(args, *events)
    (args.out / "cuda-apis.json").write_text(json.dumps(trace.report(), indent=2))


if __name__ == "__main__":
    import sys

    if "--kernels" in sys.argv:
        sys.argv.remove("--kernels")
        os.environ["ITO_GPU_TRACE_KERNELS"] = "1"
        gpu_budget.load_worker = profiled_worker
    if "--apis" in sys.argv:
        sys.argv.remove("--apis")
        gpu_budget.load_worker = profiled_worker
    output = Path(sys.argv[sys.argv.index("--out") + 1])
    try:
        gpu_budget.main()
    finally:
        (output / "upload-calls.json").write_text(json.dumps(packets, indent=2))
        (output / "display-calls.json").write_text(json.dumps(display_calls, indent=2))
