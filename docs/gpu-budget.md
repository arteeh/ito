# Shared GPU budget

Reconstruction runs concurrently with GL on one nonblocking CUDA stream at CUDA's
lowest scheduling priority (0). Torch, CuPy and every shipped native launch use that
stream. Model blocks and 4,096-query matching tiles record asynchronous events; at
most four units are queued ahead. There is no display-gap grant or per-unit fence.
CPU readbacks and the integration/publication boundary retain their required waits.
A completed GL frame over its display budget triggers a reconstruction cooldown of
at most a quarter-period; clean frames release it. Persistent lateness cannot renew
a pause longer than the frame period and starve reconstruction. CUDA priority cannot preempt an executing kernel or reserve
resources against GL, WDDM or a headset compositor.

At startup, dedicated free VRAM sizes scene storage, subject to GL buffer limits:
1/32 of free memory after **`GPU_HEADROOM_BYTES = 1,500,000,000`**, accounting for
records and sort storage. Software rendering retains its small scene. Saved/manual
budgets may choose fewer slots. After render allocation and model loading, remaining
free VRAM sizes Torch/CuPy caching pools (80/20 for SLAM). CUDA free-memory readings
are bounded by NVML's dedicated free memory when available. Pool limits prevent Ito's
allocators from spilling into WDDM shared memory; external allocations after startup
can still consume that allowance. Unused caches are released under memory pressure.
There is no fixed 6 GB or 524,288-slot ceiling.

Slot pressure fades the oldest observations, reserves space for incoming detail and
prioritizes eviction packets so acknowledgements release victims. Pressure victims
cannot be rescued indefinitely; a view touching the entire pool also yields its least
recently seen slots when new detail needs space. Four fenced GL staging buffers,
two publication packets per display frame, exact tiled sorting and integer GL fence
addresses remain. Publication bandwidth can limit freshness independently of CUDA;
logs expose dirty-slot count and the oldest pending capture age.

The overlay and motion-sheet headers show only visible reconstruction frames/s,
latest capture-to-visible ms, whole-device GPU busy % (NVML, otherwise unavailable),
and occupied slot %. A frame counts once its first changed packet completes on GL,
with its actual source timestamp; pose-only calls and evictions do not count. This
measures GL completion, not physical scanout or complete publication of every slot.
`ITO_TIMELINE` adds `backend_frame`, `display_statistics` and `run_summary` events.
`--metrics` keeps frame JSONL and writes `<stem>.summary.json`; `ITO_RECORD` retains
snapshots in `pilot/frames.jsonl` and the full summary in `summary.json`. Logs include
latency p50/p95, render p99/deadline misses, matched %, input age/skips, CUDA scope
calls/time, pool limits/peaks and cooldown state. Rolling percentiles cover 10 seconds;
scene rate covers 5 seconds. Whole-run histograms report their bin precision. CUDA
scope intervals include host launch gaps and contention; they are not GPU utilization.
Desktop deadlines include wake lateness; XR completion timing starts after runtime
frame wait and excludes compositor/scanout timing. Recording readback affects timing.

**Unverified on GPU.** Rebuild the version-5 SLAM bundle with `python -m ito.build`
in the CUDA build environment; version-4 default-stream kernels are refused. The
[#39 results](gpu-budget-results.json) describe the old serialized policy and do not
validate this design. No fixed reconstruction-rate/latency target is claimed: use all
GPU capacity rendering leaves available. The owner must compare fast sweeps, a long
tour and slot-pressure runs with main, including motion sheets, render deadlines,
dedicated/shared VRAM and whether new detail continues arriving at capacity.
