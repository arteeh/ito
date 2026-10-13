# Shared GPU budget

The reference envelope passes on the RTX 4070 Ti (12 GB): **0/72,000 missed
11.11 ms deadlines**, no consecutive misses, plus **0/6,000** at all 524,288
occupied slots. The largest conservative Ito residency bound is **6.934 GB**.
[The frozen results](gpu-budget-results.json) retain individual runs, source and
artifact hashes, kernel/transfer traces, native equivalence checks and baseline comparisons.

Later steps must respect these product limits:

- **524,288 live slots**, including slots holding fading geometry. Defaults remain
  16K for software rendering and 256K for hardware with less than 8 GiB NVIDIA VRAM.
  Sorting uses the highest resident slot rounded to a power of two; reserved empty
  tails add no sorting work. Interior holes still cost. Exact 1,024-key GPU tile sorts
  precede global merges; depth and alpha compositing remain unchanged.
- **Render first**, including both eyes and UI, then admit CUDA work until **2 ms
  before the next display frame**. Only reconstruction waits for grants. One newest
  input and one pending notification replace queues of obsolete requests. Submissions
  hold a shared lease and fence before another chunk. This is cooperative admission;
  a library kernel cannot be preempted by this policy.
- Transformer blocks and head convolutions are separate inference units. Other Torch
  work yields after **8 GPU operations or 0.25 ms of host submission**; metadata-only
  views do not queue GPU work. Native matching uses **one image / 4,096 queries** per
  fenced call, preserving the full descriptor image. Native graph solving yields after
  each unchanged iteration. Map preparation, matching and readback yield separately;
  carving uses **32K-slot batches**. Measured costs adapt the next reservation.
- **Two 4,096-record publication packets per display frame**, using four fixed staging
  slots with nonblocking GL fences. A busy upload retains its packet. Scene storage is
  reserved before the display loop and reused on restart. Integer sync addresses avoid
  cyclic ctypes ownership; automatic Python collection stays enabled.
- Reconstruction's **Torch + CuPy pools have a 6 GB allowance**; exceeding it stops
  that backend with a visible reason. The remaining **2 GB covers graphics, optimizer
  additions and driver/context residency together**. The tested optimizer unit is
  **at most 32,768 Gaussians / one 128×128 patch / forward-backward-Adam step**.
  Reconstruction and this separate optimizer alternate display grants in the stress
  run. A changed model, batch, footprint, GPU or display target must pass this envelope
  again. Counts alone do not establish a safe overdraw budget.

Windows gives the display thread MMCSS Games/high CPU priority and fine timers.
The thread's priority is reverted on exit. There is no hardware GPU reservation.
A fresh robot motion prior and recent GPU progress keep the complete scene available
while vision advances; actual loss, stale motion, no progress and faults retain fallback.
Reference runs completed **25–51 backend calls per 67.67 s** including display warm-up.
These counts include pose-only/skipped integrations and do not establish surface freshness.
Independent fast state estimation remains step 3.

Run the complete pipeline from this worktree in WSL:

```sh
bash e2e/gpu_budget_suite.sh ../m2-out/verified-nvml
```

The command checks native math, occupied count, reserved capacity, uncontrolled load,
concurrent CUDA traces, three repeats of both trajectories at both resolutions, frozen
images, and the full step-1 quality replay. Each GPU job holds `ceres.lock`, has a
1,100-second timeout and cleans up its own children. It reuses the read-only Windows
SLAM/model bundle and Linux gsplat environment; caches and bulky outputs stay in
`m2-out`. Exact completed GPU cases resume only with matching source/argument
fingerprints; capture reuse checks product, budget, script and causal artifacts.
No driver/protocol fields or wire version changed.

The trace uses recorded **pilot head matrices**, interpolated at 90 Hz and replayed
forward/backward. The robot camera position anchors the frozen scene, as in the product;
both eyes share a midpoint sort at configured **64 mm IPD**. Targets are **360×640
RGBA32F** and **2160×2160 RGBA8, 90° FOV**, per eye. The workloads are production
`SLAMBackend` integrations and actual gsplat forward/backward/Adam. Three integrations
and ten optimizer steps warm resources for sweep; tour tracking warms through source
frame 1,560 (261 integrations), exposing later keyframes and graph matching.
The Linux training bridge carries expiring grants using precise Windows UTC and a
bounded pipe. It is measurement plumbing.

Completion is a GL fence. Wake-up lateness is charged to absolute 11.11 ms deadlines;
no missed samples are removed. The following percentiles are **scheduled-start to
completion**, with the maximum render/publication service time reported separately:

| Per-eye target / load | Frames | Misses / consecutive | p99 ms | p99.9 ms | Max completion ms | Max service ms | Residency bound GB |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 360×640, both trajectories × 3 | 36,000 | 0 / 0 | 6.197 | 6.957 | 9.034 | 8.771 | 6.773 |
| 2160×2160, both trajectories × 3 | 36,000 | 0 / 0 | 6.839 | 7.905 | 8.740 | 8.495 | 6.934 |
| 2160×2160, 524,288 occupied slots | 6,000 | 0 / 0 | 7.035 | 8.087 | 10.456 | 10.243 | 6.934 |

The 131K and 262K occupied shared-load probes also have zero misses (900 frames each).
All twelve reference runs individually pass **<0.1%** misses, no consecutive misses,
residency and actual admitted-work gates. Uncontrolled tracking + training misses
**38/1,800 (2.11%)**, with **nine consecutive misses** and **15.09 ms** worst completion.
GPU memory uses a persistent [NVML connection](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlInitializationAndCleanup.html)
and WDDM counters for owned Windows PIDs. Linux CUDA is absent from those PID counters.
The bound uses maximum whole-device usage minus minimum sampled DWM dedication on
that GPU, including Linux, other apps and driver memory. Sampling and Windows counter
semantics remain accounting limits; the report also retains CUDA pool peaks.

CUPTI's concurrent kernel/memcpy records use hardware timestamps without serializing
kernels; the binding checks CUDA-12.4's ABI. Neither final trace dropped records.
The selected late integration's longest kernel falls from **19.19 ms** descriptor
matching to **1.389 ms** with admission. A longer earlier admitted diagnostic reached
**2.36 ms**. Native CUDA fuzzing verifies **24 exact matching comparisons** and **four
real graph solves**, including identical poses and increments. The admitted map's
CuPy upload/readback calls total **1.376 ms**; all recorded device copies total **3.008 ms**
for that integration. Placement/carving host scopes include grant waiting. These transfers
do not justify CUDA/GL interop.

Earlier cohorts failed and are retained. Per-packet store orphaning was replaced with
fixed fenced staging. A host trace then found **12.23 ms generation-2 collection inside
`glFenceSync`**, causing consecutive misses. PyOpenGL's opaque pointer conversion adds
an ownership cycle per fence: 2,048 real upload/poll/delete cycles reproduce **4,096
cyclic objects** with the old representation and **zero** with integer addresses.
The fixed code still caught a **21.92 ms / two-miss** burst. The monitor had been starting
`nvidia-smi` and initializing/shutting down NVML every sample. Keeping NVML attached
removes those operations from timed frames; final queries typically take **0.02 ms**.
An 18,000-frame persistent-monitor diagnostic had two isolated misses and no consecutive
misses before the full reference cohort was rerun. This change does not establish a
universal cause for every earlier driver stall. Original CPU-co-load count probes are
retained separately; clean replacements and all acceptance runs have no concurrent
baseline evaluation. Deadline thresholds and sample counting are unchanged.

All three fresh 524K-cap baseline repeats match the fixed step-1 **repeat-1 reference**:
**105/105 PNGs byte-identical**, with zero PSNR/SSIM/LPIPS/coverage/holes changes. That
same fixed reference supplies the frozen shared-load maps. Corresponding-repeat
comparisons are also retained: **100/105** byte-identical. The old third tour repeat has
one extra Gaussian (477,968 versus 477,967); its five final-view differences affect
**12–26 pixels by one channel level**, at most **0.0000061 dB** PSNR and **0.00000090**
LPIPS, with unchanged coverage/holes. The quality gate uses the fixed reference and
keeps this existing baseline variation visible.

| Camera view | PSNR dB | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| sweep 215 | 20.001 | .7895 | .4646 |
| sweep 453 | 17.254 | .7174 | .5354 |
| hard 526 | 17.010 | .7768 | .5451 |
| sweep 693 | 14.799 | .7028 | .5578 |
| tour 363 | 20.835 | .8042 | .4364 |
| tour 1501 | 22.335 | .7785 | .4789 |
| tour 2284 | 15.515 | .7895 | .6028 |

The step-1 isolated stereo rerun has **3.13–3.39 ms p50, 0/6,000 misses**, versus the
original **3.4–3.7 ms**. Its unpaced completion timing differs from the paced shared-load
reference above. Existing e2e checks pass: live app, renderer, reconstruction, faults,
stream, splat quality, frame joining and WebRTC. The live software-rendered app rerun
measured 9.65 ms median pilot input→robot and exercised e-stop, worker stall, reconnect,
settings and focus-loss safety.

Look at [tour under load](gpu-budget-img/frozen-tour.jpg),
[sweep under load](gpu-budget-img/frozen-sweep.jpg),
[baseline tour](gpu-budget-img/baseline-tour.jpg),
[baseline sweep](gpu-budget-img/baseline-sweep.jpg),
[near-wall stereo](gpu-budget-img/baseline-stereo.jpg) and
[hard 526](gpu-budget-img/baseline-hard.jpg).
I inspected the sheets: before/under-load eye pairs match. Books and poster edges remain
soft, the floor remains stretched, and the near-wall geometry/stereo mismatch persists.
Unmapped black borders appear identically in both halves; training introduces no new
black frame or shimmer. Full RGB-byte comparisons cover **990 stereo frames per
trajectory**, including the first 90 frames after starting warmed workloads. Readback
runs are excluded from deadline acceptance.

**Open:** physical headset/runtime/compositor contention, actual IPD/scanout and cold
CUDA context/model startup. The measured budget covers these warmed recorded maps,
viewpoints and workloads; later steps must repeat it when their footprint changes.
