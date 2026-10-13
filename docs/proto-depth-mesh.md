# Depth-mesh world — #47

13 October 2026, separate `proto/depth-mesh` branch from main `07ca8a3`.

**Verdict: useful sharpness experiment, reject as an “always a room, never black” layer.**
The textures are clearly sharper than frozen main. Robot-pose seams split furniture and
posters, changing depth makes the floor into overlapping strips, and a wide view still
contains large black regions. Comfort takes precedence over readable but moving labels.
The owner's preference is pending; this is my eye check, not headset acceptance.

## Look

Each row is **sim truth | frozen main | mesh**, at the same projection and true evaluation
pose. Start with [robot-pose sweep](proto-depth-mesh/img/mono-sweep.jpg),
[robot-pose tour](proto-depth-mesh/img/mono-tour.jpg), and
[wide turns/translations](proto-depth-mesh/img/wide-offsets.jpg).
Then compare [oracle-pose sweep](proto-depth-mesh/img/oracle-sweep.jpg),
[oracle-pose tour](proto-depth-mesh/img/oracle-tour.jpg),
[near-pot eyes](proto-depth-mesh/img/pot-oracle-eyes.jpg),
[near-wall eyes](proto-depth-mesh/img/mono-wall-eyes.jpg),
[ideal ToF + oracle pose](proto-depth-mesh/img/tof-oracle-tour.jpg), and
[hard turn](proto-depth-mesh/img/hard-turn.jpg).

I opened these comparisons and the moving-head/map-update filmstrips. Book labels and
floor grain survive in the mesh where main smears them. With oracle pose, the poster
and shelf outlines become much cleaner, although thin cracks and image-border holes
remain. With robot pose, sweep's shelf is cut vertically through the box and books;
tour's poster is split horizontally. The two near-pot eyes expose displaced foreground
and missing background even with oracle pose. At wide FOV the floor is a fan of separate
textured sheets. Translating 30 cm opens gaps and stretches those strips. In the observed
bookshelf filmstrip, a split shelf suddenly becomes whole at the next patch replacement.
These are visible failures, not merely a lower PSNR.

Six two-second clips, their 3× slow versions, traces and filmstrips are in
`work/p47-out/clips/metric-small-robot`. They follow the baseline's frozen-map/moving-head,
fixed-head/map-updates, and combined trajectories, including the already observed bookshelf.

## Method and comparison contract

`ito/proto/depthmesh` accepts RGB, axial depth in metres, intrinsics and a camera pose;
it contains no Microduck or sensor-specific code. OpenGL unprojects a **90×160 portrait
grid**, with projective texture coordinates preserving the original **360×640** RGB. A geometry shader rejects
triangles with a depth span greater than **2 cm + 8% of nearest depth**. A compute shader
permanently masks older surfaces in a new view's free space or overlap, with
**2 cm + 3%** tolerance. Occluded background survives. Oldest-first eviction caps the
world at **64 patches**. Rendering and retirement stay on the GPU; readback is evaluation
or file export only. Main's default pipeline is unchanged.

The frozen baseline admits every sixth frame, about 5 Hz. All seven moments have camera,
64 mm left/right eye, and ±3 cm views. Target RGB never enters its own snapshot. A virtual
publication clock charges measured isolated CUDA inference plus a declared **10 ms**
mesh-update/publication budget. This budget exceeds the measured update costs, but is
not a measurement of the full live pipeline. Model startup is outside that clock; first
inference is charged. Scores include black holes. Common-seen masks come unchanged from
`work/m1-out/verified/repeat-1`, independently of the mesh and its cap.

Oracle pose/depth are separate interventions. Robot pose uses only motion samples
available at admission, interpolated at the driver's fixed **90 ms** capture delay, with
one first-frame heading/position alignment for evaluation. No visual tracking or later
pose corrections enter it. Rotation p95 is **4.58° sweep / 3.39° tour**, translation p95
**6 / 28 mm**; the rotation alone corresponds to about **24 / 18 cm at 3 m**.

Legacy truth remains thumbnail-matched; the old intrinsics have a one-pixel principal-point
mismatch. We inherit the baseline's centred frustum and static room, not a new exact-ID
physical-camera claim. Learned inference uses fp16 and a 392-pixel short side. Relative
DA/VDA outputs get one fixed startup inverse-depth gauge: first-frame p5/p95 mean 4/.4 m.
That is an arbitrary declared prior, **not recovered metric scale** or a truth fit.
PromptDA alone receives ideal registered 8×8 zone medians, capped at 4 m; missing zones
use nearest valid-zone extrapolation. This omits real ToF FOV, extrinsics, noise and timing.

## Results

Mean **PSNR dB / SSIM / AlexNet LPIPS**, six held-out eye images per sequence;
frame 526 is kept separate in the [full results](proto-depth-mesh/results.json).
The file also retains all 490 view scores, coverage, seen holes, depth diagnostics,
model/weight fingerprints, three GPU repetitions and the prefix audit.

| Depth | Pose | Sweep | Tour |
|---|---|---|---|
| Main, frozen fused | baseline | 17.33 / 0.734 / 0.524 | 18.92 / 0.778 / 0.513 |
| Sim depth | oracle | 19.74 / 0.836 / 0.245 | 23.33 / 0.878 / 0.271 |
| Sim depth | robot | 17.54 / 0.711 / 0.385 | 18.00 / 0.754 / 0.338 |
| DA V2 Small | oracle | 18.96 / 0.770 / 0.246 | 21.80 / 0.811 / 0.266 |
| DA V2 Small | robot | 17.53 / 0.689 / 0.369 | 18.14 / 0.735 / 0.342 |
| DA V2 Large | oracle | 18.10 / 0.757 / 0.269 | 21.38 / 0.805 / 0.285 |
| DA V2 Large | robot | 18.00 / 0.727 / 0.315 | 18.04 / 0.729 / 0.356 |
| Metric DA Small | oracle | 19.56 / 0.779 / 0.278 | 22.22 / 0.816 / 0.281 |
| Metric DA Small | robot | 17.54 / 0.696 / 0.393 | 18.40 / 0.737 / 0.354 |
| Metric DA Large | oracle | 18.47 / 0.743 / 0.313 | 22.00 / 0.818 / 0.280 |
| Metric DA Large | robot | 18.01 / 0.712 / 0.351 | 18.27 / 0.742 / 0.344 |
| Video DA Small, causal | oracle | 19.50 / 0.787 / 0.252 | 21.89 / 0.823 / 0.267 |
| Video DA Small, causal | robot | 17.53 / 0.694 / 0.385 | 18.06 / 0.733 / 0.344 |
| PromptDA Small, ideal ToF | oracle | 20.01 / 0.831 / 0.244 | 23.24 / 0.870 / 0.266 |
| PromptDA Small, ideal ToF | robot | 17.55 / 0.710 / 0.386 | 17.98 / 0.753 / 0.337 |

**Best geometric upper bound:** PromptDA Small with ideal ToF and oracle pose, close to
true-depth appearance. **Best mono temporal candidate:** Video DA Small. On tour, its
same-surface p95 absolute log-depth change is **.177**, versus **.374** DA Small and
**.389** Metric Small: about 19%, 45%, and 48% multiplicative changes between 5 Hz frames.
This removes actual camera depth change using truth only in the diagnostic. Video DA
still has a median predicted/true depth ratio of **1.25** and an arbitrary metric gauge.
Metric Small is a cheap metric baseline, but its tour median scale ratio is **1.62**,
p95 **6.35**; “metric” does not mean calibrated on this low camera and simulated room.
Larger models do not cure the geometry. The official VDA streaming mode is untrained
and documented as less accurate than its offline mode; no future frames are used here.

Metric Small + robot pose has **95.31% / 96.09%** mean eye coverage and
**0.50% / 3.04%** holes in previously seen sweep/tour space. Main has
**95.16% / 98.78%** coverage and **0.54% / 0.00%** seen holes. Sharpness does not compensate
for tour's coverage regression. At the hard turn, mesh camera PSNR is **14.32 dB**, main
**17.01 dB**. Wide-angle coverage and off-axis scores are separately labelled; they have
no common-seen mask. Lighting is baked into RGB, with no relighting/exposure correction;
patch lighting differences survive as floor bands.

## Cost and licences

| Model | Tour inference p50 / p95 ms, isolated | Peak Torch GiB | Weight terms |
|---|---:|---:|---|
| DA Small | 21.68 / 22.86 | .33 | Apache-2.0 |
| DA Large | 79.30 / 80.71 | 2.50 | CC BY-NC 4.0 |
| Metric Small | 21.49 / 22.53 | .33 | Apache-2.0 model card |
| Metric Large | 79.46 / 81.43 | 2.50 | Apache-2.0 card; upstream Large NC ambiguity |
| Video DA Small | 26.03 / 36.86 | .91 | Apache-2.0 |
| PromptDA Small | 16.58 / 30.86 | .33 | Apache-2.0 |

Checked [DA V2's terms](https://github.com/DepthAnything/Depth-Anything-V2#license),
[Metric Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small),
[Metric Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large),
[Video DA's terms and streaming caveat](https://github.com/DepthAnything/Video-Depth-Anything),
and [PromptDA Small](https://huggingface.co/depth-anything/prompt-depth-anything-vits).
Repositories use Apache-2.0 root licences; Large's inherited weight grant needs resolution
before redistribution. Non-commercial research here does not establish free-software
product licensing. No weights are committed.

RTX 4070 Ti, 64 patches: three repetitions of 1,000 stereo draws at each of 360×640 and
1080×1920 per eye. Isolated p95 is **1.02–1.09 ms**. With 5 Hz Metric Small inference,
90 Hz paced p95 is **4.91–5.92 ms**, worst p99.9 **6.35 ms**, with no measured draw or
scheduled completion misses. Concurrent depth inference p95 rises to **65–69 ms**.
The concurrent probe pairs Linux CUDA with Windows OpenGL. It excludes ImGui, live
patch uploads during drawing, compositor and scanout; they do
not certify headset comfort or the full 90 Hz application. Revisit updates measured
below 10 ms. The image payload cap is **112.5 MiB**; Torch reservation is allocator
information, not total process GPU residency. All GPU jobs held `ceres.lock`, were
bounded to 1,100 seconds, and finished by **09:11 CEST**, releasing the GPU before 11:50.

## Re-run and fly

From `work/p47`, with the existing frozen recordings and room:

```sh
bash e2e/depthmesh_run.sh --cached-only
```

This revalidates the frozen 3,925 input hashes, renders all 14 depth/pose variants on
Mesa, makes clips/wide sheets, drives the actual desktop app under its own Xvfb with
keyboard input and ImGui stereo, and audits the results. Logs, model repos, weights,
raw pairs, worlds and videos stay in `work/p47-out`. A fresh environment/weights use
`bash e2e/depthmesh_setup.sh`; omit `--cached-only` to fill missing depth caches under
the lock **before the fixed GPU deadline**. After it, CUDA work refuses to start.
GPU timing scripts are separate archived runs using an isolated Windows venv that reads
baseline dependencies; shared environments and the simulator installation are untouched.

Fly the saved world with Ito desktop input and Dear ImGui:

```sh
uv run python -m ito.proto.depthmesh ../p47-out/rendered/metric-small-robot-cap64/tour-world.npz
```

WASD, PgUp/PgDn, mouse-look with Tab, Home recenter, F12 capture; an ImGui control enables
64 mm stereo. This is saved-world desktop inspection, not a live robot or VR integration.

## Decision

Keep the experiment separate. It proves that image-textured grids recover appearance
cheaply, and that temporal depth/optional metric evidence matter. It does **not** establish
a coherent, comforting room from robot pose alone. The overlap with #48 is textured grids
and source retention; this implements #47's world patches, without #48's layered
view cache, flow propagation, source selection or estimator rewrite.

Sheets/held-out metrics, both pose modes, model/licence comparisons, causal audit,
short verdict and desktop bonus are complete. Owner preference/headset acceptance,
physical ToF transfer, native live publication latency, dynamics/dropout behavior and
aggregate GPU residency remain unestablished. Do not add this underneath #43/#45 as a
never-black guarantee: unobserved directions, cap eviction and geometry errors still win.
