# DROID-SLAM as an alternative to MASt3R-SLAM

Research note, October 2026. Question: should Ito's monocular reconstruction move from
MASt3R-SLAM (`ito/reconstruction/slam.py`) to DROID-SLAM or one of its relatives, given the
Microduck problems (featureless walls during turns, 50-160 map restarts per 100 s, doubled or
tilted walls after realigns, 3.5-5 Hz tracking) on a 12 GB RTX 4070 Ti shared with the renderer.

Evidence tags used below:

- **[V] verified here**: read in the cloned source at the commit named, or checked against
  package indexes / Ito's own lockfile. No NVIDIA GPU was available, so nothing was run on CUDA.
- **[P] paper/README claim**: from the authors or third-party papers, with a link.
- **[E] estimate**: my own arithmetic or inference; to be confirmed on Ceres.

## TL;DR

**Prototype on Ceres; do not adopt yet.** DROID-SLAM itself (BSD-3, small network, no "lost"
state, dense per-keyframe depth plus confidence) fits Ito's failure mode better than MASt3R-SLAM:
on a featureless wall its learned confidence drops and the pose stays where its motion model put
it, which we can make the robot's measured heading. Its weakness is the mirror image of MASt3R's:
depth needs parallax, and a head that mostly pans gives little. The candidate worth measuring is
therefore **DROID-SLAM frontend only + robot-heading pose initialisation + a monocular metric depth
prior as its "sensor depth"** (the recipe NVIDIA's ViPE ships, Apache-2.0). Adopt only if the replay
experiment below beats MASt3R-SLAM on heading p95 and restarts without losing coverage.

## Candidates

| | DROID-SLAM | DPVO / DPV-SLAM | GO-SLAM | ViPE | MASt3R-SLAM (current) | VGGT-SLAM 2.0 |
|---|---|---|---|---|---|---|
| Repo | [princeton-vl/DROID-SLAM][droid] | [princeton-vl/DPVO][dpvo] | [youmi-zym/GO-SLAM][goslam] | [nv-tlabs/vipe][vipe] | [rmurai0610/MASt3R-SLAM][mast3r] | [MIT-SPARK/VGGT-SLAM][vggt-slam] |
| Code licence [V] | BSD-3 | MIT | Apache-2.0 | Apache-2.0 (Unik3D part CC BY-NC-SA) | **CC BY-NC-SA 4.0** | BSD-2 |
| Weights | `droid.pth`, Google Drive, no separate licence stated [V] | Google Drive zip [V] | DROID weights | DROID weights + chosen depth model | MASt3R, CC BY-NC-SA [V, `ito/build.py`] | VGGT-1B, Meta's VGGT licence ([LICENSE.txt][vggt-lic]) |
| Last commit [V] | 2025-05-04 (multi-GPU, torch 2.7) | 2026-09-20 | 2024-02-08 | 2026-09-10 (v1.2.0 on PyPI 2026-06) | 2025-11-08 | 2026-06-29 |
| Dense depth per keyframe | yes, 1/8 res + learned upsampling | no, sparse patches | yes (via DROID) + NeRF | yes | yes, pointmap + confidence | yes |
| Pose prior hook | init only (no prior factor) [V] | init only | init only | per-frame pose input + Python solver with pluggable terms [V] | none; Ito realigns around it | none |
| Verdict | **prototype** | no: too sparse for splats | no: unmaintained, needs tiny-cuda-nn | **reference implementation / plan B** | baseline | no: 1B-param model, licence |

Also considered and dropped: DBA-Fusion ([GREAT-WHU/DBA-Fusion][dbaf]) fuses DROID's dense BA
with IMU/wheel factors in GTSAM, exactly the shape of fusion we want, but it is **GPL-3.0** [V]:
reference only. 2026 feed-forward systems (AIM-SLAM, EC3R-SLAM, Flash-Mono, VGGT-GS SLAM; see
[AIM-SLAM][aim], [EC3R-SLAM][ec3r]) are either paper-only in the results I found or build on
ViT-L/1B backbones with the same VRAM and licence problems as MASt3R.

## 1. Fit

**Input.** Monocular RGB with known pinhole intrinsics (`fx fy cx cy`, optional distortion) is
DROID's native input [V, README]. Ito always knows intrinsics, so no GeoCalib step.
DROID also has an RGB-D mode: any per-pixel depth written into `video.disps_sens` is used as a
soft constraint inside BA and as the initial disparity of each new keyframe
(`droid_frontend.py` `_update`, `depth_video.py` `__item_setter`) [V]. That is the hook for a
metric depth prior.

**Pose / IMU priors.** Upstream DROID has no pose-prior factor: `droid_backends.ba(...)` solves
flow residuals plus the `disps_sens` term only; the only pose control is fixing all poses below
`t0` [V, `src/droid.cpp`, `depth_video.py:ba`]. What we can do, cheapest first:

1. **Initialise each new frame's pose from the robot.** `DroidFrontend._init_next_state` sets
   the next pose from a damped constant-velocity model [V]. Replace that with the last keyframe
   pose composed with the robot's camera rotation delta (head FK + IMU, as `slam.py` already
   computes in `predicted`). Where flow is informative BA overrides it; where it is not, the
   update operator's confidence is near zero and LM damping leaves the pose near the prior [E].
2. **Keep Ito's heading guard** (`HEADING_TOLERANCE`) as a watchdog, but correct by rewriting
   the pose and fixing it (`t0`), not by restarting the map.
3. **Add a yaw prior factor**: a 6x6 block on the pose diagonal of the Hessian in
   `droid_kernels.cu` (C++), or a `SolverTerm` in ViPE's Python BA (`vipe/slam/ba/terms.py`
   already has `DenseDepthFlowTerm`, `DispSensRegularizationTerm`) [V]. DBA-Fusion shows the
   full IMU-factor version works with DROID ([repo][dbaf]) [P].

**Low-texture walls.** No method gets flow from a blank wall. The difference is what happens
next. MASt3R-SLAM "matches almost anything to anything" there (Ito's own comment in `slam.py`),
which Ito catches as a heading stray and restarts. DROID's update operator predicts a per-pixel
confidence for every edge; blank regions get low weight and contribute little to BA, and DROID has
no lost/relocalise state at all [V, code structure]. Combined with prior initialisation (1), a
blank wall should degrade to "pose = robot heading, depth = prior" instead of a restart [E].
Kanai et al. show DROID's failure cases shrink when BA is initialised with monocular depth priors
([arXiv 2406.00929][kanai]) [P].

**Pure rotation.** Rotation is well determined by flow, and rotation produces flow, so the motion
filter (`filter_thresh` 2.4 px mean flow) keeps adding frames [V]. Depth is not observable
without baseline: during a head pan DROID's new keyframes inherit the 70th-percentile disparity of
recent keyframes (`_update`) and stay there [V]. MASt3R predicts depth from a single image pair
and does not have this problem. This is the main reason the prototype needs a depth prior
(section 3).

## 2. Cost

DROID's network works at a fixed working size; the demo resizes every input to about 384x512
pixels of area, rounded to a multiple of 8 [V, `demo.py`]. **Camera resolution (360p / 720p /
1080p) changes only decode and resize cost**, not SLAM cost. A 16:9 frame becomes 328x584
(feature grid 41x73) at the demo setting, or 240x432 if we pick a smaller working size.

| | Value | Source |
|---|---|---|
| Inference GPU memory, demo defaults (buffer 512, full backend) | "at least 11G" | [P] [README][droid] |
| Datasets needing 24 GB (TartanAir, ETH3D) | full backend | [P] [README][droid], [DPV-SLAM][dpvslam] |
| DROID-VO (frontend only), RTX 3090 | 40 FPS, 8.7 GB | [P] [DPVO paper][dpvo-paper] |
| DROID-SLAM on EuRoC, RTX 3090 | 21 FPS | [P] [DINO-VO supp.][dinovo] |
| DROID-SLAM vs MASt3R-SLAM, EuRoC, same paper | ~24 vs ~10 FPS | [P] via [FoundationSLAM][foundation] |
| DROID-SLAM on a UAV study | 12.8 FPS, 4.2 GB | [P] (hardware unstated) |
| DPVO, RTX 3090 | 2-5x real time, ~4 GB | [P] [DPVO paper][dpvo-paper] |
| DPV-SLAM | 5-7 GB, 1-4x real time | [P] [DPV-SLAM][dpvslam] |
| ViPE (whole pipeline incl. depth models) | 3-5 FPS, offline annotation | [P] [ViPE][vipe-paper] |
| MASt3R-SLAM (paper, single thread) | 14.6 FPS average | [P] [CVPR supp.][mast3r-supp] |
| MASt3R-SLAM in Ito, Microduck sim | 3.5-5 Hz | task brief |

**Ito-sized budget [E].** From the tensor shapes in `depth_video.py` and `modules/corr.py` at
328x584: one keyframe slot is about 3.7 MB (three 128-channel fp16 feature maps, image, upsampled
disparity); one correlation-volume edge (4-level fp16 pyramid) is about 24 MB; the frontend keeps
at most 48 edges, about 1.2 GB. With a 64-keyframe ring (Ito only keeps a 4 s window) that is
~0.25 GB of buffers, plus a few hundred MB of weights and activations: **roughly 2-3 GB for a
frontend-only DROID**, versus a ViT-Large MASt3R encoder. A metric depth prior adds a second
network (0.5-1.5 GB depending on model). Throughput on a 4070 Ti should be below a 3090's because
correlation lookups are memory-bandwidth bound (504 vs 936 GB/s); expect roughly 15-30 Hz
frontend-only at 328x584 before contention with the renderer [E].

**Frontend vs global BA.** Ito keeps "now plus a little memory" (4 s window), so DROID's global
backend and loop closure buy nothing and cost the 11-24 GB above. Run the frontend only, with
`buffer` sized to the window and old keyframes dropped (ViPE 2026-09 `pose_only_long` does this
sliding-window retirement [P, README]).

**Latency per frame [E].** Every frame pays the motion filter: one feature/context encode plus
one update-operator step. A frame that becomes a keyframe additionally pays the frontend:
`iters1` (4) + `iters2` (2) update/BA rounds over the local window. Expect single-digit ms for a
filtered frame and tens of ms for a keyframe; the pose of non-keyframes is not computed by
upstream DROID until the offline `traj_filler`, so Ito must take it from the robot prior (or run
a motion-only BA, which `ba(..., motion_only=True)` supports [V]).

## 3. Output

- **Depth**: `video.disps` per keyframe at 1/8 resolution, optionally convex-upsampled to full
  working resolution (`disps_up`, `--upsample`) [V].
- **Confidence**: per-edge, per-pixel weights from the update operator (`graph.weight`), and a
  multi-view consistency count from `droid_backends.depth_filter`, which DROID's own visualiser
  uses to drop inconsistent points [V]. Either maps directly to a splat keep/drop mask.
- **Points**: `droid_backends.iproj(poses, disps, intrinsics)` returns world points on the GPU
  [V], ready for `RGBDBackend.integrate_points` via DLPack exactly as `slam.py` does now.
- **Scale**: monocular DROID is up to scale, like MASt3R; Ito already handles arbitrary-unit
  maps (`slam.py` sizes splats in Sim3 units). Two ways to get metres: (a) a metric depth prior
  in `disps_sens`, which pins scale inside BA (ViPE's `keyframe_depth: moge2-l` / `metric3d-small`
  configs do exactly this [V]); (b) scale from robot translation. The Microduck mostly turns and
  its walking odometry is weak, so (a) is the one to test. Licence-clean depth models:
  MoGe-2 (MIT, DINOv2 parts Apache-2.0) ([microsoft/MoGe][moge]) [P]; Depth Anything V2 **Small**
  (Apache-2.0; Base/Large are CC BY-NC) ([repo][dav2]) [P].
- **Revisions**: the frontend keeps refining the last ~25 keyframes. `RGBDBackend` fuses each
  observation once, so fuse a keyframe when it is 2-3 keyframes old (most refinement done) and
  splat the newest keyframe's depth for immediacy [E].

## 4. Windows

| Piece | Status |
|---|---|
| torch 2.5.1+cu124 win_amd64 cp312 | in Ito's `uv.lock` already [V] |
| lietorch on MSVC + CUDA 12.4 | **already built by `ito/build.py`** for MASt3R-SLAM, with a patch renaming CPU kernels to avoid MSVC symbol clashes [V]. Ito pins lietorch `e7df865` (2025-05); DROID's submodule pins `7f68764`. DROID uses only lietorch's Python API, so the newer rev should do [E] |
| `droid_backends` (4 files, ~2.2k lines CUDA/C++) | not built on Windows here (no nvcc). Same pattern as `mast3r_slam_backends`: `typedef long LongType` in `droid_kernels.cu` and `altcorr_kernel.cu` [V] is 32-bit on MSVC and needs Ito's existing `long` to `int64_t` rewrite. Upstream issues #39, #62, #104 are Windows build/inference reports, mostly unresolved ([issues][droid-win]) [P] |
| `torch_scatter` | no Windows wheels on PyPI (sdist only, 2.1.2) [V]; PyG's wheel index was unreachable from this sandbox. Not needed: DROID uses only `scatter_sum` (BA helpers) and `scatter_mean` (one call) [V]; both are `Tensor.index_add_` / `scatter_reduce` in plain torch. ViPE vendors its own `scatter_ext` instead [V] |
| Other deps | DROID's core imports torch, lietorch, numpy, cv2, scipy (graph utils); `factor_graph.py` imports matplotlib but never uses it [V]. No new runtime dependency beyond what the `slam` extra pins |
| Prebuilt wheels | none. PyPI `lietorch` is an unrelated project (bsmetsjr) [V]. `nvidia-vipe` 1.2.0 is sdist-only and builds its CUDA extensions at install [V] |
| WSL2 fallback | works for development (MASt3R-SLAM has a `windows` branch for WSL [V, README]) but not for shipping: Ito's release is a native Windows zip, and CUDA-in-WSL plus a native OpenXR renderer means two processes on two OSes sharing one GPU. Not recommended beyond a first experiment |

Conclusion: a native Windows build is a small extension of what `ito.build` already does (one more
`build("droid_backends", ...)` call, the `long` rewrite, scatter replaced in Python) [E].

## 5. Licence and maintenance

| Project | Code | Weights | Maintained |
|---|---|---|---|
| DROID-SLAM | BSD-3 | `droid.pth` (Google Drive, pickled; load with `weights_only=True`, pin sha256, host a mirror); trained on TartanAir (CC BY 4.0) | low activity; last update May 2025 added torch 2.7 support |
| lietorch | BSD-3 | n/a | last commit May 2025 |
| DPVO / DPV-SLAM | MIT | Google Drive | active (Sept 2026) |
| GO-SLAM | Apache-2.0 | DROID | dormant since Feb 2024 |
| ViPE | Apache-2.0 except Unik3D (CC BY-NC-SA) | downloads third-party models; pick licence-clean ones | active (Sept 2026, NVIDIA) |
| DBA-Fusion | GPL-3.0 | | dormant since Mar 2025 |
| MASt3R-SLAM | CC BY-NC-SA 4.0 | MASt3R CC BY-NC-SA | low activity |
| VGGT-SLAM | BSD-2 | VGGT licence (Meta) | active |

DROID-SLAM is the first candidate that would make Ito's monocular path free of non-commercial
terms (code and, if MoGe-2 or DA-V2-Small is the depth prior, weights).

## 6. Integration sketch

What the backend has to look like, read from `ito/reconstruction/__init__.py`, `slam.py`,
`rgbd.py`, `ring.py`:

- `_run` builds a backend from `recon.backend` and calls
  `backend.integrate(rgb, depth, camera, now, measured=...)`, then reads `camera_pose`,
  `tracked`, `lost`, and the `RGBDBackend` slot arrays (`records`, `keys`, `dirty`, `retiring`)
  to publish through `UpdateRing`. `integrate` must return within one frame budget and never
  block rendering (separate process already).
- So the new backend is `ito/reconstruction/droid.py: DROIDBackend(RGBDBackend)` with the same
  constructor (`capacity, intrinsics, *, report, **options`) and the same `integrate` signature
  as `SLAMBackend`. `__init__.py` gains `"droid"` beside `"rgbd"` and `"slam"`; the ring, the
  renderer and the protocol do not change.
- A `droid_runtime.py` mirroring `mast3r_runtime.py`: verify weights by sha256, load
  `lietorch_backends` and `droid_backends` from `models/native`, fail with a clear message
  without CUDA.

```
integrate(rgb, depth, camera, now, measured):
    image = resize to working size; K scaled                       # 328x584 or 240x432
    if depth model: disp_prior = 1 / metric_depth(image)[3::8, 3::8]
    motion_filter.track(t, image, disp_prior, K)                     # may skip the frame
    frontend.next_pose = last_kf_pose * robot_delta(camera)          # replaces const-velocity
    frontend()                                                       # local BA, 25-kf window
    camera_pose = settled kf pose * robot_delta  or  motion-only BA  # every frame
    heading guard: |yaw - robot yaw| > 12 deg -> rewrite pose, fix it (no restart)
    for settled keyframes: iproj + depth_filter mask -> integrate_points (DLPack)
    drop keyframes older than the window (sliding buffer)
```

Effort [E], to a finished backend that meets AGENTS.md (no prototype lands):

| Work | Days |
|---|---|
| `ito.build`: fetch/pin DROID + weights mirror, build `droid_backends` on Windows/Linux, scatter swap | 2-3 |
| `droid_runtime.py` + `droid.py` backend, sliding buffer, per-frame pose, fusion of settled keyframes | 4-5 |
| Robot-prior initialisation and non-restarting heading guard | 1-2 |
| Metric depth prior (model fetch, licence notice, resize/align to DROID grid) | 2-3 |
| Yaw prior factor in the CUDA BA (only if the experiment shows init alone is not enough) | 2-3 |
| `slam_replay.py` backend switch + Ceres runs + tuning | 2-3 |
| **Total** | **~2-3 weeks** |

## 7. Recommendation and Ceres experiment

**Recommendation: prototype on Ceres** (research branch, not main). Gate adoption on the numbers
below. If DROID with prior init still restarts or strays on pans, try ViPE's solver (Python BA
with a yaw prior term) before giving up on the family.

**Recordings.** Existing `e2e/slam_drive.py --record` captures of the Microduck sim, at least:
(a) standing head-pan sweeps past a plain wall, (b) walking with body turns, (c) a cluttered room
control. Record each at 640x360 and 1280x720. Keep the same recordings for every run.

**Harness change (research branch only).** `e2e/slam_replay.py` is MASt3R-specific (imports
`SLAMBackend`, `mast3r_slam.config`, reads `backend.frames`). Add `--backend {mast3r,droid}`,
use backend-neutral attributes (`tracked`, `realigned`, `camera_pose`, `records`, `keys`, a
`keyframes` count), and add to the summary: `restarts_per_100s = restarts / seconds * 100`,
`vram_peak_mb` from `torch.cuda.max_memory_allocated()` plus process total from `nvidia-smi
--query-compute-apps`.

**Runs.** Each recording, each resolution, 3 repeats (DROID's README notes only its
asynchronous mode is nondeterministic; repeats show the spread for both backends):

| ID | Backend | Config |
|---|---|---|
| M0 | MASt3R-SLAM | current `slam.py` (baseline) |
| D1 | DROID | mono, upstream constant-velocity init, no guard |
| D2 | DROID | D1 + robot-delta pose init |
| D3 | DROID | D2 + heading guard (rewrite, no restart) |
| D4 | DROID | D3 + metric depth prior (MoGe-2 small/base or DA-V2-Small) |
| D5 | DROID | D4 at 240x432 working size |

Then rerun the best of M0 and D* with the desktop renderer running a splat scene on the same GPU
(`uv run ito` against the replayed driver) to measure contention: SLAM Hz and render frame time.

**Metrics** (all already in `slam_replay.py` except the two added above):

| Metric | Field | Adopt if DROID config |
|---|---|---|
| Heading error vs driver gaze, median / p95 | `heading_error_deg_median`, `_p95` | p95 < 12 deg (`HEADING_TOLERANCE`) and below M0 |
| Restarts per 100 s | `restarts_per_100s` | < 10 (M0: 50-160) |
| Flat-view fraction | `flat_fraction` | at most M0's |
| Coverage of the 1 s-ahead view | `prediction_coverage_median` | at least M0's |
| Prediction quality | `prediction_psnr_median` | not worse than M0 by > 1 dB |
| Tracking rate | `slam_hz`, `frame_ms_p95` | >= 10 Hz, p95 < 100 ms |
| VRAM | `vram_peak_mb` | <= 4 GB including depth model |
| Render frame time with SLAM running | renderer metrics | holds 90 fps on desktop |

Also look at the `predict-*.png` side-by-sides for doubled or tilted wall planes: that artefact is
a realign symptom and has no single metric.

## Sources

- DROID-SLAM repo and README: https://github.com/princeton-vl/DROID-SLAM (cloned at `2dfd39f`);
  paper https://arxiv.org/abs/2108.10869
- DROID-SLAM Windows issues: https://github.com/princeton-vl/DROID-SLAM/issues?q=is%3Aissue+windows
- lietorch: https://github.com/princeton-vl/lietorch (`e7df865`)
- DPVO / DPV-SLAM: https://github.com/princeton-vl/DPVO (`0ac95b6`); DPVO paper
  https://arxiv.org/abs/2208.04726; DPV-SLAM paper https://arxiv.org/abs/2408.01654
- GO-SLAM: https://github.com/youmi-zym/GO-SLAM (`e954a85`)
- ViPE: https://github.com/nv-tlabs/vipe (`8c9f361`), https://pypi.org/project/nvidia-vipe/,
  paper https://arxiv.org/abs/2508.10934
- DBA-Fusion: https://github.com/GREAT-WHU/DBA-Fusion (`2c658a3`)
- MASt3R-SLAM: https://github.com/rmurai0610/MASt3R-SLAM (`e6f4e3d`); paper
  https://arxiv.org/abs/2412.12392; supplement
  https://openaccess.thecvf.com/content/CVPR2025/supplemental/Murai_MASt3R-SLAM_Real-Time_Dense_CVPR_2025_supplemental.pdf
- VGGT-SLAM: https://github.com/MIT-SPARK/VGGT-SLAM (`35327ac`); VGGT licence
  https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt
- DINO-VO supplement (DROID/DPVO timings on EuRoC, RTX 3090):
  https://openaccess.thecvf.com/content/CVPR2026F/supplemental/Chen_DINO-VO_Learning_Where_CVPRF_2026_supplemental.pdf
- FoundationSLAM (DROID vs MASt3R-SLAM on EuRoC): https://arxiv.org/abs/2512.25008
- AIM-SLAM: https://arxiv.org/abs/2603.05097; EC3R-SLAM:
  https://www.researchgate.net/publication/396142880
- Kanai et al., depth-prior initialisation for DROID: https://arxiv.org/abs/2406.00929
- MoGe-2: https://github.com/microsoft/MoGe; Depth Anything V2:
  https://github.com/DepthAnything/Depth-Anything-V2
- torch_scatter on PyPI: https://pypi.org/project/torch-scatter/

[droid]: https://github.com/princeton-vl/DROID-SLAM
[droid-win]: https://github.com/princeton-vl/DROID-SLAM/issues?q=is%3Aissue+windows
[dpvo]: https://github.com/princeton-vl/DPVO
[dpvo-paper]: https://arxiv.org/abs/2208.04726
[dpvslam]: https://arxiv.org/abs/2408.01654
[goslam]: https://github.com/youmi-zym/GO-SLAM
[vipe]: https://github.com/nv-tlabs/vipe
[vipe-paper]: https://arxiv.org/abs/2508.10934
[mast3r]: https://github.com/rmurai0610/MASt3R-SLAM
[mast3r-supp]: https://openaccess.thecvf.com/content/CVPR2025/supplemental/Murai_MASt3R-SLAM_Real-Time_Dense_CVPR_2025_supplemental.pdf
[vggt-slam]: https://github.com/MIT-SPARK/VGGT-SLAM
[vggt-lic]: https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt
[dbaf]: https://github.com/GREAT-WHU/DBA-Fusion
[aim]: https://arxiv.org/abs/2603.05097
[ec3r]: https://www.researchgate.net/publication/396142880
[kanai]: https://arxiv.org/abs/2406.00929
[dinovo]: https://openaccess.thecvf.com/content/CVPR2026F/supplemental/Chen_DINO-VO_Learning_Where_CVPRF_2026_supplemental.pdf
[foundation]: https://arxiv.org/abs/2512.25008
[moge]: https://github.com/microsoft/MoGe
[dav2]: https://github.com/DepthAnything/Depth-Anything-V2
