# Independent estimation: repairs and remaining error

Issue #41; Ceres RTX 4070 Ti. **This does not complete #41 and is not ready for acceptance.** The lasting sweep heading jump is gone, motion publication is independent, and floor-based scale is removed. Tour rotation has not halved, dense depth remains inconsistent, and edge stability remains limited by inconsistent depth.

The following sections retain round 1; [Round 2 (joint)](#round-2-joint) records the current results. All round-1 comparisons use velstand v8, the real **62° long-side lens**, **360p30**, **Full / Input 100% / Max splats 1,000,000**, and no fixed GPU memory cap. Main's estimator is frozen at `0a125d1` (same estimator as `ff9993b`). Main's existing causal sweep/tour outputs were re-scored at the start; the independent short recording was captured and run through both estimators here. The mapper stays at this branch's inherited oracle configuration; no round-2 mapper changes are included.

## Changes

- `floor_scale`, camera-height/floor inference and the per-frame scale override are removed. One shared log-scale uses bidirectional KLT tracks, triangulation with odometry's baseline, parallax/uncertainty gates and robust consensus. Three consistent observations establish a target; subsequent updates are bounded to 0.5% log-scale per admitted frame. Pure rotation retains an explicitly unobserved visual gauge. Registered ToF is not integrated.
- MASt3R retains confidence-weighted pointmaps. Accepted dense correspondences transfer accumulated keyframe depth onto new image rays on CUDA, with confidence and a bounded reference-view weight. Clouds are extracted after local optimization; failed retries and invalid local solves restore their inputs. Refined frames: sweep 101/118, tour 479/528, short 145/146. This improves some surfaces but does **not** establish consistent metric depth.
- Changing the fitted camera delay without retiming retained poses/anchor caused the lasting heading jump. Delay fitting now requires several observations, rejects unsupported fits, limits changes and retimes the window consistently. Exact zero-delay simulation metadata disables fitting; receipt-stamped input still fits. Visual rotation innovations are gated against combined motion/vision covariance, so a noisy prior widens the gate. The canonical exact-time runs rejected zero visual factors: the gate alone is not the measured sweep fix.
- Robot motion publishes through the shared pose boundary immediately. Dense vision publishes its correction separately, applying it to the newest propagated pose without reversing the timestamp. The real worker ran while surface work took over 750 ms.
- Destructive carving requires depth agreement with a recent view. New neural observations still add geometry. This removes the new sweep floor blink seen with unconditional carving; it does not solve the remaining depth or image-boundary failures.

## Causal measurements

Each entry is **main → estimator**. Rotation/translation use one first-frame rigid alignment; there is no retrospective pose interpolation, map alignment or truth-assisted inference. Depth error is the median across frames of each frame's median absolute depth error. Inference cost changes which inputs are causally admitted, so these are complete causal runs, not identical frame subsets.

| Sequence | Admitted frames | Rotation p95, ° | Translation p95, m | Median absolute depth error, m |
|---|---:|---:|---:|---:|
| sweep180 | 103 → 118 | 12.851 → **1.807** | .0393 → .0278 | .415 → .306 |
| tour | 508 → 528 | 1.795 → **1.774** | .1090 → .0965 | .209 → .214 |
| independent short | 147 → 146 | 7.445 → **.499** | .0601 → .0344 | .328 → .258 |

| Sequence | Frame-median depth ratio: median [min, max], main → estimator | Median within-frame ratio p10–p90, main → estimator |
|---|---|---|
| sweep180 | .848 [.395, 7.776] → .921 [.658, 1.727] | .638–1.000 → .780–1.048 |
| tour | .869 [.215, 41.941] → .993 [.288, 20.503] | .660–1.000 → .802–1.270 |
| short | .842 [.421, 1.018] → 1.113 [.670, 1.702] | .691–1.000 → .923–1.303 |

The metric calibrator remains 1.0 on the unobservable sweep; tour changes slowly from 1.0 to .755, short from 1.0 to 1.112. Large output-depth errors remain, including the tour's 20.5× frame ratio. Removing floor-driven jumps has not fixed all output scale/shape failures. Longer-baseline calibration and feeding scale back into MASt3R regressed the tour and were rejected.

The real reconstruction-process replay published **49.73 Hz** distinct poses while active, versus main's **4.36 Hz** (17 s totals including startup: 727 versus 64 distinct poses). Motion-call-to-shared-pose p95 was **.039 ms**, maximum .146 ms; no timestamp regressions. A receipt-stamped run published 49.79 Hz, p95 .036 ms, without regressions. Main's fast read of a stale shared pose is not an application-latency measurement. These measurements exclude transport, render, compositor and scanout latency.

Bidirectional feature reprojection p95 remains **7.84 → 9.86 px** (sweep), **12.45 → 8.00 px** (tour), **6.92 → 7.25 px** (short). Tracks exclude failed KLT, forward/back disagreement, out-of-image and invalid depth; **no reprojection-error pruning**. Supported feature fractions are respectively 49.2→50.6%, 48.8→48.6%, 42.0→41.7%; these do not certify common-region dense coverage. Finite depth covers the image, including uncertain network guesses. The ≤2 px requirement is unmet.

## Motion review

Columns in the linked sheets are exact-source RGB | main | estimator, with arm, source identity, time and settings on every cell. Rows have labelled exposure times; common-view gaps can exceed 100 ms. Display clocks start at each estimator's first availability unless explicitly fixed for the independent/repeated comparisons. The display uses the true current camera to isolate map geometry; it does not certify the corrected pilot eye or headset comfort. `main-EE` names the production point-map placement control. EE/ET/TE mean estimated/true pose then depth through the unchanged experimental oracle mapper.

| Comparison | By-eye verdict and cells |
|---|---|
| Production point map, sweep | [10 s](estimator-img/sweep-point-10.jpg), [15 s](estimator-img/sweep-point-15.jpg), both rows: shelf/poster placement plainly closer to RGB than main's persistent heading shift; soft, warped floor remains. [Repeat 10 s](estimator-img/sweep-repeat-10.jpg), rows 1–2: shelf gain persists; the new small floor blink is absent after agreement-based carving. |
| Production point map, tour | [50 s](estimator-img/tour-point-50.jpg), rows 1–2: similarly soft books, imperfect floor. [76 s](estimator-img/tour-point-76.jpg), row 1: larger black ceiling boundary; row 2 is over six seconds later and cannot establish temporal stability. Repeat verdict below. |
| Independent short | [5 s](estimator-img/short-point-5.jpg), [20 s](estimator-img/short-point-20.jpg), both rows: better shelf placement, still soft floor. [15 s](estimator-img/short-point-15.jpg), rows 1–2: black right boundary where main has image. [Repeat](estimator-img/short-repeat-15.jpg), rows 1–2, reproduces the band, but source 506 (the first supporting view of the right edge) does not finish placement until +15.196 s. [After arrival](estimator-img/short-repeat-15.3.jpg), rows 1–2: most of the band fills. This is newly revealed space; the remaining thin edge mismatch is not a confirmed regression. |
| Oracle EE | [Sweep 10 s](estimator-img/sweep-EE-10.jpg), rows 1–2: correct shelf placement but coloured halos and floor distortion remain. [Tour 76 s](estimator-img/tour-EE-76.jpg), row 1: cabinet recovered from main's smear, still irregular boundaries. |
| Oracle ET | [Sweep 10 s](estimator-img/sweep-ET-10.jpg), rows 1–2: much cleaner shelf with true depth and repaired pose. [Tour 50 s](estimator-img/tour-ET-50.jpg), rows 1–2: less mangled books, soft floor persists. |
| Oracle TE | [Sweep 10 s](estimator-img/sweep-TE-10.jpg), rows 1–2: unresolved halos/warped floor with estimated depth. [Tour 76 s](estimator-img/tour-TE-76.jpg), row 1: cabinet more recognizable; holes/rings remain. No oracle-mapper promotion. |

[Repeated tour turn](estimator-img/tour-repeat-75.9.jpg), rows 1–2: the broader black ceiling boundary persists. Those ceiling points were outside earlier admitted views, so this is first-seen space under the motion-sheet rule. [A moment later](estimator-img/tour-repeat-76.3.jpg), rows 1–2: both maps have holes through newly exposed space. These selected cells show no confirmed repeatable quality regression after the carving fix; they also do not demonstrate sharp, stable edges throughout a turn.

All six requested EE/ET/TE runs and the supported point-map controls completed without OOM. Their recording runs miss many GL deadlines and are not timing acceptance; image readback and WSL D3D12 are included. Better pose numbers do not establish a stereo or headset quality pass.

## Timing and challengers

Exact-source RGB matching measured simulated exposure-to-receipt delay **67.51 / 69.82 ms median**, **69.38 / 73.21 ms p95**, robust jitter **1.04 / 1.75 ms** (sweep/tour). Delivery-to-Ito medians were **113.13 / 117.11 ms**. Distinct matches: 364/513 and 1,645/2,661; ambiguous static images were excluded. A controlled +70 ms receipt stamp with an initial 90 ms prior settled near 78 ms, rotation p95 1.12°, without a lasting jump. Truth-only diagnostics suggest an additional 5–7.5 ms SDK-pose/image phase offset; no truth-fitted offset was supplied to inference. Physical exposure/readout, rolling shutter and extrinsics remain uncalibrated.

[MapAnything](https://github.com/facebookresearch/map-anything)'s Apache model ran on four causal 280×504 views in full fp32, with calibrated intrinsics and optional generic odometry. Sweep/tour with motion took .98/1.03 s, reserved 7.16 GB, and produced depth ratios 1.70–2.14× / 2.44–2.53×. Image-only sweep showed a gross heading failure in this rapid-turn window. These bounded results do not justify replacing MASt3R. VGGT, DPVO and raw-IMU OpenVINS/Basalt were not evaluated. No raw head IMU or registered ToF was available.

## Reproduction and limits

`bash e2e/estimator_suite.sh OUT` extracts causal estimates and runs EE/ET/TE plus the supported point-map control under `ceres.lock`. `--reuse-estimates` requires identical source hashes. The report, reprojection, timing, stream, challenger and sheet scripts in `e2e/estimator_*.py` / `camera_timing.py` include one-command invocations. MapAnything needs an explicitly supplied external checkout/weights; it adds no product dependency.

The [result manifest](estimator-results.json) preserves exact settings, input/source hashes, schedules' hashes, diagnostics, all replay results and sheet identities. Canonical extraction used `8dc8804` (sweep `5d79db8`, before a variable rename on an unexecuted path). `563b6b0` adds pose revisions, corrected source timestamps and carving agreement, validated by real-process and supported point-map runs. Artifacts, rejected trials, recordings and logs remain under `~/Projects/ito/out/estimator-41/`; no weights are committed.

Still required: halve tour rotation p95, resolve dense scale/shape and cross-view consistency, verify boundaries throughout turns, validate longer sessions and calibrated physical sensor timing/extrinsics. The owner should check shelf/door sliding, floor breathing, close-object stereo at actual IPD, correction comfort during fast turns/reversals, and sustained OpenXR/compositor timing. Headset comfort is untested.

## Round 2 (joint)

**Still not accepted; #41 remains incomplete.** The producer is `2e852b6`, on the merged joint base `0453550`. Mapping below uses the inherited round-2 mapper `0634bbd`, with narrow replay hooks through `9ba83fc`; the other mapper agent’s new candidate is not included. All drives retain Full / Input 100% / Max 1,000,000, v8 policies and the 62° lens. No GPU memory cap.

Depth now carries registered float32 `depth_confidence` in [0,1]. These are uncalibrated weights: provisional predictions receive a quarter of network confidence; accepted multi-view depth agreement promotes them; geometric contradictions or excessive depth dispersion receive zero. Only reliable depth fusion feeds back into retained pointmaps. Missing confidence remains compatible with older callers/recordings. The inherited Gaussian mapper excludes zero-weight depth and seeds; its continuous loss weighting is still mapper work. The legacy point display retains provisional appearance while zero confidence prevents destructive carving. Dropping that appearance caused large black wall regions; a labelled ablation and the [corrected default](estimator-img/joint-tour-history-fix-76.jpg), both rows, isolate and repair that regression. This does not turn provisional depth into a trusted measurement or improve the reported support fraction.

The attempted post-hoc restoration of individual Sim3 scales was removed: changing those scales without compensating the pointmaps changes geometry. Native calibrated local adjustment remains. Star motion factors, additional clock fitting, removing local adjustment, pair-depth alignment and an independent feature rotation observer were evaluated and rejected after trajectory/depth regressions. None of those pose experiments ships.

Whole causal streams against sim truth (sweep / tour; translation p95 and median frame-median absolute depth error in metres):

| Cohort | Frames | Rotation p95, ° | Translation p95, m | Depth error, m |
|---|---|---|---|---|
| Current main 9a4e7ac | 120 / 511 | 15.296 / 1.962 | 0.054 / 0.074 | 0.314 / 0.325 |
| Round 1 9aea288 | 118 / 528 | 1.807 / 1.774 | 0.028 / 0.097 | 0.306 / 0.214 |
| Joint start (confidence export only) | 110 / 514 | 1.863 / 1.879 | 0.027 / 0.174 | 0.253 / 0.259 |
| Joint final 2e852b6 | 118 / 538 | 1.617 / 1.154 | 0.028 / 0.089 | 0.311 / 0.195 |

Admission schedules differ because real model work determines which frames arrive while busy; all schedules and source hashes are retained. These are whole-stream comparisons, not a matched-schedule ablation. All raw depth errors include uncertain network predictions. Frame-median depth ratios are .861 / .800 on fresh main versus .914 / .917 final; maxima are 1.03 / 59.38 versus 1.78 / 20.59. Thus the worst sampled sweep ratio is worse despite removing floor fitting. The independent metric multiplier stays 1 on the unobservable sweep and changes slowly to .887 on tour; no floor measurement participates. Near-wall shape/scale failures remain.

| Reprojection p95, px (sweep / tour) | Main | Round 1 | Joint final |
|---|---|---|---|
| Original KLT | 7.567 / 7.991 | 9.858 / 8.003 | 11.035 / 9.028 |
| Mutual SIFT + subpixel LK | 7.038 / 8.143 | 11.904 / 8.108 | 16.015 / 8.698 |

Residuals above use the original 360×640 source grid. With positive confidence in both views, final SIFT p95 is **8.379 / 5.607 px**, retaining **54.0% / 53.1%** of truth-visible common-region samples. The 2 px target and preservation of common-region support are not established. SIFT truth controls are 1.050 / 1.468 px; the old KLT truth control was 5.589 px on round-1 sweep. Changing the matcher is not an estimator improvement. All bidirectional valid feature residuals are retained; truth identifies common-region coverage only and never prunes scored residuals.

The real reconstruction process was replayed for 20 s per build/drive, with source hashes verified against main and the final code. Motion publication remains independent of dense surface work:

| Boundary measurement, sweep / tour | Current main | Final |
|---|---|---|
| Distinct poses per active second | 4.990 / 4.453 | 49.705 / 49.670 |
| Newest received motion → shared pose timestamp lag p95, ms | 901.6 / 1225.6 | 0 / 0 |
| Motion-call → shared-pose read p95, ms | stale reads; not fresh publication | .0359 / .0347 |

No timestamp regressed. Final sweep had 849 distinct stamps in 850 reads (one 20.3 ms maximum timestamp lag); tour had 911/911, maximum zero. Earlier runs without this repeat are also retained. These measurements exclude transport, render, compositor and scanout; they do not certify headset latency.

Images below use one source clock, exact shared RGB exposures and each arm’s own camera. E-pose cameras replay the latest received generic motion with only already-available visual corrections; legacy main holds its visual output, and T-pose arms use truth. Availability checks cover every recorded propagated camera. Render caps are labelled (98,304 for the oracle mapper, uncapped for point placement). A raw RGB exposure can precede the displayed propagated camera. Initial missing-motion/first-seen space may be black. Sheets are camera-FOV monoscopic diagnostic renders alongside concurrent 2160²/eye, 90° stereo, not headset recordings. These image-readback runs are not timing acceptance.

| Motion comparison (rows 1–2; RGB / main / branch) | By-eye result |
|---|---|
| Sweep EE [18 s](estimator-img/joint-sweep180-EE-18.jpg) | Clearer shelf and poster than main; melted book edges and floor seams remain. |
| Sweep ET [18 s](estimator-img/joint-sweep180-ET-18.jpg) | True depth gives cleaner book spines; floor remains imperfect. |
| Sweep TE [15 s](estimator-img/joint-sweep180-TE-15.jpg) | **Fails:** large white floaters intrude onto the plain wall in both rows. True pose does not repair estimated depth. |
| Tour EE [76 s](estimator-img/joint-tour-EE-76.jpg) | **Fails:** large spiky floaters/black regions versus main’s filled wall in both rows. |
| Tour ET [76 s](estimator-img/joint-tour-ET-76.jpg) | Clearer box stripes, but the box/wall still distort even with true depth. |
| Tour TE [76 s](estimator-img/joint-tour-TE-76.jpg) | **Fails:** white streaks intrude onto the right wall in both rows. |
| Production point display [76 s](estimator-img/joint-tour-point-76.jpg) | Broad black-wall regression removed; soft/broken cabinet detail still prevents a sharp-edge pass. [Ablation](estimator-img/joint-tour-ablation-76.jpg) explicitly labels ignored confidence. |

These are joint representation/estimator comparisons, not an isolated mapper ablation. Floaters in already visible surfaces count as failures; first-seen black alone does not. The inherited mapper is not promoted by these results.

The independent short recording from round 1 was reused, without a new capture: fresh main 145 frames versus final 144. Rotation p95 **7.445 → .708°**, translation p95 **.0596 → .0312 m**, raw depth error **.332 → .179 m** (round 1: .499°, .0344 m, .258 m). Supported SIFT remains **5.191 px**, with 67.3% common-region support. [20 s](estimator-img/joint-short-point-20.jpg), both rows: better wall/poster placement but soft/ghost surfaces remain. This checks the independent point-display path; the independent Gaussian-map arms were not rerun.

[MapAnything](https://github.com/facebookresearch/map-anything) was expanded to six prior views over 8 s, selected with odometry alone (.754 m baseline, 13.6° heading range). Its full-fp32 280×504 confidence batch took 1.44 s and reserved 8.20 GB; depth errors .14–.16 m still yielded 8.76 processed-grid px reprojection p95 (8.68 px confidence-supported, 95.4% common-region support). Earlier views in this batch are retrospective outputs, not a causal pose stream. This does not justify a product replacement. **DPVO was not tested:** its CUDA/C++ build exceeded the job’s 5 GB limit even with one job and one nvcc thread. It was dropped; no runner or product dependency lands.

Remaining acceptance work: consistent multi-view depth, tour rotation halved without worse translation, stable edges throughout turns and independent-sequence stereo quality. Physical exposure/readout, rolling shutter, camera/joint/IMU extrinsics, raw head IMU, ToF and the headset remain untested. The owner should check shelf/door sliding, wall floaters, floor appearance as it leaves view, close-object stereo at actual IPD, fast-turn/reversal correction comfort and sustained compositor timing. The round-1 simulated receipt-delay/jitter measurements above remain the timing evidence; no truth-fitted delay is supplied to inference.

The floor-assumption removal and independent publication are implemented. The complete “Done when” conditions do **not** hold: tour rotation has not halved, tour translation is worse than freshly measured main, cross-view depth is inconsistent, and the sheets fail stable stereo geometry. No headset quality pass is claimed.

`bash e2e/estimator_suite.sh FRESH_OUT` runs a new causal cohort and EE/ET/TE plus the supported point control, on source time zero, serially under `ceres.lock`. The report, reprojection, stream and sheet drivers are runnable independently. The existing native-load helper also completed a 3 s TT compatibility check (22 observations, 345 optimization steps, no OOM) with optional worker fields absent. This job reused the existing gsplat binary; no further native builds were attempted after dropping DPVO. The [joint manifest](estimator-joint-results.json) preserves all main/round-1/joint metrics, both publication passes, confidence support, source/admission hashes, rejected trials, run settings and the labelled images. Full logs/arrays and the bounded prebuilt evaluation bootstrap remain under `~/Projects/ito/out/recon-40-41/estimator/`.

## Round 4: DPVO on Ceres

DPVO now builds in the worktree's Linux `.venv`: the first successful native build reused surviving object files and completed in 115 seconds with `MAX_JOBS=3`, `NVCC_THREADS=2`, CUDA 12.4, GCC 13 and Torch 2.5.1+cu124. The job limit was 14,408,482,816 bytes; `memory.high=max`. The earlier three failures were under the old 5 GB job limit. No DPVO dependency is imported by the application or Windows release; this remains an explicit e2e challenger, not a bundled Windows tracker.

Reproduction uses the official [DPVO source](https://github.com/princeton-vl/DPVO/tree/0ac95b656d1fda91c271d2a106460d19ad966fc7), pinned at `0ac95b656d1fda91c271d2a106460d19ad966fc7`, Eigen 3.4.0 under its `thirdparty/`, and the official `dpvo.pth` (SHA256 `30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12`). No viewer, retrieval or classical loop-closure extensions are needed. Existing Ceres inputs are under `~/Projects/ito/out/recon-40-41/estimator/{dpvo,dpvo-weights,cuda-dpvo}`.

```bash
export DPVO_ROOT="$HOME/Projects/ito/out/recon-40-41/estimator/dpvo"
# For a fresh checkout (keep weights outside the Ito repository):
git clone --recursive https://github.com/princeton-vl/DPVO.git "$DPVO_ROOT"
git -C "$DPVO_ROOT" checkout 0ac95b656d1fda91c271d2a106460d19ad966fc7
git -C "$DPVO_ROOT" submodule update --init --recursive
curl -fL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip \
  -o "$DPVO_ROOT/thirdparty/eigen.zip"
.venv/bin/python -m zipfile -e "$DPVO_ROOT/thirdparty/eigen.zip" "$DPVO_ROOT/thirdparty"
git -C "$DPVO_ROOT" apply "$PWD/e2e/dpvo-build.patch"
uv pip install --python .venv/bin/python --no-deps \
  'torch-scatter==2.1.2+pt25cu124' \
  -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
uv pip install --python .venv/bin/python \
  yacs==0.1.8 numba==0.68.0 llvmlite==0.50.0 pypose==0.9.5
# Keep the existing torch==2.5.1+cu124, torchvision==0.20.1+cu124,
# numpy==2.4.6 and scipy==1.15.3. Do not resolve DPVO's environment.yml.
export DPVO_ROOT="$HOME/Projects/ito/out/recon-40-41/estimator/dpvo"
export CUDA_HOME="$HOME/Projects/ito/out/recon-40-41/estimator/cuda-dpvo"
export CXX="$HOME/Projects/ito/out/oracle-40/gcc/root/usr/bin/g++-13"
export CC="$HOME/Projects/ito/out/oracle-40/gcc/root/usr/bin/gcc-13"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=3 NVCC_THREADS=2 TORCH_CUDA_ARCH_LIST=8.9
flock "$HOME/Projects/ito/ceres.lock" uv pip install --python .venv/bin/python \
  --no-deps --no-build-isolation "$DPVO_ROOT"
```

Two source/toolchain adaptations are required on this host; the committed `e2e/dpvo-build.patch` contains the exact source edits. In DPVO's `setup.py`, add `['dpvo.loop_closure']` to `find_packages()` (upstream otherwise omits an imported directory), and add `'--threads=2'` to all three `nvcc` argument lists. The Ceres glibc 2.43 headers conflict with CUDA 12.4's six declarations: in a **private copy** of the toolkit, append `noexcept` before the semicolon of the `extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__` declarations for `rsqrt`, `rsqrtf`, `sinpi`, `sinpif`, `cospi`, `cospif` in `include/crt/math_functions.h`. Ceres's shared toolkit was not edited. The private toolkit copies `bin/` and `include/`; other toolkit directories are symlinks to the original. The exact private-copy preparation is:

```python
# Run with the worktree Python before the install above.
from pathlib import Path
import re, shutil
source = Path.home() / "Projects/ito/out/oracle-40/cuda"
private = Path.home() / "Projects/ito/out/recon-40-41/estimator/cuda-dpvo"
private.mkdir(exist_ok=True)
for entry in source.iterdir():
    target = private / entry.name
    if not target.exists():
        if entry.name in {"bin", "include"}:
            shutil.copytree(entry, target)  # Dereference; shared files stay untouched.
        else:
            target.symlink_to(entry.resolve(), target_is_directory=entry.is_dir())
header = private / "include/crt/math_functions.h"
text = header.read_text()
for name in ("rsqrt", "rsqrtf", "sinpi", "sinpif", "cospi", "cospif"):
    pattern = rf"(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ [^;\n]*\b{name}\([^;\n]*\))([ \t]*);"
    text = re.sub(pattern, r"\1 noexcept;", text)
header.write_text(text)
```

 On a compatible glibc/toolkit pair that header adaptation is unnecessary. The package-manifest fix then rebuilds/reinstalls in about four seconds from the object cache.

`e2e/estimator_dpvo.py` runs calibrated 288×512 BGR, 96 patches, a ten-frame optimization window, no loop closure and no map alignment. Full precision also disables upstream's hard-coded update autocast. Each pose is frozen when that call returns, without `terminate()` or retrospective trajectory reconstruction. Arrivals while busy are dropped. It retains raw monocular poses and a separate uncertainty-gated rotation-only correction; metric translation is explicitly the generic robot prior. Startup, skipped observations and fallback are labelled. Concurrent live tracker/mapper contention and a Windows DPVO bundle are untested; neither is implied by isolated replay throughput.
