# Independent estimation: repairs and remaining error

Issue #41; Ceres RTX 4070 Ti. **This does not complete #41 and is not ready for acceptance.** The lasting sweep heading jump is gone, motion publication is independent, and floor-based scale is removed. Tour rotation has not halved, dense depth remains inconsistent, and edge stability remains limited by inconsistent depth.

All comparisons use velstand v8, the real **62° long-side lens**, **360p30**, **Full / Input 100% / Max splats 1,000,000**, and no fixed GPU memory cap. Main's estimator is frozen at `0a125d1` (same estimator as `ff9993b`). Main's existing causal sweep/tour outputs were re-scored at the start; the independent short recording was captured and run through both estimators here. The mapper stays at this branch's inherited oracle configuration; no round-2 mapper changes are included.

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
