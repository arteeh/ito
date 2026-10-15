# Oracle mapper: three prove/kill rounds

Current verdict: see [round 3, joint #40/#41](#round-3-joint-4041). Rounds 1–2 below remain history.

## Round 1 — kill this configuration

Issue #40; main baseline `ff9993b`; Ceres RTX 4070 Ti. **Reject this mapper/scheduler configuration before investing in tracking.** True inputs eventually produce sharper books and poster edges, but first-second detail remains soft. More decisively, the optimized map breaches the 90 Hz GL budget: training causes sweep misses, and the mature tour map is already too expensive to draw with training stopped. This is a configuration verdict, not a rejection of optimized Gaussians.

Both fresh recordings use **velstand v8** (policy commit `3750776`), the real **62° long-side lens**, **360p30**, delivered upright as **360×640** (`fx=fy=532.5694, cx=179, cy=320`). Every comparison uses **Full / Input 100% / Max splats 1,000,000**, with main's **live VRAM manager and no fixed memory cap**.

## Look at the sheets

Motion columns are RGB | main | mapper; read two moments 100 ms apart, top to bottom. Every cell states its arm, time and settings. Prefix page 1 shows 100/500 ms; page 2 shows 1/2 s. Black in newly exposed space is not itself a regression. All verdicts below are by eye.

| Pose/depth arm | Result and marginal gain |
|---|---|
| **TT** | [Sweep, first second](gsplat-oracle-img/sweep180-TT-motion-0.9.jpg), rows 1–2: unresolved labels and coloured shelf halos. [Tour, 2 s](gsplat-oracle-img/tour-TT-prefix-2.jpg), row 2: cleaner shelf/wall; [sweep 10 s](gsplat-oracle-img/sweep180-TT-motion-10.0.jpg) and [tour 50 s](gsplat-oracle-img/tour-TT-motion-50.0.jpg), both rows: plainly sharper poster/books, still soft floor grain. Gains arrive unevenly and fail the display budget. |
| **TE** | True pose alone leaves floaters, transient shelf holes and a warped floor: [sweep 10 s](gsplat-oracle-img/sweep180-TE-motion-10.0.jpg), [tour 50 s](gsplat-oracle-img/tour-TE-motion-50.0.jpg), both rows. Replacing depth with truth removes substantial damage. |
| **ET** | True depth alone leaves displaced/bent books and unstable outlines: [sweep](gsplat-oracle-img/sweep180-ET-motion-10.0.jpg), [tour](gsplat-oracle-img/tour-ET-motion-50.0.jpg), both rows. True pose restores coherence. |
| **EE** | [Sweep against main-EE](gsplat-oracle-img/sweep180-EE-v-main-EE-motion.jpg), both rows: changing holes through the shelf/floor absent from main. [Tour](gsplat-oracle-img/tour-EE-v-main-EE-motion.jpg): somewhat cleaner wall/fill but distorted books and floor. No consistent win over main. |
| **ToF + true pose** | Very soft initially; sharper revisited books, but distorted floors and missing surfaces on turns: [sweep](gsplat-oracle-img/sweep180-TF-motion-10.0.jpg), [tour](gsplat-oracle-img/tour-TF-motion-50.0.jpg). Optional metric evidence, not a demonstrated dense-depth replacement. |

Every arm has all four causal ages:

| Arm | Sweep: 100/500 ms; 1/2 s | Tour: 100/500 ms; 1/2 s |
|---|---|---|
| TT | [early](gsplat-oracle-img/sweep180-TT-prefix-1.jpg), [later](gsplat-oracle-img/sweep180-TT-prefix-2.jpg) | [early](gsplat-oracle-img/tour-TT-prefix-1.jpg), [later](gsplat-oracle-img/tour-TT-prefix-2.jpg) |
| TE | [early](gsplat-oracle-img/sweep180-TE-prefix-1.jpg), [later](gsplat-oracle-img/sweep180-TE-prefix-2.jpg) | [early](gsplat-oracle-img/tour-TE-prefix-1.jpg), [later](gsplat-oracle-img/tour-TE-prefix-2.jpg) |
| ET | [early](gsplat-oracle-img/sweep180-ET-prefix-1.jpg), [later](gsplat-oracle-img/sweep180-ET-prefix-2.jpg) | [early](gsplat-oracle-img/tour-ET-prefix-1.jpg), [later](gsplat-oracle-img/tour-ET-prefix-2.jpg) |
| EE | [early](gsplat-oracle-img/sweep180-EE-prefix-1.jpg), [later](gsplat-oracle-img/sweep180-EE-prefix-2.jpg) | [early](gsplat-oracle-img/tour-EE-prefix-1.jpg), [later](gsplat-oracle-img/tour-EE-prefix-2.jpg) |
| ToF | [early](gsplat-oracle-img/sweep180-TF-prefix-1.jpg), [later](gsplat-oracle-img/sweep180-TF-prefix-2.jpg) | [early](gsplat-oracle-img/tour-TF-prefix-1.jpg), [later](gsplat-oracle-img/tour-TF-prefix-2.jpg) |

The [sweep](gsplat-oracle-img/sweep180-interventions-1.0.jpg) and [tour](gsplat-oracle-img/tour-interventions-1.0.jpg) grids compare all inputs at exactly the same camera. [Sweep eyes](gsplat-oracle-img/sweep180-TT-stereo.jpg) and [tour eyes](gsplat-oracle-img/tour-TT-stereo.jpg) use 64 mm IPD at the calibrated camera FOV: corresponding outlines, but soft/warped close geometry. These do not certify stereo foreground comfort.

## Display and memory

The [result manifest](gsplat-oracle-results.json) records input hashes, causal/format audits, memory samples and individual deadline runs. Native Windows NVIDIA GL uses MMCSS Games/high, two 2160×2160 RGBA8 eyes, 90° FOV, mid-eye sorting, ±3 cm head motion and two upload packets/frame. Three 20-second repeats compare main's frozen mature map, the optimized frozen map, then that optimized map with the real online CUDA mapper running. No readback, estimator contention, compositor or scanout is included. Feedback-file writes run off the display thread. This controlled frozen-renderer load check supplements the live-map replay; it is not headset acceptance.

| Sequence / workload | Missed 90 Hz GL deadlines, three repeats | Completion p99 range |
|---|---|---|
| sweep180 / main frozen | 1, 0, 0 | 7.95–8.57 ms |
| sweep180 / optimized frozen | 3, 0, 0 | 8.09–8.16 ms |
| sweep180 / optimized + optimizer | 287, 270, 298 | 12.31–12.43 ms |
| tour / main frozen | 0, 0, 0 | 8.57–8.74 ms |
| tour / optimized frozen | 900, 900, 900 | 15.31–15.37 ms |
| tour / optimized + optimizer | 913, 908, 914 | 22.62–23.02 ms |

The visual runs retained at least **9.75 GiB device free** at manager samples. No normal run reported OOM; each failure-injection run reported exactly one and recovered in bf16.

The tour's optimized working set is about 798k splats at this one-million setting. VRAM headroom does not imply a safe draw cost. **Next: bound the live render working set and optimization work, and improve initialization/refinement convergence; re-prove the oracle case before tracking work.** Both pose and depth consistency still matter after that.

## Method and reproduction

[gsplat 1.5.3](https://github.com/nerfstudio-project/gsplat/tree/v1.5.3), Apache-2.0, fits centers, rotations, three scales, opacity, SH0 colour and per-frame affine exposure against retained calibrated RGB views. The [mapper](../ito/reconstruction/mapper.py) keeps 96 GPU images plus archived sources, seeds up to 6,000 faint Gaussians/observation, densifies every 100 steps and publishes every 100 ms. Classic rasterization matches production GL. Native parameters/rasterization stay fp32; **Half uses bf16 image loss on this RTX**. Real allocation failures after seeding and densification verify rollback, dropped work, continued mapping and no switch to Full; the seeding run also captures the [ImGui degradation warning](gsplat-oracle-img/half-oom-warning.png). CPU launch fails clearly without CUDA. Promotion as a pilot backend is rejected.

Live `ITO_RECORD` sessions have exact uncompressed RGB/depth/pose sidecars from one copied MuJoCo scene. Current-main MASt3R/fusion freezes estimates as first available, skipping busy inputs. All arms use those same RGB observations and availability times; truth never receives extra/earlier frames. No map alignment, retrospective interpolation or truth-assisted scale correction. A single first-frame rigid alignment defines the world. Main controls use unchanged `SLAMBackend.place` and production retirement/acknowledgement publication; saved snapshots bake actual GL fades.

The common schedule has 103/508 observations. Evaluated windows are 17/85 s after first common availability, from 19.81/90.86 s recordings. Visual capture is 10 Hz. Prefix snapshots retain the last GL-completed state before each deadline; compression follows the run. ToF is synthetic: 8×8 angular-zone medians, 45° square FOV, 15 Hz, 100 ms delay, 4 m range and Gaussian noise of 20 mm + 1% distance, with true-pose delay compensation. No dense boundary interpolation.

Artifacts and environment: `~/Projects/ito/out/oracle-40/` (`{sweep180,tour}/{source,causal-full}/`, `visual/`, `native/`, `review-corrected/`, `NOTES.md`). The isolated CUDA/GCC environment is selected by `e2e/oracle.sh`; model bundles/releases remain read-only. After `uv sync --extra slam --group oracle`, capture fresh drives with `e2e/oracle_capture.py`, then run:

```sh
bash e2e/oracle_estimate.sh
bash e2e/oracle_suite.sh
bash e2e/oracle_native.sh
PYTHONPATH=. .venv/bin/python e2e/oracle_audit.py ~/Projects/ito/out/oracle-40
PYTHONPATH=. .venv/bin/python e2e/oracle_report.py --out ~/Projects/ito/out/oracle-40 --sequence sweep180 --arm TT
```

All GPU work holds `ceres.lock`. Preliminary baselines with incorrect retirement handling, WSL timing proxies and smoke runs were discarded; the committed sheets use corrected baselines. One completed baseline trace lacks its optional worker footer at shutdown; its measured trace/snapshots remain auditable and the manifest leaves unrecorded peaks null. Shutdown now completes between CUDA integrations and passes a separate real drive.

**Owner-only:** OpenXR/compositor deadlines, foreground rivalry at actual IPD, head translation, fast-turn shimmer/popping and v8-waddle comfort. Physical camera/ToF timing, dynamic subjects, long sessions and real memory-pressure eviction remain untested. Board reading was blocked by missing `read:project` scope; no board/issue/PR edits were made.

## Round 2 — sharper early, bounded drawing; configuration still rejected

Owner-requested second round on the same recordings. **KILL for pilot promotion:** the first-second sharpness and sustained GL overload from round 1 are substantially improved, but TE/EE still damage surfaces relative to matching main, and the wide-FOV foreground geometry is not clean enough to certify the oracle case. This does not reject optimized Gaussians. The next bottleneck is depth consistency and the map/visible-selection treatment of foreground geometry; the small pose correction cannot repair the estimator's large jump. Estimator fixes remain outside this branch.

### What changed

New unexplained depth regions receive up to 80k near-opaque (`0.995`) anisotropic seeds per observation. Surface tangents and thickness come from calibrated depth/focal length, with finer sampling at RGB edges. Removing thin-surface mip attenuation prevents these seeds turning back into haze. RGB, geometry, opacity, rotation, scale and per-frame exposure continue to optimize through real gsplat.

Training batches one or two **full calibrated images**, keeps 24 GPU RGB views and archived sources, focuses on pending keyframes toward 100 image iterations, and uses a recent six-view local map. Keyframes arrive after one second or 30°/30 cm motion; intervening observations still seed and train. Fused Adam, a compiled Gaussian transform, cheaper RGB-gradient loss and densify/prune every 400 updates reduce overhead. Half uses bf16 image loss; gsplat parameters remain fp32. Map-side SE(3) corrections update every eight steps, with priors and component bounds of 10 cm/5°. Neither published estimator poses nor its output files change.

A separate **98,304-instance render budget** selects two distinct near representatives per projected tile entirely on the GPU, shares selection/sort between eyes and applies a conservative LOD footprint. Max splats still caps the map at one million. Scanning the map remains proportional to map size; this is bounded sort/draw work, not proof of constant total cost for arbitrary maps. A real-map check verified identical selected IDs/bounds across 60 views before/after the final shared-workgroup bounds reduction; its median sort time barely changed, so that micro-change is not credited with the overall gain.

### Fairness and settings

**Full / Input 100% / Max splats 1,000,000; 360p30 raw 360×640, velstand v8, real 62° long-side lens.** Both main and mapper now fit the same **288×512** observation grid, which is what main's Input 100% means. Round 1's fitter instead used raw 360×640; before/after cells label this difference. Current main `9a4e7ac` has byte-identical reconstruction/render code to baseline `ff9993b`.

All arms replay identical RGB observations and availability prefixes: 103 sweep observations, 488 tour observations within the 17/85-second evaluation windows (508 in the complete tour schedule). ET/EE and their main controls render from their own latest **causally available published estimated pose**, never a truth camera or the map's correction. TE has a matching main-TE control; TF uses main-TT. RGB columns show the physical true camera, so estimated-map columns need not spatially align with RGB. Estimator interfaces are unchanged; no estimator-41 handoff was present when this cohort completed.

Startup primes synthetic checkerboard/plane data, then discards all state before the observation clock. Frame times now mean completed GPU readback, not render start. All four prefix deadlines use the last completed display state; the 100 ms cell can honestly be empty. Runs with cold imports/incorrect timestamps or unmatched grids were discarded. The committed evidence comes only from the corrected cohort.

ToF remains synthetic 8×8 angular zones, approximately 45° square FOV, 15 Hz, 100 ms delay, 4 m range and 20 mm + 1% noise, with true-pose delay compensation. The RGB horizontal FOV clips some zones. **This round adds a smooth inverse-depth completion prior for initialization**, interpolating inside the measured hull and extending nearest values beyond it. Only original measured zones contribute depth loss or justify nearer-surface replacement. It uses no dense truth depth, but its sharp colour is not evidence of measured fine geometry.

### Look at the corrected sheets

Each motion sheet has two rows 100 ms apart; columns are RGB | matching main | round 2. All cells label time/settings/camera/grid/cap. Before/after sheets are main | round 1 | round 2, re-rendered from the same own camera. Round 1's original estimated-arm truth-camera sheets above remain historical evidence only.

| Arm | By-eye result and marginal gain |
|---|---|
| TT | [Sweep first second](gsplat-oracle-round2-img/sweep180-TT-motion-0.9.jpg), [tour first second](gsplat-oracle-round2-img/tour-TT-motion-0.9.jpg), both rows: plainly sharper labels/poster than main, floor retained in observed areas. [Tour before/after](gsplat-oracle-round2-img/tour-TT-before-after.jpg), rows 1–2, shows haze removed. [Late native eyes](gsplat-oracle-round2-img/tour-native-78.0-whole.jpg), middle row, still distort close geometry/wall; complete oracle comfort condition is unproven. |
| TE | [Sweep 10 s](gsplat-oracle-round2-img/sweep180-TE-motion-10.0.jpg), [tour 50 s](gsplat-oracle-round2-img/tour-TE-motion-50.0.jpg), both rows: sharper but repeated shelf flecks, ragged spines and wall/floor damage versus main-TE. **Fails.** Replacing estimated depth with truth removes much of this damage at the same true camera. |
| ET | [Sweep 10 s](gsplat-oracle-round2-img/sweep180-ET-motion-10.0.jpg), [tour 50 s](gsplat-oracle-round2-img/tour-ET-motion-50.0.jpg), both rows: clearer books than main-ET from the same estimated camera; residual bent outlines, but no comparable repeatable new hole in these cells. Own-pose comparison removes the former misleading shift. Small refinement is not a fix for the 12° path jump. |
| EE | [Sweep 10 s](gsplat-oracle-round2-img/sweep180-EE-motion-10.0.jpg), [tour 50 s](gsplat-oracle-round2-img/tour-EE-motion-50.0.jpg), both rows: sharper detail with recurring shelf floaters/surface damage absent from blurred main-EE. **Fails**, even with own-pose rendering. |
| TF | [Sweep first second](gsplat-oracle-round2-img/sweep180-TF-motion-0.9.jpg), [tour first second](gsplat-oracle-round2-img/tour-TF-motion-0.9.jpg): colour sharpens quickly. [Tour 1/2 s](gsplat-oracle-round2-img/tour-TF-prefix-2.jpg), both rows: extra floor-edge gaps; [tour 76 s](gsplat-oracle-round2-img/tour-TF-motion-76.0.jpg): bowed foreground box/wall. **Fails the complete geometry bar**; optional coarse metric evidence remains useful. |

Every arm retains all causal snapshots and motion moments, including less flattering late views:

| Arm | Sweep prefixes 100/500 ms; 1/2 s | Tour prefixes 100/500 ms; 1/2 s | Before/after sweep; tour |
|---|---|---|---|
| TT | [early](gsplat-oracle-round2-img/sweep180-TT-prefix-1.jpg), [later](gsplat-oracle-round2-img/sweep180-TT-prefix-2.jpg) | [early](gsplat-oracle-round2-img/tour-TT-prefix-1.jpg), [later](gsplat-oracle-round2-img/tour-TT-prefix-2.jpg) | [sweep](gsplat-oracle-round2-img/sweep180-TT-before-after.jpg), [tour](gsplat-oracle-round2-img/tour-TT-before-after.jpg) |
| TE | [early](gsplat-oracle-round2-img/sweep180-TE-prefix-1.jpg), [later](gsplat-oracle-round2-img/sweep180-TE-prefix-2.jpg) | [early](gsplat-oracle-round2-img/tour-TE-prefix-1.jpg), [later](gsplat-oracle-round2-img/tour-TE-prefix-2.jpg) | [sweep](gsplat-oracle-round2-img/sweep180-TE-before-after.jpg), [tour](gsplat-oracle-round2-img/tour-TE-before-after.jpg) |
| ET | [early](gsplat-oracle-round2-img/sweep180-ET-prefix-1.jpg), [later](gsplat-oracle-round2-img/sweep180-ET-prefix-2.jpg) | [early](gsplat-oracle-round2-img/tour-ET-prefix-1.jpg), [later](gsplat-oracle-round2-img/tour-ET-prefix-2.jpg) | [sweep](gsplat-oracle-round2-img/sweep180-ET-before-after.jpg), [tour](gsplat-oracle-round2-img/tour-ET-before-after.jpg) |
| EE | [early](gsplat-oracle-round2-img/sweep180-EE-prefix-1.jpg), [later](gsplat-oracle-round2-img/sweep180-EE-prefix-2.jpg) | [early](gsplat-oracle-round2-img/tour-EE-prefix-1.jpg), [later](gsplat-oracle-round2-img/tour-EE-prefix-2.jpg) | [sweep](gsplat-oracle-round2-img/sweep180-EE-before-after.jpg), [tour](gsplat-oracle-round2-img/tour-EE-before-after.jpg) |
| TF | [early](gsplat-oracle-round2-img/sweep180-TF-prefix-1.jpg), [later](gsplat-oracle-round2-img/sweep180-TF-prefix-2.jpg) | [early](gsplat-oracle-round2-img/tour-TF-prefix-1.jpg), [later](gsplat-oracle-round2-img/tour-TF-prefix-2.jpg) | [sweep](gsplat-oracle-round2-img/sweep180-TF-before-after.jpg), [tour](gsplat-oracle-round2-img/tour-TF-before-after.jpg) |

Native stereo uses the actual 90° selection, with main/TT/TF rows and left/right columns: sweep [1 s whole](gsplat-oracle-round2-img/sweep180-native-1.0-whole.jpg)/[crop](gsplat-oracle-round2-img/sweep180-native-1.0-camera-crop.jpg), [12 s whole](gsplat-oracle-round2-img/sweep180-native-12.0-whole.jpg)/[crop](gsplat-oracle-round2-img/sweep180-native-12.0-camera-crop.jpg); tour [1 s whole](gsplat-oracle-round2-img/tour-native-1.0-whole.jpg)/[crop](gsplat-oracle-round2-img/tour-native-1.0-camera-crop.jpg), [78 s whole](gsplat-oracle-round2-img/tour-native-78.0-whole.jpg)/[crop](gsplat-oracle-round2-img/tour-native-78.0-camera-crop.jpg). Crops are explicitly labelled, not different narrower-FOV selections. No numerical image-quality gate was used.

### Throughput, deadlines and memory

| Arm | Sweep steps/s; full-image iterations/s; keyframes ≥100 | Tour steps/s; full-image iterations/s; keyframes ≥100 |
|---|---|---|
| TT | 86.6; 169.4; 20/23 | 76.6; 151.0; 83/87 |
| TE | 83.2; 163.2; 20/23 | 65.8; 130.3; 80/87 |
| ET | 86.6; 169.4; 20/23 | 76.6; 151.1; 82/86 |
| EE | 80.9; 159.0; 20/23 | 66.2; 130.9; 79/86 |
| TF | 58.5; 115.9; 17/23 | 52.5; 104.3; 65/87 |

Median keyframe image iterations are **100 in every arm**; unfinished/recent keyframes remain in the denominators. Batched distinct views count separately, never patches or duplicate views. Round 1's approximately 60 steps/s processed one larger raw image, so steps/s alone is not a matched-pixel throughput comparison. These experiments change initialization, optimization, pose refinement and LOD together; they establish their combined gain, not an isolated causal gain for each code change.

Native NVIDIA GL: two 2160² RGBA8 eyes, 90° FOV, 64 mm IPD, ±3 cm head motion, mid-eye sort, two real upload packets/frame, MMCSS Games/high. Three 20-second repeats per workload. Mature map is frozen at 12/78 s; loaded cases concurrently run the real cold-prefix TT/TF online CUDA mapper, whose progress is audited. No readback during timing. This tests controlled render/optimizer contention, **not changing-map uploads, estimator inference, HUD, compositor or headset scanout**.

| Sequence / workload | Missed native 90 Hz deadlines, three repeats | Completion p99 range |
|---|---|---|
| sweep / main frozen | 1, 0, 1 | 8.40–8.63 ms |
| sweep / TT frozen | 0, 1, 1 | 8.53–8.65 ms |
| sweep / TT + optimizer | 1, 0, 0 | 6.28–6.46 ms |
| sweep / TF frozen | 0, 0, 0 | 8.51–8.66 ms |
| sweep / TF + optimizer | 0, 0, 0 | 6.11–6.27 ms |
| tour / main frozen | 3, 0, 0 | 8.79–8.98 ms |
| tour / TT frozen | 1, 0, 1 | 8.89–9.15 ms |
| tour / TT + optimizer | 0, 0, 0 | 8.78–8.87 ms |
| tour / TF frozen | 0, 1, 0 | 8.83–9.07 ms |
| tour / TF + optimizer | 0, 0, 1 | 8.17–8.32 ms |

No additional repeatable deadline deficit versus main appears in this cohort; isolated misses also occur in main. This is not a zero-miss claim or headset acceptance. Selection draws at most 98,304 instances regardless of map count, including empty tiles. Full maps continue growing beyond that budget.

Live-manager samples retain at least **10,202 MiB device free** across the visual cohort, with zero normal OOMs and no fixed memory limit. Real impossible-allocation faults after seed and refinement each report one OOM and recover while staying **Half/bf16**; [the degradation warning](gsplat-oracle-round2-img/half-oom-warning.png) is captured. Genuine system memory pressure/eviction remains untested. The [round-2 manifest](gsplat-oracle-round2-results.json) includes input hashes, completed-frame/own-camera/causal audits, iteration counts, all 30 native results, selection checks and image hashes.

Artifacts: `~/Projects/ito/out/oracle-40/round2/` (`visual/`, `native/`, `native-views/`, `review/`, `audit.json`, `NOTES.md`). Reproduce with `bash e2e/oracle_round2.sh ~/Projects/ito/out/oracle-40/round2` after the round-1 recording/estimate preparation; it takes `ceres.lock` around each GPU run. Reports use software GL only for snapshot comparisons; native captures independently check wide-FOV fidelity. All windows/processes started for this round are closed; the owner's virtual display remains running.

**Owner checks / remaining work:** actual OpenXR and compositor misses with the estimator and changing map active; foreground rivalry at real IPD, head translation, LOD popping, fast turns and v8 waddle; physical ToF/camera delay, dynamics, long sessions and real pressure recovery. The native foreground distortion and TE/EE surface damage must be resolved before promotion. Re-run the unchanged mapper input interfaces on estimator-41 output when available. No issue/PR/board edits were made.


## Round 3 (joint #40/#41)

**Reject this mapper/scheduler configuration under the requested joint-round gate.** TT retains first-second sharpness and substantially improves the close cabinet/wall, but a close foreground edge remains pitted even with true inputs. TF still breaks surfaces at the close turn. This rejects the configuration, not optimized Gaussians as a representation. Estimated-input claims below concern the actual propagated pilot eye, never a truth-camera rendering of an estimated map.

### Inputs and comparisons

Ceres RTX 4070 Ti; **Full / Input 100% / Max splats 1,000,000**, separate **98,304 render cap**; **360p30**, upright **360×640**, velstand v8, real **62° long-side lens**, fitting **288×512**. The same two original live `ITO_RECORD` drives are reused; no new drive is implied. Every arm consumes the same frozen observations and causal availability schedule, including true-input arms. The final extraction uses estimator **9ba83fc** (inference core `2e852b6`), with **117 sweep / 534 tour** estimates; the display windows admit 117/514 observations over 17/85 seconds. Inputs, settings, source hashes and actual code revisions are retained in the manifest.

First, all nine arms/controls were rerun with the unchanged round-2 mapper and merged estimator `9aea288`: sweep TE/EE lose much of the earlier shelf damage; tour TE still has floor creases, and tour EE develops a **worse black floor blotch**. [Initial tour EE](gsplat-oracle-round3-img/initial-before-tour-EE-50.jpg), rows 1–2, columns main | old round 2 | unchanged mapper/new estimates, records that failure. Better depth alone did not solve the mapper. These historical E cameras hold the published visual pose, as their captions state.

The final paired comparisons use immutable mapper `0634bbd` and the round-3 mapper against **identical final frozen input files, release times and the same propagated-camera reader**. Columns are RGB | same-input point-map control | round-2 mapper | round-3 mapper; rows are 100 ms apart. The point control includes #41's provisional-appearance/carving changes; it is not claimed to be byte-identical main. Separate actual-main sheets retain main `9a4e7ac`'s old estimator and its own published camera. Historical and final schedules differ; historical comparisons align physical source time, not displayed age.

For E poses, each available visual correction (`pose × inverse(motion_pose)`) applies to the latest received 50 Hz motion pose. Before a visual result, the correction is identity. No future motion or visual result is consulted. Map-side pose refinement never changes this pilot camera or the estimator's published path.

### What changed

Raw depth from three recent views supplies independent agreement weights and support counts. The fitted map cannot certify its own geometry. A nearer replacement needs two confirming raw observations, sufficient weight and unexplained RGB. Two independent contradictions can retire a provisional floater; supported surfaces are protected. A confidence-weighted Cauchy depth loss limits inconsistent observations' influence.

Unknown neural depth retains **provisional colour in uncovered space**, while remaining excluded from metric depth loss and destructive evidence. Dropping those rays initially produced black walls in TE/EE and was rejected. Explicit sparse-sensor modality prevents low-confidence dense images from accidentally invoking ToF interpolation. Missing confidence arrays remain compatible with earlier exports.

The close-up diagnostic separated seeding from fitting: the same TT wall was flat before optimization and rippled after; disabling the render cap did not remove the ripple. Seed-local centre and normal priors, a two-times scale-growth bound, tighter depth-discontinuity footprint guards and a fixed initial map gauge reduce that drift. Other keyframes still refine bounded SE(3) corrections. These priors travel with anchor retirement, clone/split/prune and OOM rollback. Real anisotropic Gaussians, calibrated full-image batches, exposure fitting and retained source observations remain.

ToF completion trials using RGB-guided interpolation, slope extrapolation, inverse-depth curvature and zone-median fitting were **rejected by eye and reverted**. Final ToF keeps linear inverse-depth completion inside the measured hull and nearest continuation outside it. Completed rays carry no centre/normal metric prior; raw zones retain a weak prior. This is colour-bearing initialization, not measured dense geometry. The same noisy delayed 8×8, 45°, 15 Hz sensor model is used throughout; no extra true depth is supplied.

### Motion review

Read rows 1–2, especially the rightmost round-3 column, against the point control and unchanged fitter beside it. No numerical image-quality gate is used.

| Arm | Result and marginal evidence |
|---|---|
| **TT** | [Sweep first second](gsplat-oracle-round3-img/sweep180-TT-motion-0.9.jpg) and [tour first second](gsplat-oracle-round3-img/tour-TT-motion-0.9.jpg): plainly sharper books/poster than the point control. [Tour close-up](gsplat-oracle-round3-img/tour-TT-motion-76.0.jpg), both rows: cabinet stripes and wall corner straighten relative to the unchanged mapper, without the earlier large wall ripple. The later [native close-up](gsplat-oracle-round3-img/tour-native-78.0-camera-crop.jpg), middle row, retains a pitted side edge in both eyes. [Removing the cap](gsplat-oracle-round3-img/tour-TT-boundary-cap-check.jpg), both rows, reproduces it: stored geometry remains a limiter. [Sweep late](gsplat-oracle-round3-img/sweep180-TT-motion-10.0.jpg) retains sharp labels. |
| **TE** | [Sweep](gsplat-oracle-round3-img/sweep180-TE-motion-10.0.jpg), both rows: most large wall floaters removed and shelf filled. [End turn](gsplat-oracle-round3-img/sweep180-TE-motion-15.0.jpg) still has jagged floor seams; [tour](gsplat-oracle-round3-img/tour-TE-motion-50.0.jpg) retains imperfect book/surface detail. True pose alone does not supply correct geometry. |
| **ET** | [Sweep](gsplat-oracle-round3-img/sweep180-ET-motion-10.0.jpg) and [tour](gsplat-oracle-round3-img/tour-ET-motion-50.0.jpg), both rows: cleaner spines/outlines than the point control. Residual doubled/bent edges with true depth implicate pose or map-side fitting; these are not cured estimator poses. |
| **EE** | [Tour](gsplat-oracle-round3-img/tour-EE-motion-50.0.jpg), both rows: earlier floor blotch removed; shelf and wall filled. [Sweep](gsplat-oracle-round3-img/sweep180-EE-motion-10.0.jpg): much less ceiling damage, but some books are softer than the unchanged fitter and poster ghosting remains. Floor seams remain. This is a partial robustness gain, not stable metric geometry. |
| **TF** | [Sweep first second](gsplat-oracle-round3-img/sweep180-TF-motion-0.9.jpg) and [tour first second](gsplat-oracle-round3-img/tour-TF-motion-0.9.jpg): sharp colour arrives quickly. **Fails:** [tour close-up](gsplat-oracle-round3-img/tour-TF-motion-76.0.jpg), both rows, has pitted wall surfaces, broken stripes and floor-edge gaps where main is filled. [Sweep late](gsplat-oracle-round3-img/sweep180-TF-motion-10.0.jpg) also retains imperfect geometry. More interpolated colour is not more sensor evidence. |

Actual main `9a4e7ac` with its old estimator remains visible in [sweep EE](gsplat-oracle-round3-img/sweep180-actual-main-EE.jpg) and [tour EE](gsplat-oracle-round3-img/tour-actual-main-EE.jpg): actual main | same-input point control | new mapper, each from its own camera. The final E close-up was also reviewed at 76/76.1 s in the retained output; the old giant wall floaters are suppressed, but estimated-depth cabinet stripes remain incomplete. No all-frames or headset quality pass is claimed.

| Arm | Sweep 100/500 ms; 1/2 s | Tour 100/500 ms; 1/2 s |
|---|---|---|
| TT | [prefixes](gsplat-oracle-round3-img/sweep180-TT-prefix.jpg) | [prefixes](gsplat-oracle-round3-img/tour-TT-prefix.jpg) |
| TE | [prefixes](gsplat-oracle-round3-img/sweep180-TE-prefix.jpg) | [prefixes](gsplat-oracle-round3-img/tour-TE-prefix.jpg) |
| ET | [prefixes](gsplat-oracle-round3-img/sweep180-ET-prefix.jpg) | [prefixes](gsplat-oracle-round3-img/tour-ET-prefix.jpg) |
| EE | [prefixes](gsplat-oracle-round3-img/sweep180-EE-prefix.jpg) | [prefixes](gsplat-oracle-round3-img/tour-EE-prefix.jpg) |
| TF | [prefixes](gsplat-oracle-round3-img/sweep180-TF-prefix.jpg) | [prefixes](gsplat-oracle-round3-img/tour-TF-prefix.jpg) |

Clocks start at first common estimate availability. Prefixes use the last completed GPU state before each deadline; a 100 ms cell may honestly be empty. Source-time and own-camera captions distinguish actual age from the displayed map's completion. Cells retain their real run commits: dense TT/ET paths were unchanged by the reverted ToF-only trials, and final E-depth cells include provisional-colour handling. The legacy adapter excludes zero-weight depth; the new adapter permits it for provisional colour. This handling is part of the mapper intervention, not a different frozen recording.

### Throughput and limits

| Arm | Sweep steps/s; image iterations/s; KFs ≥100 | Tour steps/s; image iterations/s; KFs ≥100 |
|---|---|---|
| TT | 65.1; 128.6; 16/26 | 62.9; 125.1; 85/91 |
| TE | 65.5; 129.2; 16/26 | 62.5; 124.2; 85/91 |
| ET | 65.7; 129.6; 16/26 | 62.8; 124.9; 85/90 |
| EE | 65.9; 130.1; 16/26 | 63.3; 125.8; 85/90 |
| TF | 38.2; 75.5; 8/26 | 37.1; 74.2; 45/91 |

Dense-arm median keyframe iterations are 100. TF medians are 32.5/93; its sparse-depth CPU completion/addition work leaves less fitting time. Recent and unfinished keyframes remain in the denominators. Counts are full calibrated images, including both distinct views in a batched step. Robustness costs throughput compared with the unchanged fitter; this is not a measured scheduler-speed improvement. Recording readback is included in these live-fitting runs, and their GL miss counters are **not** native deadline acceptance.

Latest estimator `92b8dba` is merged. Its report, assets and omitted-argument compatibility fix change no inference or producer code: `git diff 9ba83fc..92b8dba -- ito/reconstruction e2e/oracle_estimate.py` is empty. Final frozen inputs retain their actual extraction provenance and hashes; no inference rerun or different availability schedule is implied by the later report commit.

### Native 90 Hz and memory

Windows NVIDIA GL uses the same controlled load as round 2: two 2160×2160 RGBA8 eyes, 90° FOV, 64 mm IPD, ±3 cm head translation, one mid-eye sort and two 4,096-record uploads/frame, MMCSS Games/high. There is no timing readback. Each optimized arm draws its saved mature map while the real cold-prefix CUDA fitter runs and drains publications. **The drawn map is frozen; this excludes the live estimator, changing-map upload workload, OpenXR/compositor and scanout.** This is GL contention evidence, not headset acceptance.

Missed 90 Hz GL deadlines in three 20-second repeats:

| Saved map / concurrent workload | Sweep | Tour |
|---|---|---|
| main-TT | 0, 1, 0 | 1, 0, 1 |
| main-EE | 1, 4, 0 | 900, 900, 900 |
| TT | 0, 0, 0 | 0, 0, 0 |
| TE | 0, 0, 0 | 0, 0, 0 |
| ET | 0, 0, 0 | 0, 0, 0 |
| EE | 0, 2, 0 | 0, 1, 1 |
| TF | 1, 0, 0 | 0, 0, 0 |


`main-TT`/`main-EE` above are the **same-input point controls**, without the new render cap; they are not full current-main process measurements. The uncapped new-input EE tour map runs at roughly half rate. Additional archived actual-main `9a4e7ac` EE maps, with the old estimator and original own camera, were rendered through the same uncapped GL harness: **sweep 0, 0, 0; tour 0, 0, 0** misses (p99 7.07–7.11 / 9.47–9.52 ms). This separates the actual-main map baseline from the changed-input point control. It does not establish a full live point-backend regression.

Optimized TT/TE/ET record zero misses in all six runs each. **EE records four, TF one**; all individual results are retained, not replaced by a favourable repeat. Optimized completion p99 is 5.71–8.27 ms, but the isolated EE overruns relative to the archived-main EE control mean the **zero-new-miss requirement is not established**. No sustained overload appears in the bounded optimized renderer; no headset pass follows from that.

Native left/right views preserve the actual 90° selection: [sweep whole eyes](gsplat-oracle-round3-img/sweep180-native-12.0-whole.jpg)/[explicit camera crops](gsplat-oracle-round3-img/sweep180-native-12.0-camera-crop.jpg), [tour whole eyes](gsplat-oracle-round3-img/tour-native-78.0-whole.jpg)/[crops](gsplat-oracle-round3-img/tour-native-78.0-camera-crop.jpg). Rows are point control | TT | TF. Corresponding contours are present, but the TT edge defect and TF surface damage preclude a complete foreground-shape claim. First-second native captures were also inspected and remain in the output; black outside previously observed space is not scored as a regression.

All 18 normal visual runs and 30 loaded native mapper runs complete without OOM, under main's live VRAM manager and **no fixed VRAM cap**. Visual samples retain at least **10,048 MiB device free**. Real impossible-allocation faults after seeding and during refinement each produce exactly one OOM, retain **Half/bf16**, drop work and continue: 87/88 admitted observations and 827/823 optimizer steps. The refinement drive uses three anchor slots to exercise retirement/rollback. The [actual ImGui warning](gsplat-oracle-round3-img/half-oom-warning.png) is captured. These faults do not test genuine system pressure or eviction.

### Verdict and reproduction

**Kill this configuration; do not promote it as the pilot backend.** TT proves early sharpness, but its remaining true-input foreground defect identifies **boundary geometry/optimization** as a mapper bottleneck independent of the estimator. TE/EE add inconsistent-depth seams and incomplete close detail; ET's remaining bent edges show residual pose is secondary, not the sole problem. TF adds a sparse-depth initialization/optimization problem and reaches fewer keyframe iterations. The next mapper work should resolve occlusion boundaries and uncertainty without locking an incorrect initial surface into the map; its expensive sparse addition and rare EE deadline overruns also remain. These results do not establish that physical ToF is useless, or that better tracking alone will fix the map.

`e2e/oracle_estimate.sh` freezes inputs; `e2e/oracle_joint.sh` replays the five arms and four same-input point controls. A second joint run with `ITO_ORACLE_ARMS='TT TE ET EE TF'` and `--mapper-revision 0634bbd` produces the immutable paired baseline. `e2e/oracle_native.sh` runs the 42 native checks. Use `e2e/oracle_audit.py` on `final/after`, then `e2e/oracle_joint_report.py --out OUT/final --stage after --images docs/gsplat-oracle-round3-img`; `oracle_closeup.py --camera CALIBRATION` isolates stored geometry from selection. Each helper's command/arguments are in its header or `--help`. Ceres uses the `feat-recon-40-41` Windows runtime and the existing gsplat binary; no new native build was performed.

The [round-3 manifest](gsplat-oracle-round3-results.json) includes raw/frozen payload and schedule hashes, all confidence-array checks, actual per-cell revisions, before/paired/final iteration counts, memory, all 48 timing runs and image hashes. The committed review is **39 JPEGs plus the warning PNG, under 10 MB**. Complete drives, rejected trials, close-up probes, first-second native captures and exact finishing commands remain under `~/Projects/ito/out/recon-40-41/mapper/`; `NOTES.md` identifies archived failed configurations. The virtual DISPLAY5 stayed 1920×1080; only owned processes were started/closed.

**Untested:** full live estimator + mapper + changing map + OpenXR/compositor/scanout; physical ToF/timing/extrinsics; Half image quality, long sessions and genuine GPU pressure. The owner should check sustained headset deadlines, near-object silhouettes at actual IPD during head translation, floor/edge breathing and fast-turn/reversal comfort. Headset acceptance remains owner-only.
