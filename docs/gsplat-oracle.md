# Oracle mapper: two prove/kill rounds

Current verdict: see [round 2](#round-2--sharper-early-bounded-drawing-configuration-still-rejected). The original report below is preserved as round-1 history.

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
