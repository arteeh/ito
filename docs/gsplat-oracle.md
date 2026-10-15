# Oracle mapper: kill this configuration

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
