# Layered camera-view cache — #48

13 October 2026; separate `proto/view-cache` from `07ca8a3`.

**Reject this configuration at stage 2.** It is sharper than main, but its added
source/layer policies do not earn a rewrite over #47. Fixed-camera updates are quieter;
head motion introduces new source switches, mottled walls and floor lighting seams.
Do not use it as a “never black” room layer. The failed gate stops mono and full
resource/continuity testing. Main and the #39 worktree are unchanged.

## Look

Still sheets are **sim truth | frozen main | #47 | view-cache**, with matching
projection/poses and original 1:1 pixels. Start with [sweep](proto-view-cache/img/oracle-sweep.jpg),
[tour](proto-view-cache/img/oracle-tour.jpg), and
[near-wall eyes](proto-view-cache/img/oracle-tour-eyes.jpg).
Also inspect [sweep eyes](proto-view-cache/img/oracle-sweep-eyes.jpg),
[hard turn](proto-view-cache/img/hard-526.jpg),
[bookshelf head motion](proto-view-cache/img/bookshelf-head-motion.jpg), and
[wall head motion](proto-view-cache/img/wall-head-motion.jpg).
The last filmstrip has truth/#47/cache; frozen main has no tour temporal clip.

I opened the sheets, raw eye PNGs, both filmstrips and the actual desktop stereo capture.
Both image methods preserve individual books and white labels that main smears. The
cache's floor becomes a mosaic of differently lit source images. Tour's near wall is
covered in brightness islands absent from truth and mostly absent from #47. Near the
pot, the two eyes still expose missing strips and broken foreground edges. In the
frozen bookshelf filmstrip, moving the head changes a relatively clean wall into mottled
patches at 1.5 s, then changes it back: the cache received no new images. At the wall,
source seams move with the head. The hard turn stays black until it has actually been
observed. Higher texture detail does not make these transitions comfortable.

These are my offscreen eye checks, not owner preference or headset acceptance.

## Method and variants

`ito/proto/viewcache` accepts calibrated RGB, axial depth in metres, rigid
world-from-camera pose, capture time and frame identity. It contains no robot-specific
logic. Original RGB is the appearance representation; no Gaussian training, permanent
fused surface or retirement mask. Every view retains its original timestamp.

The tested configuration has **64 retained views, 12 active candidates, two midpoint
layers, 0.15 ownership hysteresis**, full **360×640** source RGB and the same **90×160**
projective grid/discontinuity rejection as #47: **2 cm + 8% of nearest depth**. Both
candidates get identical geometry resolution, poses, depths and publication times.

Retention evicts the most redundant viewpoint using angular separation, position and
image sharpness, preserving isolated coverage rather than the latest 64 frames. At
sweep's end it still retains source 0; the latest-64 queue would start at 312. Tour
retains source 258, versus 1902 for the latest-64 queue. Candidate ranking combines
view angle, distance and a capped age penalty. A spatially smoothed RGB Laplacian is a
blur score, not calibrated confidence.

Four GL passes select the nearest supported depth, choose its source by quality,
find an observed background beyond **2 cm + 1.5%** range, and select that source.
Ownership cost includes border support, sharpness, angle, distance and age. Previous
ownership is reprojected through the old midpoint camera and receives the hysteresis
bonus. Both eyes share these choices but depth-test independently. Color is sampled
from original mipmapped RGB. There is no cross-source blending or exposure correction.
The second layer is small in these views: **0.55% sweep / 0.26% tour** midpoint coverage
at the final reference snapshots; retaining it does not supply unobserved background.

RGB and depth use separate bounded publication calls. A late depth result for an evicted
source is ignored. Without verified correspondence, pending RGB supplies no scene
geometry. **Flow-based RGB propagation and a measured RGB fast path were not built**
because stage 2 failed. No freshness gain is claimed by merely storing RGB sooner.

The oracle variant uses #47's prepared simulated depth and true source poses. #47's
true-depth/true-pose retirement policy is the matched reference. The frozen main remains
its original fused baseline. No learned-model variant or robot-pose run follows the
failed gate; #47's learned caches remain untouched. The best tested cache is the oracle
configuration above; #47 wins the appearance/stability comparison overall.

## Results and gates

Six held-out eye images per sequence at configured **64 mm IPD**; hard #526 is separate.
All seven moments additionally have camera and ±3 cm views: **35 scored images**.
PSNR/SSIM/AlexNet LPIPS are the unchanged baseline evaluator and include black holes.
The [results JSON](proto-view-cache/results.json) retains every score, publication,
input hash, temporal diagnostic and audit count.

| Representation | Sweep PSNR / SSIM / LPIPS | Tour PSNR / SSIM / LPIPS |
|---|---|---|
| Frozen main | 17.33 / .734 / .524 | 18.92 / .778 / .513 |
| #47, true depth/pose | 19.74 / .836 / .245 | 23.33 / .878 / .271 |
| View cache, true depth/pose | 19.47 / .827 / .291 | 22.49 / .852 / .357 |

| Representation | Sweep coverage / seen holes | Tour coverage / seen holes |
|---|---|---|
| Main | 95.16% / .54% | 98.78% / .00% |
| #47 | 94.46% / .91% | 97.35% / .90% |
| View cache | 94.22% / 1.12% | 96.54% / 1.72% |

Stage 1 uses the issue's **≥2 dB OR ≥20% LPIPS** threshold, plus visible labels/edges.
The stricter research proposal's coverage limits are already missed; the requested
oracle numeric gate nonetheless passes on each sequence.

Temporal shimmer uses the baseline's p95 mean absolute change of candidate-minus-truth
residuals between consecutive frames, plus changed edge-mask fraction. Each clip is
61 frames at 30 Hz over two seconds. This is a diagnostic, not a comfort score.

| Already-observed clip | Residual p95 #47 → cache | Edge-change p95 #47 → cache |
|---|---|---|
| Bookshelf, frozen cache/moving head | .00907 → .01078 (**19% worse**) | 3.65% → 3.77% |
| Bookshelf, fixed head/cache updates | .000708 → .000169 (**76% better**) | .250% → .000% |
| Bookshelf, both | .00965 → .01697 (**76% worse**) | 3.53% → 3.69% |
| Tour wall, frozen cache/moving head | .00558 → .00796 (**43% worse**) | 1.68% → 3.05% (**81% worse**) |
| Tour wall, fixed head/cache updates | .00522 → .00167 (**68% better**) | .555% → .161% |
| Tour wall, both | .00649 → .00831 (**28% worse**) | 1.88% → 3.06% (**62% worse**) |

Mean clip-p95 residual change across the three observed modes is **44% worse on sweep,
4% worse on tour**. Mean clip-p95 edge change is approximately unchanged on sweep and
**52% worse on tour**. The fixed-head gains cannot compensate for added head-motion
shimmer. Tour also loses **0.84 dB**, with **32% worse LPIPS** and **0.82 point more seen
holes** versus #47, exceeding research's 0.5 dB / 0.5 point retention limits.

The three hard-turn clips are retained separately. Its frozen unknown view is black
in both candidates, so equal zero edge change does not earn a stability gain. All nine
raw clip arrays, traces, normal/3× slow MP4s and filmstrips are in
`work/p48-out/clips/truth-oracle/{sweep180,tour}`. Tour's moving-head clip freezes the final
prefix; its update clips approach that prefix from two seconds earlier. Sweep uses the
baseline's exact head trajectory and decoded truth/main clip panels. Tour truth is
rerendered from the same saved static room; missing main tour clips are explicitly null.

| Stage | Verdict |
|---|---|
| 1 — Oracle | **PASS requested gate:** +2.13/+3.57 dB, 44.4%/30.4% LPIPS reduction vs main; labels visibly sharper. |
| 2 — Added value over #47 | **FAIL:** head-motion shimmer regresses; image quality/coverage regress; no ≥20 ms measured freshness alternative. |
| 3 — Mono and near-field stereo | **SKIPPED at the failed stage.** No robot-pose/learned-depth metrics or stereo-error-halving claim. |
| 4 — Continuity, display/memory | **SKIPPED.** Offline prefix audit passes, but full continuity, <0.1% display misses and <8 GB aggregate residency are not certified. |

## Reproduction, timing and limits

The frozen reference audit verifies **3,925 original input hashes**. The cache audit
checks **503 consumed prepared inputs, 35 snapshot views, 549 clip deadlines and 35,136
retained-source memberships**, with zero future observations/publications. These are
oracle intervention traces and legacy source-clock availability assumptions, not live
causal pose estimation. Cache membership remains ≤64.

#47 inputs, recordings, worlds, depth caches and code at `932c7d6` were read only.
The temporal runner imports its renderer with a pinned SHA256 solely to produce new
matched temporal evidence, without repeating #47 inference or its evaluation sweep.
Both policies receive #47's exact queued publication times, including its declared
10 ms update budget; this does not establish measured native publication latency.
Target-eye RGB never enters the cache. The legacy thumbnail-matched truth and one-pixel
original principal-point discrepancy remain inherited limitations.

Hardware captures ran in an isolated Windows venv reading baseline dependencies,
with hidden offscreen NVIDIA GL, under `ceres.lock`, in separate <1,100-second jobs.
No depth model was resident. All hardware work finished before 10:42 CEST, leaving the
GPU free for #39. LPIPS, truth images and the desktop driver use the private CPU/Mesa
environment. The inspector reached about 5 FPS on llvmpipe: it verifies controls,
not 90 Hz. The handful of fenced hardware snapshot samples are not a timing distribution
or runtime-resolution certificate; some selection+stereo samples already exceed 11.11 ms.

Allocated image payload is bounded by a nominal **168.75 MiB**, plus about **22 MiB** midpoint
attachments and geometry at this source resolution, excluding context/driver overhead.
This is arithmetic, not aggregate measured VRAM. No weights or bulky outputs are in git.
Dynamic contradiction/invalidation, misregistration confidence, native freshness,
reconnection/loss, physical calibration, full-resolution concurrent 90 Hz and headset
acceptance remain unestablished. They are stopped work, not product promises.

From this worktree, revalidate archived evidence and drive the inspector with one command:

```sh
bash e2e/viewcache_run.sh --cached-only
```

For a fresh private environment: `bash e2e/viewcache_setup.sh`. Then
`bash e2e/viewcache_run.sh` regenerates oracle stills and matched clips on Mesa, scores,
audits, and drives the inspector. It stops before temporal rendering if stage 1 fails;
it contains no stage-3/4 continuation. GPU capture commands, when needed, are separate:

```sh
bash e2e/viewcache_gpu.sh e2e/viewcache_render.py --work C:/Users/me/Projects/ito/work --out C:/Users/me/Projects/ito/work/p48-out
bash e2e/viewcache_gpu.sh e2e/viewcache_clips.py --work C:/Users/me/Projects/ito/work --out C:/Users/me/Projects/ito/work/p48-out
```

Fly a saved cache with the real Ito desktop input and Dear ImGui:

```sh
../p48-out/venv/bin/python -m ito.proto.viewcache ../p48-out/rendered/truth-oracle/tour-world.npz
```

WASD, PgUp/PgDn, Tab mouse-look, Home recenter, F12 capture, ImGui 64 mm stereo and FOV.
The Xvfb driver passed keyboard movement, recentering, screenshots and the stereo control
in 80 display frames. Ruff and whitespace checks pass. This is saved-cache desktop
inspection, not live robot/VR integration.

## Licences and decision

No new pretrained depth model or upstream novel-view implementation is imported.
The renderer implements image-cache/layer selection directly, reusing Ito/#47 grid ideas.
[ModernGL](https://github.com/moderngl/moderngl/blob/main/LICENSE) is MIT;
[OpenCV 4.11](https://github.com/opencv/opencv/blob/4.11.0/LICENSE) is Apache-2.0;
[LPIPS](https://github.com/richzhang/PerceptualSimilarity/blob/master/LICENSE) is BSD-2-Clause.
LPIPS/AlexNet weights are evaluator-only and remain outside git. Existing Ito, MuJoCo,
UI and runtime dependency terms still apply; no new redistribution clearance is claimed.

Keep this as a rejected isolated experiment. It shows a real benefit for fixed-head map
updates, but quality-based per-pixel ownership must address lighting and view-dependent
switches before it is comfortable under head motion. Oracle geometry does not cure those
appearance seams. This result rejects the tested configuration, not every layered-image
method. Prefer simpler #47 over this cache for the same geometry; neither establishes a
truthful, comfortable never-black room. Owner/headset judgment remains open, and no
integration with the overhaul is warranted by this evidence.
