# Optimised 3D Gaussian splatting: go/no-go (2026-10-12)

The question: Ito's pilot view hits a wall. Its "prediction PSNR" (render the map at the next
camera pose, compare with the real frame) is 21.3 dB on `sweep180` and 19.1 dB on `tour`, even
with true poses (rung L1). Does optimised 3DGS (gsplat) get past that, and can it run online?

Setup: Ceres (RTX 4070 Ti 12 GB, WSL), gsplat 1.5.3 prebuilt wheel, torch 2.4.0+cu124,
Python 3.10; MASt3R extraction with Ito's own runtime on Windows (bundle v4). Recordings:
`work/rf-run/out/{sweep180,tour}` from the PR #37 Ceres drives (sweep180: 890 frames, a 180 deg/s
turn in place; tour: 3022 frames, 10 m walk with head turns), 360x640 frames. Every 8th frame is
held out. gsplat DefaultStrategy, L1 + 0.2 SSIM, SH 0 to 3, a Mip-Splatting-style 3D filter and
antialiased rasterisation. Training time excludes data loading and evaluation. Speed: two
1832x1920 eyes, 90 deg vertical FOV, 64 mm IPD. Scripts: `tools/gsplat-gonogo/` (made to run
from `work/gs/scripts`; `queue_*.sh` record the exact command lines).

## Verdict per GO criterion

| criterion | result | verdict |
|---|---|---|
| held-out PSNR >= 27 dB with true poses, visibly sharp | sweep 33.23 dB (180 s); tour 30.81 dB (180 s) **with scale-corrected init**; tour with Ito's raw MASt3R cloud only 19.24 dB | **GO** if the cloud's scale is right |
| fused (own tracking) poses within 2 dB of true | sweep 28.61 vs 33.23 (-4.6); tour 23.08 vs 30.81 (-7.7), estimated frames only 24.17 vs 28.46 (-4.3) | **NO-GO** |
| >= 25 dB within 30 s of training | sweep 30.12 (27.7 already at 5 s); tour calibrated 29.49 (25.4 at 5 s) | **GO** (true poses) |
| >= 90 stereo pairs/s | gsplat CUDA: 100 pairs/s at ~220k Gaussians (15 s), 80 at 400k, 40 at 1.1M. **Ito's own GL renderer**: 144 at 223k, 109 at 440k, 63 at 1.13M | **GO** at <= ~450k Gaussians (needs a cap) |
| loads in Ito's renderer | unmodified `load_ply` + `GaussianRenderer`: 43.9 dB vs gsplat antialiased, 49.2 dB vs gsplat classic (223k); Ito lacks only the 2D-AA opacity compensation | **GO** |

Causal "predict the next view" score, same targets and coverage masks as Ito's L1 replay
(map from up to 1 s earlier; MASt3R/pose cost not charged, gsplat gets the whole GPU):

| recording | poses | cloud | causal 3DGS median | Ito L1, same targets |
|---|---|---|---|---|
| sweep180 | true | MASt3R (L1) | **26.75** | 21.21 |
| sweep180 | fused | Ito online (fused scale) | 23.58 | 21.21 |
| tour | true | scale-calibrated | **23.29** | 18.94 |
| tour | true | MASt3R (L1) | 18.88 | 18.94 |
| tour | true | Ito online (fused scale) | 20.17 | 18.94 |
| tour | fused | scale-calibrated | 19.70 | 18.94 |
| tour | fused | Ito online (fused scale) | 18.39 | 18.94 |

Ito's own L4 (fused) replay scored 21.53 (sweep) and 19.76 (tour) on its own targets.

**Overall: the representation was a wall, and optimised 3DGS removes it, but only with right
geometry.** With true poses and correctly scaled points it gives +4 to +5.5 dB causally and
30-33 dB offline, fast enough to train and to render. With Ito's current inputs (fused poses,
MASt3R cloud) the gain shrinks to +2.4 dB (sweep) and nothing (tour). The new wall is the inputs:

1. **MASt3R depth scale is wrong in Ito's map.** Against training-only SIFT triangulation with
   true poses (2772 points, 0.26 px median reprojection), the L1 cloud's depths are 4.7x too
   small (p10-p90 3.4-15.6x): in oracle mode `SLAMBackend` passes scale 1.0 to the frontend and
   never folds a metric scale. Ito's fused (L4) cloud is better but still 1.7x off (0.8-4.9x).
   This alone explains why L1 "true poses" scores like bad poses on tour: the points sit at the
   wrong depth, so parallax is wrong. Rescaling MASt3R per frame from triangulation takes tour
   3DGS from 19.2 to 30.8 dB.
2. **Fused poses are not pixel-accurate.** Sweep: rotation error p50 0.29 deg, p95 1.49 deg at the
   estimates (2 / 16 px at fx 533), correlated with turn speed (r=0.67; 0.18 deg below 30 deg/s,
   0.74 deg above 90 deg/s). Not a time offset (best constant offset +2.5 ms, gains 0.04 deg),
   not scale (path ratio 0.986), only a 0.15 deg constant yaw bias. Splitting the 4.6 dB sweep
   gap by re-rendering the frozen model at photometrically refined held-out poses: the fused-
   trained model reaches 32.13 dB vs 33.63 for the true-trained one under the same refinement,
   so ~1.5 dB is a blurrier model and ~3 dB is held-out pose error. On tour the fused poses also
   have to be interpolated between ~10 Hz estimates through head turns (p95 8.9 deg at held-out
   frames); using only estimated frames halves the gap (7.7 to 4.3 dB) and the refined split is
   ~2.4 dB model + ~1.9 dB pose.

## Every run (held-out PSNR/SSIM, every 8th frame)

| recording | poses | init | s | PSNR | SSIM | Gaussians | stereo pairs/s (gsplat) |
|---|---|---|---|---|---|---|---|
| sweep180 | true | MASt3R | 5 / 15 / 30 / 60 / 180 | 27.70 / 29.95 / 30.12 / 31.56 / 33.23 | 0.904 / 0.928 / 0.931 / 0.942 / 0.950 | 42k / 209k / 402k / 636k / 1133k | - / - / - / - / 40 |
| sweep180 | true (rerun 30 s) | MASt3R | 5 / 15 / 30 | 27.31 / 29.64 / 30.59 | 0.901 / 0.925 / 0.934 | 47k / 223k / 400k | 146 / 100 / 79 |
| sweep180 | fused | MASt3R | 5 / 15 / 30 / 60 / 180 | 25.49 / 26.68 / 27.58 / 27.48 / 28.61 | 0.872 / 0.886 / 0.894 / 0.896 / 0.903 | 47k / 193k / 341k / 556k / 1039k | 141 / 103 / 81 / 62 / 39 |
| tour | true | MASt3R | 5 / 15 / 30 / 60 / 180 | 20.77 / 20.66 / 20.76 / 19.95 / 19.24 | 0.825 / 0.826 / 0.827 / 0.818 / 0.805 | 44k / 140k / 262k / 414k / 699k | 223 / 184 / 86 / 51 / 42 |
| tour | true | sparse (triangulated) | 5 / 15 / 30 / 60 / 180 | 22.35 / 26.83 / 27.13 / 27.74 / 26.81 | 0.857 / 0.906 / 0.913 / 0.918 / 0.911 | 6k / 214k / 409k / 644k / 1069k | 491 / 145 / 102 / 71 / 49 |
| tour | true | calibrated | 5 / 15 / 30 / 60 / 180 | 25.42 / 28.29 / 29.49 / 29.98 / 30.81 | 0.890 / 0.923 / 0.932 / 0.938 / 0.943 | 45k / 229k / 440k / 724k / 1329k | 137 / 107 / 83 / 68 / 48 |
| tour | true | Ito online cloud | 5 / 15 / 30 / 60 | 23.40 / 25.77 / 26.13 / 26.17 | 0.865 / 0.903 / 0.911 / 0.916 | 44k / 189k / 379k / 617k | 146 / 122 / 74 / 54 |
| tour | true, estimated frames only | calibrated | 5 / 15 / 30 / 60 | 25.37 / 27.92 / 28.34 / 28.46 | 0.892 / 0.922 / 0.925 / 0.928 | 46k / 237k / 462k / 733k | 147 / 119 / 90 / 74 |
| tour | fused | MASt3R | 5 / 15 / 30 / 60 / 180 | 20.17 / 20.73 / 20.23 / 19.43 / 19.54 | 0.819 / 0.823 / 0.820 / 0.809 / 0.808 | 44k / 128k / 235k / 374k / 631k | 225 / 171 / 82 / 53 / 46 |
| tour | fused | calibrated | 5 / 15 / 30 / 60 / 180 | 22.86 / 23.12 / 22.88 / 22.72 / 23.08 | 0.856 / 0.859 / 0.853 / 0.847 / 0.848 | 42k / 179k / 340k / 539k / 1019k | 128 / 109 / 87 / 70 / 39 |
| tour | fused, estimated frames only | calibrated | 5 / 15 / 30 / 60 | 23.50 / 23.71 / 23.91 / 24.17 | 0.865 / 0.874 / 0.875 / 0.880 | 47k / 222k / 415k / 673k | 151 / 118 / 89 / 66 |
| tour | fused | Ito online cloud | 5 / 15 / 30 / 60 | 21.67 / 22.01 / 21.32 / 20.25 | 0.841 / 0.848 / 0.842 / 0.828 | 45k / 139k / 264k / 411k | 141 / 109 / 64 / 43 |

Init clouds: "MASt3R" = Ito's oracle-pose (L1) MASt3R points from training frames; "sparse" =
SIFT triangulation of training-frame pairs with true poses; "calibrated" = MASt3R points
rescaled per frame by the triangulated/MASt3R depth ratio (interpolated between frames);
"Ito online cloud" = the cloud Ito's fused (L4) backend actually places (fused poses, fused
scale). GPU memory at 30 s is about 1 GB, at 180 s 2.3-3.2 GB.

Pose split (frozen model, every held-out frame; refinement = 150 Adam steps on a 6-DoF delta):

| model | own poses | true poses | refined | refine rot p50 / p95 |
|---|---|---|---|---|
| sweep180 true, 180 s | 33.23 | 33.23 | 33.63 | 0.03 / 0.31 deg |
| sweep180 true, 15 s checkpoint | 29.64 | 29.64 | 30.64 | 0.10 / 0.69 deg |
| sweep180 fused, 180 s | 28.61 | 25.25 | 32.13 | 0.13 / 2.58 deg |
| tour fused calibrated, 180 s | 23.08 | 22.03 | 25.57 | 1.10 / 10.0 deg |
| tour fused calibrated, estimated only, 60 s | 24.17 | 22.82 | 26.07 | 0.90 / 6.14 deg |

## Pictures (Ceres, `C:\Users\me\Projects\ito\work\gs\out\`)

- `tour\true-stream-calibrated\predict-074-frame-001416.png`: real | causal 3DGS 30.8 dB | Ito 17.1 dB
  (Ito's bookshelf is misplaced and smeared, 3DGS matches).
- `tour\true-calibrated\heldout-001512.png` vs `tour\true\heldout-001512.png`: same held-out
  frame, scale-calibrated init (sharp) vs Ito's raw MASt3R init (floaters, unusable).
- `sweep180\true-stream\predict-032-frame-000695.png`: causal 3DGS 25.5 vs Ito 20.8 dB; 3DGS
  is hole-free but soft on the near floor.
- `tour\fused-calibrated-est\heldout-001920.png`: fused-pose training, doubled book edges.
- `sweep180\true-30\checkpoint-15s\ito-match.png`: gsplat | Ito renderer | difference x5.

## What this does not show

- Only simulator recordings; real camera noise, exposure changes and motion blur are untested.
- The causal test does not charge MASt3R, pose fusion or the VR renderer for GPU time, and the
  "calibrated" cloud uses triangulation from training-frame pairs up to 60 frames (2 s) ahead and a scale
  interpolated across frames: an upper bound, not something Ito can run today.
- gsplat stereo speed is the CUDA rasteriser's; Ito's GL renderer was timed on one pose per
  model, without the rest of the app running. Live models grow past 450k Gaussians within
  ~30 s; a cap or pruning is needed to hold 90 Hz.
- Ground truth poses come from `e2e/truth.py` thumbnail matching (sub-degree). The refined-pose
  split assumes 150 steps of 6-DoF refinement does not paint over model errors; the control
  (true-pose 180 s model) gains only 0.4 dB from it.
- No LPIPS. One seed per configuration.

Next steps this points to: fix the cloud scale (fold a metric scale in oracle mode; check the
fused scale estimate, 1.7x off on tour), then make poses pixel-accurate (photometric pose
refinement against the splat map during fast turns, or joint pose optimisation in the 3DGS
loop), then an online gsplat optimiser with a Gaussian budget.
