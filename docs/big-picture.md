# Ito: why the wall, and what should replace it

Review date: 12 October 2026. Code baseline: `fff04390daa945af16ddf7f97b7afc0ae8118b88`.

Ito's core idea is sound: a locally rendered world can follow the pilot's head without waiting for the robot. The present wall has several independent causes. Its live map is a collection of repeatedly replaced, unoptimized, isotropic depth samples. Its depth and camera trajectories disagree at the pixel scale needed to optimize a sharp scene. Its reconstruction work is serialized, and its renderer shares a GPU with that work without a demonstrated deadline budget. A narrow, moving monocular camera also cannot observe everything that two independently moving eyes need.

**Recommendation:** build an optimized Gaussian mapper around a separate, sensor-aided estimator and a renderer with a measured GPU reservation. Preserve observations and uncertainty so the map can improve; distinguish recent, provisional geometry from persistent, optimized surfaces. Do not make alignment against the current blurry map the estimator's foundation. Better sensing and feed-forward Gaussian initialization are serious alternative investments, evaluated below.

## 1. Look first

I ran four offline replays of current main: true-pose L1 and robot/vision-fused L4 for each of `sweep180` and `tour`. I captured map snapshots and rendered them with Ito's actual OpenGL Gaussian renderer. I also rendered the existing offline gsplat checkpoints through that renderer. I opened and inspected all six sheets below, plus the existing online mapper bookshelf comparison. No live drive, desktop input, headset session, or physical-robot experiment was performed.

Every row below has **sim ground truth on the left and the reconstruction on the right**, at the **same true camera pose, resolution and projection**. L4 uses one fixed first-frame heading/position alignment, never a per-image best fit. The optimized sheets show true-pose training first, older fused-pose training second; both are evaluated at the true camera. Open the images at full resolution.

| Comparison | Moments / what to inspect |
|---|---|
| [Main, sweep, true poses](big-picture/img/sweep180-main-L1.jpg) | 7.13, 15.07, 23.07 s: plant, bookshelf, floor |
| [Main, sweep, own tracking](big-picture/img/sweep180-main-L4.jpg) | Same cameras: displaced bookshelf and missing image borders |
| [Main, tour, true poses](big-picture/img/tour-main-L1.jpg) | 12.07, 50.00, 76.10 s: pot, poster, near wall |
| [Main, tour, own tracking](big-picture/img/tour-main-L4.jpg) | Same cameras: poster position, wall/floor junction, overlapping patches |
| [Optimized 3DGS, sweep](big-picture/img/sweep180-optimized.jpg) | 15.07 and 23.07 s, true then older fused training |
| [Optimized 3DGS, tour](big-picture/img/tour-optimized.jpg) | 50.00 and 76.10 s, true then older fused training |

**What I see:**

- Main L1 already loses the books' small white labels and floorboard grain. The poster frame has a scalloped edge; a flat painted wall becomes speckled. The pot and ball remain recognizable but have soft outlines. This is not primarily a tracking failure: true poses were supplied.
- Main L4 adds geometric error. The sweep's final floor has a black missing strip; the middle frame has a missing upper-left patch. At tour 50 s, the poster and bookshelf are displaced relative to truth. At 76 s, a second wall/floor boundary and a vertical patch edge appear. Those are visible inconsistencies, not merely lower texture fidelity.
- Offline true-pose 3DGS restores individual books, labels and a clean poster silhouette. It does **not** reproduce all floor grain; floor colors are patchy, and the near wall at tour 76 s has broad cloudy shading absent from truth. “33 dB” does not mean the scene is finished.
- Older fused-pose optimized tour geometry has stretched shelf edges, a distorted poster and a visibly bent/mottled wall. Optimization can encode inconsistent observations as a plausible-looking but wrong surface. The sweep is less damaged. These old fused checkpoints do **not** measure the newer `fix/pixel-poses` estimator.
- The existing online mapper image `work/mapper/out/r-m2-t-L1/predict-08.png` has roughly correct bookshelf placement but blurred books and floor. It supports the reported convergence problem; it is not evidence that optimized Gaussians have reached their quality ceiling.

These stills establish blur, misplacement, holes and inconsistent surfaces. They cannot establish hitching, popping, temporal shimmer, stereo comfort or motion-to-photon latency. The absence of an obvious hole in one still is not a coverage guarantee.

### How these images were made, and their limits

[Capture script](../e2e/big_picture_capture.py), [render script](../e2e/big_picture_images.py), and [image manifest](big-picture/images.json) preserve the procedure, frame indices, source hashes, Gaussian counts and diagnostic scores. Each script has a command-line invocation in its header. Source recordings/checkpoints remain outside Git; generated raw pairs and replay outputs are in `work/bigpic-out`. No model weights are committed.

The left panel is a new MuJoCo 3.3.7 render of the saved static room `work/truth/data/scan/scene.mjb`, using thumbnail-matched true poses from `work/gs/out/*/data.npz`. This is static-scene ground truth at the matched instant, not a reconstruction of arbitrary historical dynamic state. Both panels use 360×640 and the recorded focal length, with a centered principal point: one pixel different from recorded `cx=179`. Shadows are disabled as in the recorded scene setup. The fresh sim images differ from recorded decoded RGB by **28.1–30.9 dB** at these six moments, owing to rendering/encoding and possible truth-match differences. That discrepancy is disclosed, not subtracted from scores. Exact frame identity still needs a source frame ID in future recordings.

The main maps contain only observations through the selected snapshot; they exclude retiring slots to inspect the persistent surface. Real publication queues, retirement fades and display timing are not reproduced. When L1/L4 capture schedules differ, both are rendered at the later camera instant, retaining each map's available snapshot. The largest mismatch is sweep's first L4 snapshot, frame 181 versus reference 215 (about 1.1 s); later mismatches are at most three frames. Consequently these are diagnostic comparisons, not a controlled L1-versus-L4 quality benchmark. Snapshot compression also perturbs replay scheduling, so their run times are not performance measurements.

The offline models are final **180-second optimized, whole-recording** checkpoints, with roughly **1.0–1.33 million Gaussians**. They contain future observations relative to these screenshots and exceed the earlier 90 Hz budget. They show an achievable appearance bound, not a live result. Mesa llvmpipe rendered the sheets; per-view SH color was evaluated before upload to fit its buffer limit, retaining all SH terms. All graphics/SLAM jobs ran under `ceres.lock`; no shared environment was modified. The four replays used the existing Windows SLAM environment read-only; image generation used an isolated environment under `bigpic-out`.

For orientation only, the full-frame diagnostic PSNRs against the new left panels are below. Do not compare these directly with the old covered-pixel replay scores, nor use their ordering to override the visible artifacts.

| Run, reference frame | Main L1 | Main L4 | Offline GS, true | Offline GS, old fused |
|---|---:|---:|---:|---:|
| sweep 215 | 19.93 | 19.99 | — | — |
| sweep 453 | 21.55 | 17.92 | 25.72 | 22.29 |
| sweep 693 | 26.38 | 15.06 | 27.15 | 24.38 |
| tour 363 | 21.05 | 21.02 | — | — |
| tour 1501 | 23.60 | 18.65 | 26.39 | 22.64 |
| tour 2284 | 26.25 | 23.43 | 26.56 | 23.16 |

Notice tour 2284: a largely blank wall scores well even when blurry. That is exactly why a whole-frame mean cannot stand in for the pilot's experience.
