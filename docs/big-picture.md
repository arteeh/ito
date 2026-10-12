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

The offline models are final **180-second optimized, whole-recording** checkpoints, with roughly **1.0–1.33 million Gaussians**. They contain future observations relative to these screenshots and exceed the earlier 90 Hz budget. These selected frames are not a new held-out set and can overlap training views. They show an achievable appearance bound, not a live or novel-eye result. Mesa llvmpipe rendered the sheets; per-view SH color was evaluated before upload to fit its buffer limit, retaining all SH terms. Replay capture and comparison-sheet generation ran under `ceres.lock`; no shared environment was modified. The four replays used the existing Windows SLAM environment read-only; image generation used an isolated environment under `bigpic-out`.

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

## 2. The actual walls

### Input: some missing information cannot be recovered by computation

**Field of view and visibility.** A monocular camera supplies one ray color per pixel, not a stereo light field. Pure rotation exposes new directions without triangulating depth. Translation exposes previously hidden surfaces; no representation can recover their true appearance before observation. Learned models can supply plausible priors, not measurements. This remains true with perfect tracking and unlimited optimization.

The recordings are **360×640, fx=fy=532.57**: approximately **37.3° horizontal × 62° vertical**, derived from their calibration. That is substantially narrower than a typical headset view and different from the physical camera's described “~90°.” IMX219 names a sensor, not its lens/crop. Calibrate the actual rectified physical camera before designing around that figure. Rotating the pilot's head outside the observed cone necessarily reveals remembered, uncertain or unobserved space. A pan/tilt head improves coverage only after mechanical motion and capture arrive.

**Sampling, exposure and rolling shutter.** At this focal length, one degree is about **9.3 source pixels**. At 90°/s, a 10 ms exposure integrates about **8.4 pixels** of rotational motion; at 180°/s, **16.7 pixels**. A sharper mapper cannot restore information that was never sharply sampled. Short exposure trades signal for noise, so lighting, gain and compression need measurement together. Rolling shutter additionally means one rigid pose does not describe every row: a hypothetical 10 ms readout during 180°/s spans 1.8°. The physical readout time is **not measured here**. Treating the camera as instantaneous is a model error; finite light and bandwidth are physical limits.

Main already predicts blur and skips visual work above eight pixels ([slam.py:126](https://github.com/arteeh/ito/blob/fff0439/ito/reconstruction/slam.py#L126)). Skipping can be correct for estimation but leaves the newest scene unobserved during exactly the turns when new content enters view. Reporting those pose-only updates as reconstruction throughput hides this distinction. Autoexposure also violates constant brightness; an affine exposure model helps brightness, but cannot undo clipping or blur.

**ToF is valuable metric evidence, not dense depth.** ST specifies the VL53L5CX at up to **15 Hz in 8×8**, versus up to 60 Hz in 4×4, a 65° diagonal square FoV and ranges up to 4 m under specified conditions. An approximately 45° side spans about 5.6° per zone: roughly 52 source-camera pixels near the center, or 10 cm across at 1 m. Associate a range with its calibrated angular footprint and return quality; do not upsample 64 measurements into invented object boundaries. It can constrain scale and coarse surfaces, especially when the floor disappears. Coverage, occlusion and timing between ToF and RGB still matter. [ST datasheet](https://www.st.com/resource/en/datasheet/vl53l5cx.pdf)

**Depth inconsistency is not merely a scalar.** The earlier gsplat investigation measured a 4.7× scale error; the coordinator's plan also reports 10–35% depth variation, a statistic I have not remeasured on `fff0439`. `d91e443` corrected a major scale defect, not all shape disagreement. Neighboring frames can disagree about surfaces after global rescaling. Main's `floor_scale` searches a histogram of downward point heights and divides known camera height by its dominant mode ([slam.py:227](https://github.com/arteeh/ito/blob/fff0439/ito/reconstruction/slam.py#L227)). It does not independently identify a floor plane. A table, steps, sloped ground, an elevated platform, a held robot or no floor can invalidate the premise. The scale gauge of unconstrained monocular vision is fundamental; assigning arbitrary robot-world Y the meaning “height above a visible floor” is our fixable mistake.

### Estimation: centimeter-level robotics accuracy can still be visibly wrong

A 1 cm lateral error at 1 m is **5.3 source pixels**. One degree of rotation is about **16.8 pixels** in a hypothetical 1920-pixel-wide, 90° eye view. A 64 mm stereo baseline gives about 34 source pixels of disparity at 1 m; a 20% depth overestimate changes that by about **5.7 pixels**, even at the original capture location. Near objects are less forgiving. Accordingly, 2°/5 cm “usable” thresholds in [truth.py:24](https://github.com/arteeh/ito/blob/fff0439/e2e/truth.py#L24) are much too loose to imply a sharp stereo map. Absolute room alignment matters less than consistency between observations and between each eye and the map.

**Timing is geometry.** The Microduck camera adapter explicitly stamps **receipt**, because mediad does not expose per-frame sensor capture time ([camera.py:85](https://github.com/arteeh/ito/blob/fff0439/drivers/microduck/camera.py#L85)). A 10 ms error at 180°/s produces the same 1.8° discrepancy above. Link clock-offset estimation cannot recover an unknown exposure time inside the camera pipeline. A fitted constant delay is useful, but variable buffering, exposure changes and rolling shutter need their own observations/uncertainty. Robot joint timing, head-camera extrinsics and IMU location also matter; body IMU orientation alone is not the moving head camera's pose.

Robot-first tracking is the right foundation: fuse a fast motion prior with visual evidence, rather than making the pilot wait for dense inference. It is not a license to treat odometry as truth, or to count raw IMU and an odometry estimate derived from that same IMU as independent information. Inertial/visual fusion remains useful for bias, velocity, metric scale under excitation, calibration and uncertainty—not just rotation.

The `fix/pixel-poses` improvement is real but costly in the reported runs: the [merge record](https://github.com/arteeh/ito/commit/fff0439) reports rotation p95 falling from 3.85° to 2.06° on live sweep, and every-frame tour error from 7.72° to 1.71°. I checked `work/pp-run/out/live-{m,p}-{sweep,tour}/summary.json`: published tracking rates fell **5.76 → 4.18/s** on sweep and **6.42 → 3.32/s** on tour; tour capture-to-scene median rose **238.65 → 294.51 ms**. Those are different measurements, not a single “better tracking” verdict. Also, [fusion.py:276](https://github.com/arteeh/ito/blob/fff0439/ito/reconstruction/fusion.py#L276) can interpolate between surrounding corrected poses and revise past keyframes. An every-frame retrospectively queried trajectory is not necessarily the trajectory the pilot had available at each deadline. Score both explicitly.

### Representation: the live map throws away the evidence needed to become sharp

A naming correction: **`fusion.py` is the pose graph; `rgbd.py` contains the point-splat map.**

The live splats are Gaussian primitives, but they are **not an optimized 3DGS scene**. [slam.py:195](https://github.com/arteeh/ito/blob/fff0439/ito/reconstruction/slam.py#L195) chooses a roughly three-pixel sampling footprint. [rgbd.py:154](https://github.com/arteeh/ito/blob/fff0439/ito/reconstruction/rgbd.py#L154) rounds cells to power-of-two levels, selects one point per cell, and overwrites its position/color. It fixes opacity to 0.85, assigns equal XYZ scales of 0.65 cell widths, and uses SH0 color. No multi-view image loss fits centers, orientation, anisotropy, opacity or appearance. Broad isotropic footprints average across edges; power-of-two changes and noisy depth can move samples between levels and slots. The new L1 images show the resulting loss directly.

[Carving](https://github.com/arteeh/ito/blob/fff0439/ito/reconstruction/rgbd.py#L109) retires older surfaces seen through or replaced by a new depth image. With inconsistent depth, this can erase a correct surface or alternate between incompatible ones. Fades hide a discontinuity but do not resolve it. Conversely, out-of-view or occluded surfaces persist until pressure/contradiction: the current implementation is **not** a uniform “everything fades with age” model. A single recency rule is inadequate for a static bookshelf, a newly seen doorway and a moving person.

The [earlier gsplat investigation](https://github.com/arteeh/ito/blob/13c2873/docs/gsplat-gonogo.md) demonstrates substantial representation headroom: true-pose optimized results of **33.23/30.81 dB** versus substantially weaker point maps, with a **4.6–7.7 dB** loss using older fused poses. But the often-repeated **21.3/19.1 dB** baseline used a simpler covered-pixel point renderer and included the old depth-scale problem. It is evidence of a weak end-to-end pipeline, **not a clean proof that representation alone accounts for the whole gap**. True pose plus estimated depth is not a true-geometry oracle. Our new true-pose sheets still independently establish a serious appearance problem after the scale fix.

The online mapper results sharpen this diagnosis. I extracted these values directly from the existing `r-m2-*` summaries; [the evidence file](big-picture/mapper-evidence.json) includes their source hashes:

| Experiment | Covered PSNR, main → mapper | Full-frame PSNR, main → mapper | Coverage, main → mapper |
|---|---:|---:|---:|
| sweep, L1 | 22.19 → 24.51 | 20.46 → 21.08 | .988 → .984 |
| tour, L1 | 21.50 → 22.05 | 19.20 → 19.19 | .991 → .988 |
| sweep, L4, no map alignment | 21.87 → 22.49 | 20.44 → 17.67 | .991 → .969 |
| tour, L4, no map alignment | 21.28 → 20.49 | 18.41 → 18.11 | .991 → .984 |

These historical runs have different frame schedules and are not our new baseline; their pattern is nevertheless informative. “+2.3 dB” obscures coverage and full-image behavior. Tour's mapper without alignment has every-frame rotation p95 **1.40°**; with alignment it becomes **5.17°**, translation p95 **2.8 → 20.5 cm**, and full-frame PSNR **18.11 → 15.54 dB**. A blurry immature map supplies biased gradients. Matching a camera to that map can make the estimator and wrong map agree more closely while moving both away from reality. Approximately ten training steps per new keyframe are not demonstrated to produce an alignment-quality map.

### Compute and latency: asynchronous processes do not reserve a GPU

The display/reconstruction process split is good. Inside reconstruction, [SLAMBackend.integrate](https://github.com/arteeh/ito/blob/fff0439/ito/reconstruction/slam.py#L107) is serial: frontend tracking, delay/scale fitting, pose/window solve, corrections, map placement. The worker publishes external pose and scene updates **after** integration returns ([worker:100](https://github.com/arteeh/ito/blob/fff0439/ito/reconstruction/__init__.py#L100)). A heavy mapping or keyframe step can delay otherwise cheap pose publication. The worker's “integrated” counter includes calls that only update a pose; it does not mean a fresh, sharp surface reached the pilot.

Nor do all Gaussians stay on the GPU: `rgbd.py:181–192` transfers matching results, positions and colors back to NumPy; carving transfers live centers to the GPU again. Shared-memory deltas then pass through CPU staging into GL. This may be acceptable at a bounded update rate, but it contradicts the architectural shorthand and must be timed. Replacing it with direct interop is only justified after a trace identifies transfer/synchronization as the bottleneck; Windows CUDA/GL interoperability is an integration cost, not a free optimization.

At 90 Hz the complete display frame has **11.11 ms**. Earlier isolated GL measurements of 144 Hz at 223k and 109 Hz at 440k correspond to **6.94 and 9.17 ms**. The latter leaves only **1.94 ms** before the deadline, even before allowing for other GPU work and compositor overhead. A mapper thread sleeping to average 25% duty does not guarantee its kernels never block a display deadline. Neither CUDA streams nor separate processes establish that guarantee. Conversely, a 200 ms inference step is not proof of a 200 ms display stall; only a shared-load GPU/display trace can establish the contention.

[The renderer](https://github.com/arteeh/ito/blob/fff0439/ito/render/renderer.py#L113) bitonic-sorts allocated capacity each frame, not just visible occupied splats. It shares a mid-eye sort for both eyes. Cost therefore depends on capacity, overdraw and viewpoint as well as the advertised Gaussian count. Live updates are 64-byte SH0 records ([renderer.py:68](https://github.com/arteeh/ito/blob/fff0439/ito/render/renderer.py#L68)); loading an SH3 PLY does not prove the incremental live path supports the same model. AA opacity compensation is worthwhile, but cannot explain or cure the several-pixel book blur in main.

Finally, local 90 Hz head rendering and scene freshness are different latency budgets. At 0.4 m/s, 294 ms means 11.8 cm of robot travel. Reprojection can compensate camera motion through known static geometry; it cannot make an unobserved obstacle or moving object current. [AnchorGlide](https://github.com/arteeh/ito/blob/fff0439/ito/render/anchor.py#L1) intentionally adds roughly one update interval to smooth translation. That explains a useful comfort improvement and an additional freshness cost, both of which should be measured.

### Architecture: the policy is underspecified, not the number of modules

The present boundaries—driver, transport, reconstruction worker, renderer—are a good skeleton. The missing contracts are **when an estimate became available, how uncertain it is, which observations support a surface, and which pose/map versions belong together**. Keyframe ownership allows corrections, but does not decide what survives keyframe retirement or how obsolete updates are rejected. A global trajectory correction should not accidentally rotate the entire visible world under a stationary pilot: distinguish local continuity from global consistency and publish coordinate changes coherently.

The wire contract still assumes motion for the **first camera**, and requires its Y coordinate to be height over the floor, with states `standing/walking/turning/held/fallen` ([protocol:205](https://github.com/arteeh/ito/blob/fff0439/ito/protocol/__init__.py#L205)). Those are legged-robot assumptions, not a general contract for drones, boats or independently moving cameras. Sixteen camera descriptors do not create a multi-camera estimator. Camera exposure is static description metadata; sensor-frame IDs, per-frame timing/calibration and optional range/IMU observations need explicit semantics as they are implemented. A generic named-frame observation contract is a small change in principle; a generic factor framework for every hypothetical sensor would be unnecessary infrastructure.

**Fundamental:** visibility, monocular gauge/degeneracy, sampling/blur, finite shared compute, delayed information about moving things. **Fixable mistakes:** unstable depth insertion without consistency checks, unoptimized appearance, arrival-time geometry, floor assumptions in the core contract, serial publication, weak resource accounting, and evaluation that confuses pose updates, surface updates and display frames. No single model replacement removes all six categories.

## 3. What the research actually offers, 2024–2026

This is a selection for Ito's constraints, not a claim to catalogue every paper. Reported numbers below are **authors' experiments or explicitly identified cross-paper comparisons**, not measurements on Ito. Tracking FPS, inference throughput, mapping FPS and isolated rendering FPS are different quantities. None establishes 90 Hz stereo while tracking and optimizing on our consumer GPU. Unless stated otherwise, native Windows integration and coexistence with Ito's GL/OpenXR renderer remain unverified; the research implementations are CUDA/C++/PyTorch integrations to evaluate, not interchangeable Python packages.

### Online mapping and SLAM

| Method | Reported result and useful idea | Fit for Ito |
|---|---|---|
| **MonoGS**, CVPR 2024 | Gaussian-only photometric tracking/mapping, mono/stereo/RGB-D. Original project reports about 3 FPS; repository's speedup branch reports up to **10 FPS**, TUM fr3/office, RTX 4090/i9-12900K. [Project](https://rmurai.co.uk/projects/GaussianSplattingSLAM/), [implementation](https://github.com/muskie82/MonoGS) | Useful optimization/reference design; risky primary tracker for a waddling camera and an immature map. Authors recommend global shutter and avoiding aggressive initial motion. The 4090 result leaves our shared-GPU budget unresolved. |
| **Photo-SLAM**, CVPR 2024 | Feature-based localization and Gaussian appearance learning run separately; geometric densification, pyramids, loop corrections. Replica monocular: **33.30 dB**, **41.65 tracking FPS**, **911.26 rendering FPS**, **6 GB** on RTX 4090, 1200×680 images. [Paper, Table 1](https://arxiv.org/html/2311.16728v2) | Strongest architectural reference for proposal 1. Robot/IMU priors can supplement tracking. Supports several camera modalities, but a substantial C++/CUDA integration; sparse initialization and texture-poor near-floor views need testing. The rendering number is not a VR scheduling guarantee. |
| **SplaTAM**, CVPR 2024 | RGB-D Gaussian tracking/mapping with explicit silhouette/coverage reasoning. In GlORIE's comparison, **0.14 FPS, 18.54 GiB**, Replica room0, RTX 3090; those are comparison values, not a universal SplaTAM rate. [Method](https://spla-tam.github.io/), [comparison, Table 7](https://arxiv.org/html/2403.19549v3) | Good reference for coverage-aware admission. Not a drop-in mono/8×8-ToF solution or a demonstrated fit under 8 GB. Dense RGB-D assumptions are substantial. |
| **GS-ICP SLAM**, ECCV 2024 | Shares Gaussian/covariance geometry between G-ICP tracking and appearance mapping; reports **up to 107 FPS for the entire system**, RTX 4090/7800X3D. [Paper](https://arxiv.org/html/2403.12550v2), [conference version](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05296.pdf) | Strong reference when dense metric depth exists. Avoids treating a blurry rendered texture as the sole tracking reference. Applicable to sim oracle experiments and a sensor upgrade; 64 ToF zones do not satisfy its RGB-D input assumption. |
| **RTG-SLAM**, SIGGRAPH 2024 | Separates stable and unstable Gaussians; local optimization, opaque geometry and transparent residual appearance; ICP plus a graph backend. Azure Kinect large rooms: around **16 FPS**, including **17.9 FPS / 8.8 GB** on the home scene, RTX 4090. [Paper](https://arxiv.org/html/2404.19706v2) | Borrow stable-surface freezing and local work allocation. The reported home configuration exceeds this review's 8 GB limit before Ito's renderer; it also needs dense depth. Do not adopt the whole stack on the strength of “real-time.” |
| **GlORIE-SLAM**, ECCV 2024 | RGB-only, globally deformable **implicit-encoding neural point cloud**, not a Gaussian-splat renderer. Joint depth/scale/pose optimization; **0.23 FPS / 15.22 GiB**, Replica room0, RTX 3090. [Paper, Table 7](https://arxiv.org/html/2403.19549v3) | Important consistency lesson: correct geometry and anchored map deformation together. Poor live deployment fit. Its name appearing in Gaussian discussions should not make it a candidate Gaussian renderer. |
| **MASt3R-SLAM**, CVPR 2025 | Dense learned reconstruction priors with matching and graph optimization; paper reports **15 FPS**. Ito's observed rate is lower and workload-specific. [Paper](https://arxiv.org/abs/2412.12392) | Useful current depth initializer, matching fallback and offline reference. It should not dictate the only visual-update clock. It does not supply independently measured metric scale or pixel-consistent depth by default. Existing Windows integration is an advantage; licence is a deployment constraint. |

### Feed-forward geometry and Gaussian prediction

| Method | Reported result / distinction | Fit for Ito |
|---|---|---|
| **pixelSplat / MVSplat**, CVPR/ECCV 2024 | Two posed views predict Gaussians; MVSplat adds explicit matching cost volumes. MVSplat's comparison at **256×256** reports **104 ms / 25.89 dB** for pixelSplat and **44 ms / 26.39 dB** for MVSplat on RealEstate10K; timing includes encoder and rendering. [Paper, Table 1](https://arxiv.org/html/2403.14627v2) | Small, concrete initializer candidates. Need usable baseline and consistent poses; pure pan and mostly unseen space remain difficult. Preserve portrait intrinsics when resizing/padding. A 256² result is not headset texture quality. |
| **DepthSplat**, CVPR 2025 | Combines multi-view matching with monocular features; predicts depth and Gaussians. Two 512² views on A100: **small 91 ms / 3.1 GB**, base 155 ms / 5.4 GB, large 313 ms / 8.4 GB. Small at 256²: 65 ms / 1.4 GB. [Paper, Appendix A](https://arxiv.org/html/2410.13862v2) | First learned initializer I would trial, alongside MVSplat. Addresses cold-start optimization and texture-poor depth better than raw per-frame blobs. Use a small model at keyframe rate, retaining robot-derived poses/metric constraints. A100 timings and source-domain performance require local validation. |
| **AnySplat**, SIGGRAPH Asia 2025 | Infers poses and Gaussians from unposed views. Mip-NeRF360: **3 views 0.171 s / 16.20 dB; 16 views 0.767 s / 21.85 dB; 32 views 1.4 s / 22.31 dB**. Timing hardware was not established in this review. Its metric evaluation uses target images for pose/scale alignment. [Paper](https://arxiv.org/html/2505.23716v2) | Useful offline initializer/recovery challenger, not evidence of causal metric tracking. It discards some of Ito's already-known calibration/prior advantages unless adapted. Window stitching, scale stability, near-floor domain shift and moving subjects remain open. |
| **Long-LRM**, 2024/2025 | Many **posed** views to a Gaussian scene: **32 images at 960×540 in 1.3 s on A100 80 GB**. [Paper](https://arxiv.org/abs/2410.12781) | Interesting background submap consolidation, not a 5–30 FPS live path on our budget. An 80 GB GPU model is not evidence that its actual inference footprint is 80 GB, but fit under 8 GB has not been shown here. |
| **VGGT / π³**, CVPR/NeurIPS 2025 | Feed-forward geometry/poses, not inherently optimized Gaussian maps. π³ reports **57.4 FPS** versus VGGT **43.2 FPS**, KITTI on **A800**, with 959M versus 1.26B parameters. [π³ paper, Table 5](https://arxiv.org/html/2507.13347v1), [VGGT](https://github.com/facebookresearch/vggt) | Batch throughput does not establish causal sliding-window latency, persistent map consistency or consumer-VRAM fit. Use as offline geometry teachers or bounded recovery candidates. They are not justified replacements for every incoming camera frame. |
| **Flash-Mono**, ICLR 2026 | Recurrent feed-forward Gaussian/pose inference, 2D Gaussian surfaces and loop correction; proceedings report an **order-of-magnitude efficiency improvement**. Absolute FPS, hardware, public implementation and licence were not independently established here. [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5fcacc84d87c97bee5baecb3fb8f6e88-Abstract-Conference.html) | The direction directly addresses repeated cold starts and inter-frame scale inconsistency. A research lead to reproduce, not a dependency to promise. Keep the independent estimator and explicit metric observations even if recurrent mapping wins. |

Feed-forward initialization is compatible with the owner's request for real optimized Gaussian splatting: initialize the full Gaussian parameters quickly, then fit them against actual observations. It is **not** a reason to accept a plausible predicted room as observed reality.

### Estimation and surface geometry worth borrowing

**DPVO first, DROID as a measured challenger.** DPVO predates 2024 but remains pertinent; its paper reports **60 FPS / 4.9 GB**, minimum 48 FPS, on RTX 3090, or **120 FPS / 2.5 GB** for its fast EuRoC configuration. The same source reports DROID-VO at **40 FPS / 8.7 GB**. DPV-SLAM's 2024 extension reports a bounded **5–7 GB** footprint and 1–4× real-time performance. These are promising tracking alternatives, not dense map generators. Adding metric robot factors, delay uncertainty, recovery and Windows extension builds is real work. Do not assume an unmeasured “DROID frontend, 2–3 GB” will fit. [DPVO paper](https://arxiv.org/html/2208.04726v2), [DPV-SLAM paper](https://arxiv.org/abs/2408.01654)

**Visual-inertial fusion deserves an actual baseline.** OpenVINS explicitly estimates camera/IMU extrinsics, temporal offset, sensor biases and intrinsics; Basalt is another reusable estimator. Correctly sampled raw gyro/accelerometer data, calibrated transforms, and sufficient motion excitation are prerequisites. A 50 Hz SDK pose stream is a prior, not a substitute for raw inertial measurements. Neither engine automatically solves the moving pan/tilt camera's calibration. A CPU visual-inertial estimator could also free the GPU for appearance. This review establishes no Microduck FPS or accuracy for either; run a recorded physical sensor sequence before selecting one. [OpenVINS calibration](https://docs.openvins.com/gs-calibration.html), [Basalt repository](https://github.com/VladyslavUsenko/basalt-mirror)

**2DGS is a representation experiment, not a mesh detour.** Oriented Gaussian disks with perspective-correct intersection and depth/normal regularization can discourage thick, floating surfaces. The original DTU comparison reports Chamfer distance **0.80 versus 1.96** for its 3DGS baseline, with **10.9 minutes** training and **34.52 versus 35.76 dB**. Geometry improves without universally improving image PSNR. These are offline results, not online convergence evidence. Ito would need a compatible rasterizer; flattening one covariance axis in its existing affine-projected 3DGS shader is not the complete method. Trial only after the bounded 3DGS baseline exposes a remaining geometry/stereo problem. [2DGS paper, Tables 2–3](https://arxiv.org/html/2403.17888v3)

### Licence findings affect the build choice

Checked the repositories/model cards on the review date. The paper's publication licence is **not** the software or weight licence. “Root licence” below does not relicense third-party code. In particular, several permissive-root Gaussian projects use an Inria research-only rasterizer. A product implementation should use a deliberately audited stack, such as Apache-2.0 gsplat, rather than assume all transitive parts are permissive.

| Component | Verified terms and consequence |
|---|---|
| MASt3R-SLAM / MASt3R | [SLAM licence](https://github.com/rmurai0610/MASt3R-SLAM/blob/main/LICENSE.md) and [MASt3R notices](https://github.com/naver/mast3r): **CC BY-NC-SA 4.0** restrictions matter for distribution/commercial use. Preserve current research results, but do not base the product's only path on presumed commercial permission. |
| MonoGS | [Custom Imperial licence](https://github.com/muskie82/MonoGS/blob/main/LICENSE.md), limited noncommercial/research rights; not MIT/BSD. Borrow published ideas or secure appropriate rights. |
| Photo-SLAM | Both [GPL-3.0 `LICENSE`](https://github.com/HuajianUP/Photo-SLAM/blob/main/LICENSE) and [Inria `LICENSE.md`](https://github.com/HuajianUP/Photo-SLAM/blob/main/LICENSE.md) exist. Requires resolving file/dependency coverage; describing the complete stack simply as GPL is insufficient. |
| SplaTAM / GS-ICP | Root [BSD-3-Clause](https://github.com/spla-tam/SplaTAM/blob/main/LICENSE) / [MIT](https://github.com/Lab-of-AI-and-Robotics/GS_ICP_SLAM/blob/main/LICENSE). Audit or replace inherited Gaussian rasterization components separately. |
| RTG / GlORIE | Root [GPL-3.0](https://github.com/MisEty/RTG-SLAM/blob/main/LICENSE) / [Apache-2.0](https://github.com/zhangganlin/GlORIE-SLAM/blob/main/LICENSE); dependent models/rasterizers remain separate. |
| pixelSplat / MVSplat | Root [MIT](https://github.com/dcharatan/pixelsplat/blob/main/LICENSE) / [MIT](https://github.com/donydchen/mvsplat/blob/main/LICENSE). Separate weight grants were not independently resolved here; verify the exact checkpoint and dependencies selected. |
| DepthSplat | [MIT code](https://github.com/cvg/depthsplat/blob/main/LICENSE) and [MIT model card](https://huggingface.co/haofeixu/depthsplat). Verify the selected pretrained backbone and rasterizer as well; a model-card label alone does not settle inherited terms. |
| AnySplat | [MIT code](https://github.com/InternRobotics/AnySplat/blob/main/LICENSE); the [official model card](https://huggingface.co/lhjiang/anysplat) explicitly labels code and models MIT. Do not infer its weight terms merely from VGGT's historical licence. |
| Long-LRM | [Apache-2.0 code](https://github.com/arthurhero/Long-LRM/blob/main/LICENSE); [README](https://github.com/arthurhero/Long-LRM) specifies **Adobe Research License** for weights. Not a permissive-weight default. |
| VGGT | [Custom licence](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt) updated July 2025 permits commercial use except military applications; only the new **VGGT-1B-Commercial** checkpoint has that permission, not the original. Access is gated. |
| π³ / π³X | [BSD-3-Clause code, CC BY-NC 4.0 weights](https://github.com/yyfz/Pi3#-license). A permissive source tree does not clear the pretrained model for a commercial Ito build. |
| DPVO / DROID | Root [MIT](https://github.com/princeton-vl/DPVO/blob/main/LICENSE) / [BSD-3-Clause](https://github.com/princeton-vl/DROID-SLAM/blob/main/LICENSE). Confirm bundled checkpoint notices and CUDA dependencies when packaging; no blanket transitive clearance is implied here. |
| OpenVINS / Basalt | [GPL-3.0](https://github.com/rpng/open_vins/blob/master/LICENSE) / [BSD-3-Clause](https://github.com/VladyslavUsenko/basalt-mirror/blob/master/LICENSE). Both can be evaluated; distribution strategy depends on selected code and linkage. |
| 2DGS / gsplat | Original [2DGS research-only Inria licence](https://github.com/hbb1/2d-gaussian-splatting/blob/main/LICENSE.md); [gsplat Apache-2.0](https://github.com/nerfstudio-project/gsplat/blob/main/LICENSE) provides a different implementation route. A renderer implementation licence and a paper's method are distinct. |

My shortlist is therefore **gsplat for optimization; independent robot-aided tracking, with a CPU VIO baseline and DPVO challenger; DepthSplat-small/MVSplat as optional initializers**. It is an architectural selection, conditional on the shared-load experiments below, not a promise that installing three libraries produces Ito.

## 4. Three serious iterations, ranked

The ranking favors the current robot, visible gains on one consumer GPU, and a small maintainable system. These are **investment choices with falsifiable first experiments**, not three backends to implement simultaneously. Effort estimates are my engineering estimates for one experienced engineer, including integration and offline end-to-end validation; physical deployment and headset acceptance can extend them.

### 1 — Independent state estimation, persistent optimized Gaussians, explicit display budget

**Build this first.** Keep Ito's driver/link/render boundaries, but separate the fast state estimate from appearance learning and preserve enough observations to revise geometry correctly.

```text
Robot: calibrated RGB + source times + motion prior [+ raw IMU / range / depth]
                                |
                  bounded timestamped observations
                       /                       \
              STATE ESTIMATOR              MAP WORKER
       propagate motion immediately      select useful keyframes
       visual updates at input rate      metric, consistent geometry seed
       delayed factors / bias / scale    optimize recent local Gaussians
             |             |             refine/freeze stable surfaces
             |        pose revisions ---------->|
             |             |                    |
             |       retained observations + anchored submaps
             |                                  |
             +-------- coherent pose/map revision ----+
                                                     |
Headset predicted pose --> DISPLAY, 90 Hz, last complete scene
                            shared GPU budget; no waits on mapping
```

**Estimator.** Publish propagation from robot motion independently of dense reconstruction. Track image features/patches between observations, with metric priors and timing uncertainty; run expensive dense depth only on useful keyframes. Use current MASt3R to establish a transition baseline, then compare CPU visual-inertial tracking and DPVO rather than immediately committing to a DROID rewrite. Preserve the distinction between a current propagated pose and a retrospectively refined keyframe pose. If raw inertial input is absent, use the available pose/velocity prior honestly; do not pretend a VIO mode exists.

**Map.** Fit genuine anisotropic Gaussians against multiple calibrated RGB observations: center, rotation, scale, opacity and color, with exposure handling and measured geometric constraints where available. Retain bounded source keyframes and support/uncertainty metadata. Existing static surfaces accumulate evidence; recent uncertain surfaces are provisional and get priority. Keyframe removal from the estimator must not erase useful room geometry: consolidate surfaces into persistent submap anchors before forgetting their source node. Marginalize/prune according to support, view contribution and resource cost, not simply “oldest keyframe dies.” Large optimizer deformations invalidate the affected region and trigger local refitting rather than publishing a sequence of visibly inconsistent partial revisions.

**Publication.** One scene revision identifies its coordinate frame, compatible anchor transforms, observation age and completion fence. The renderer reads the most recent complete revision; superseded results cannot overwrite a newer world. Start with bounded shared-memory deltas and measure their cost. Introduce CUDA/GL interop only if it materially helps and remains robust on Windows. Keep the record boundary narrow enough for SH degree and anchor transforms to evolve; avoid copying an entire room every update.

**Scheduling.** Render first, then admit estimator/mapping work according to measured remaining capacity. Use small local training batches, bounded queues and cancellation of obsolete requests. Measure longest GPU kernels and driver scheduling; CPU sleep-based duty control is inadequate. If dense inference cannot coexist with deadlines, reduce its frequency/size, replace it, or move the visual estimator to CPU. Do not silently lower display rate. Start experiments around 200–400k active Gaussians and adapt to measured screen coverage/overdraw; no fixed count is declared safe in advance. Train at camera-appropriate resolution and render at eye resolution. Persistent map quality can improve over seconds while current pose and recently observed hazards stay on the fast path.

**Why it can beat the wall.** It preserves multi-view evidence instead of repeatedly replacing coarse blobs; isolates pose publication from slow optimization; and makes resource contention that can cause hitches an explicit admission decision. It uses the architectural lesson of Photo-SLAM without requiring Ito to inherit that entire codebase. The head remains responsive even if no map update is ready.

**Risks/cost.** A bad initial depth field may still converge too slowly; a cheap tracker may fail on blur and featureless walls; Windows GPU scheduling may prevent the desired reservation with a particular model. Estimate **6–10 engineer-weeks** after the initial experiments, with the largest uncertainty in bounded optimization and correction publication. Reuse gsplat and an existing estimator; do not build a new generic SLAM framework.

**First prove/kill experiment, 2–4 working days:** replay the two trajectories with **true poses and true sim depth**, no MASt3R, a fixed memory budget, and the actual renderer running a recorded 90 Hz stereo head trajectory on the same GPU. Deliver causal snapshots after 100 ms, 500 ms, 1 s and 2 s of observation. Continue only if books/poster boundaries are visibly sharper than current main, full-frame quality improves by at least **3 dB** on the fixed informative views after at most **1 s**, no extra holes appear on the common observed region, and display misses remain below **0.1%** at 90 Hz with **<8 GB** total Ito GPU residency. Failure with oracle inputs falsifies this mapper/scheduler configuration before spending weeks on tracking. Success does not yet validate mono inputs; replace truth depth and truth poses separately next.

### 2 — A sensor-complete Microduck, using the same optimized map

**Highest confidence in a higher ceiling if physical changes are acceptable.** Treat good sensing as part of Ito's product requirements rather than demanding that a learned prior compensate indefinitely for missing information. First characterize the existing camera/ToF/IMU; then compare a synchronized global-shutter stereo or compact RGB-D head and a rigidly calibrated head IMU. A second camera without usable baseline, synchronization or texture is not automatically an improvement.

```text
Synchronized camera pair / RGB-D + rigid head IMU + joint/motion prior
                   |                    |
           metric depth/confidence    metric pose estimator
                   +---------+----------+
                             |
                 optimized Gaussian local mapper
                 stable surfaces + recent changes
                             |
              same revision contract and 90 Hz renderer
```

**Why it can beat the wall.** Measured metric geometry attacks the scale/shape inconsistency directly, reducing both optimizer work and pose ambiguity. Global shutter and exposure control attack the input corruption before inference. GS-ICP-style geometric tracking becomes a realistic option; CPU stereo/VIO or device depth can free GPU time. Wider observed coverage reduces the frequency with which the pilot looks into unobserved space. This route can improve sharpness, freshness and compute simultaneously rather than exchange one for another.

**Risks/cost.** Microduck's head mass, torque, power, minimum working distance, illumination, available interfaces and network bandwidth may rule out the best sensor on paper. Stereo fails on textureless/reflective regions; active depth has its own lighting/material failures. Preserve invalid masks and uncertainty. Proposed procurement cost is deliberately unspecified until those mechanical/electrical constraints are measured. Estimate **3–6 engineer-weeks of additional integration/calibration**, excluding hardware lead time and any mechanical redesign. Other robots benefit through optional sensor observations in the same contract; never make this sensor suite mandatory for all drivers.

**First prove/kill experiment, 1–3 working days:** use sim truth depth/poses to establish the ceiling, then progressively restrict depth to noisy delayed 8×8 angular zones, realistic stereo-like depth, and no depth while retaining the same images, tracker and mapper. At near-field stereo eye poses, require **≥50% lower p95 disparity error** and **≥3 dB better full-frame quality** than the mono baseline, with no render/freshness regression. Visually inspect pot/ball occlusion, floor seams and bookshelf edges. If dense accurate depth does not help, a sensor purchase is not the immediate fix. If it does, a recorded hardware bench experiment must then demonstrate the input quality and packaging constraints; the synthetic intervention alone does not justify a specific device.

This is ranked second because it changes the physical robot. If hardware flexibility is high, it may be the most effective first investment alongside proposal 1's oracle experiment.

### 3 — Feed-forward Gaussian initialization, followed by bounded optimization

**Research bet with a short, decisive trial.** Predict a useful local set of full Gaussian parameters from two to four overlapping, posed observations, rather than beginning every local region from isotropic point samples. Keep proposal 1's independent state estimate and publication policy. Optimize against observations after prediction; prevent uncertain predictions from becoming permanent geometry merely because they look plausible.

```text
Calibrated RGB window + robot-aided poses + optional metric constraints
                                  |
                   DepthSplat-small / MVSplat seed
                   confidence + local Gaussians
                                  |
                bounded multi-view Gaussian optimization
                merge supported surfaces into persistent map
                                  |
                  same local pose + 90 Hz stereo renderer
```

**Why it can beat the wall.** The failed online mapper spends its few available steps repairing a poor starting shape/appearance. A learned initializer can amortize that work across training scenes and start near a sharper solution. Joint multi-view inference can also be more consistent than independent single-frame depth. Recurrent systems such as Flash-Mono suggest a longer-term way to avoid rebuilding overlapping windows, but reproducibility and resource fit come first.

**Risks/cost.** Small portrait views close to the floor, nearly pure rotation, blur and exposure changes differ from many training datasets. A model can produce convincing wrong surfaces. Consecutive windows can change scale, geometry or opacity and create popping when merged. An inference pass that is fast in isolation may still block VR; duplicating a large depth model, estimator and optimizer can exceed 8 GB. Select one model and share/reuse its outputs; do not run MASt3R, VGGT, DPVO and DepthSplat concurrently by default. Estimate **2–4 engineer-weeks to evaluate/port a chosen small model**, then **6–10 additional weeks** for persistent-map integration if it passes; retraining or recurrent-model development is a separate, larger research project.

**First prove/kill experiment, 2–4 working days:** for 20 fixed windows spanning slow movement, turns and revisits, predict from **past/current views only** using true poses first, then own poses. Compare at held-out true stereo/head-offset poses after the same allowed GPU milliseconds of optimization. Require **≥2 dB full-frame improvement over the point-seeded optimizer**, ≥20% lower LPIPS on informative regions, no worse coverage/disparity, and at least a **2× reduction in time to the same visible sharpness**. Then run it beside the renderer; the whole path must pass the same <8 GB / 90 Hz limits. Kill this initializer if the gain vanishes under pure turns or own poses, requires future views, or depends on a large model that cannot meet the shared budget. This trial was **not run** in this review.

### The experience all three must deliver

“World rendered at 90 Hz” is necessary, but it is not a comfort certificate. There are separate clocks:

| Information | What should be live | What can lag |
|---|---|---|
| Pilot head/eyes and controls | Current predicted display pose each frame; controls keep their independent sender | Never wait for scene optimization |
| Robot anchor motion | Latest propagated, uncertainty-bounded motion; measure deliberate comfort filtering separately | Global loop corrections can be deferred/applied coherently |
| New obstacles / moving objects / newly revealed space | First trustworthy observation as soon as transport and processing allow; initial target **≤150 ms p95 at 30 FPS**, measured from exposure | Fine appearance can converge later; a seconds-old obstacle is not “live” |
| Previously observed static room | Stable geometry/color and smooth reprojection | Seconds of appearance refinement, and much longer persistence if not contradicted |

The 150 ms figure is a proposed engineering target, not a measured result or human-safety threshold. At 5 FPS, the sampling interval alone is 200 ms; report that limit honestly and adapt the allowed motion/experience. Do not promise the same freshness at every input rate.

Keep **one Gaussian representation with different evidence/lifetime policies**, rather than introducing a second mesh system. Newly observed geometry can be provisional while its parameters converge. Moving or contradictory regions must not be optimized into the permanent room: hold them in a short-lived local layer, expire them when unsupported, and restore the background only from valid observations. Distinguish physical opacity from confidence; fading an uncertain opaque obstacle into transparency can be misleading. A complete first implementation needs invalidation, age and failure behavior, not necessarily a learned deformable model for every object.

For unobserved directions, show a consistent, neutral missing-region treatment and an age/coverage indication; keep valid static content stable in both eyes. Do not stretch the last image around the pilot or invent detailed geometry to hide holes. If tracking/coverage is inadequate, Ito already has the right class of fallback: a clearly framed camera panel in a stable VR environment, with stop/recenter available. Its latency must remain visible. This is an explicit degraded state, not the normal immersive view. Reacquisition should introduce verified scene regions coherently without a world jump.

A stable local rendering frame should absorb coordinate-gauge changes applied equally to eye and map; actual corrected robot motion is a different event. Do not add the robot head rotation to the pilot's gaze twice. Preserve the owner's liked smooth waddling response as a tested preference; do not assume physically exact vertical acceleration is comfortable for everyone. Main's 2°/s and 5 cm/s correction limits cite redirected-walking research in a comment, but that does not establish imperceptibility for this seated, near-floor teleoperation setting. Headset checks must measure correction-induced world motion and comfort directly.

## 5. Critique of the coordinator plan and the evidence process

I reviewed `work/ARCHITECTURE-PLAN.md`, dated 12 October, SHA-256 `67e98984c1fb2e44d31db108d6eb3f8f3c2e2cbfdc9f06272aa33b1bd5f23712`. I have not modified it.

| Keep | Change before treating it as the implementation plan |
|---|---|
| Robot motion fused with visual factors; every frame can receive a pose | Separate the immediate propagated pose from the delayed visual correction and reconstructed-surface update. Specify publication times and uncertainty. The current graph window is 30 keyframes, not the older 12-keyframe description in README. |
| Optimized Gaussians, keyframe-local coordinates and asynchronous mapping | Make map lifetime independent of estimator window lifetime; retain observations, define consolidation, and publish coherent anchor/map revisions. “Background” alone does not specify a scheduling or consistency policy. |
| Photometric refinement as a possible constraint | Remove **MAP ALIGNMENT** as the mandatory route to accurate poses. Gate it on mature, independently validated static geometry, sufficient texture and bounded correction; compare to frame-to-frame tracking. The measured aligner regression is evidence against using it as the default now. |
| A budget sized for VR | Replace the isolated “~400k means GO at 90 Hz” conclusion with a shared-load frame-time experiment. The README's hardware default can be **1M** splats on ≥8 GB NVIDIA, which is also inconsistent with that asserted budget. Occupancy, allocated capacity and overdraw must all be recorded. |
| Per-camera calibration, optional metric sensing, exposure correction | Implement source timing and actual frame semantics early. Do not postpone lens/rolling-shutter metadata when evaluating a physical rolling-shutter camera; unknown values must be explicitly unknown. Replace first-camera/floor-specific assumptions with named-frame observations as real drivers need them. |
| Truth-backed replays and pilot feedback | Add matched stereo/head-offset views, temporal checks, deadline misses and event freshness. Keep the owner's feedback authoritative for the experience, but give them concrete visual evidence before asking them to spend a headset session discovering obvious blur or holes. |

The “what we know” section needs versioned claims. Its 1.9 m “unexplained tracking drift” is the odometry quantity identified in the later investigation; it should not remain evidence of tracker divergence. Its ~10 Hz claim predates the more expensive `fix/pixel-poses` runs. The older true-pose 21 dB result is confounded by depth scale and renderer differences. “Inputs are now the wall” is too singular: the online mapper has not yet shown sufficiently fast convergence even with true poses, and joint display scheduling has not been demonstrated.

The [PR #36 tracking study](https://github.com/arteeh/ito/blob/29a3adc/docs/research/tracking.md) correctly explains why replacing a jointly solved rotation after the fact can damage the rest of the estimate, and why a robot prior belongs inside fusion. Its categorical dismissal of classical VIO because the robot already knows rotation goes too far: raw inertial fusion also estimates bias, velocity, scale and timing. Its low DROID frontend memory figure is an estimate, not a reproduced budget. The [earlier representation study](https://github.com/arteeh/ito/blob/e0ae55e/docs/research/scene-representation.md) correctly distinguishes immediate observations from persistent memory. Its depth-mesh-first recommendation adds a second representation and still inherits timing/depth consistency problems; I would not reopen that rejected complexity before the optimized-Gaussian oracle experiment. Its “Photo-SLAM is GPL” shorthand also omits the separate Inria licence file.

These proposed pass criteria are engineering decision aids, **not a new approval or merge policy**. This review changes no product code and makes no merge decision.

### What current metrics miss

1. **The renderer being scored is often not the pilot renderer.** Main's [replay preview](https://github.com/arteeh/ito/blob/fff0439/e2e/slam_replay.py#L59) draws nearest-depth, square point footprints with radii clamped to three pixels. It does not reproduce Gaussian alpha compositing, covariance, SH, stereo sorting or temporal fades. The mapper branch's different renderer further complicates historical comparisons. Use one production rendering path for decisions about appearance.
2. **Coverage-conditioned scores reward hiding mistakes.** A method can delete difficult pixels and improve covered PSNR. Report full-frame PSNR/SSIM/LPIPS, an externally defined common visibility mask, and coverage separately. Treat newly exposed unknown space separately from holes in previously observed space; a truthful unknown region should not be scored as though hallucinating it were the only acceptable answer.
3. **“Causal” must include all upstream evidence and runtime.** The earlier calibrated gsplat initialization uses triangulation pairs up to 60 frames ahead and interpolated scale. Its apparent streaming experiment also does not charge MASt3R/pose computation or competing display work. Every-eighth-frame held-out evaluation is useful for fit, but heavily correlated neighbors from the same static room make it an optimistic deployment test. Retain those results as upper bounds.
4. **Pose error at capture is not pose available at display.** The current replay fills intermediate-frame estimates after processing a later frame ([replay:392](https://github.com/arteeh/ito/blob/fff0439/e2e/slam_replay.py#L392)). Log the estimate that existed at the original deadline as well as the retrospectively optimized one. A per-image photometric pose fit against a held-out image is an oracle analysis, not an online tracking result.
5. **One freshness number conflates three things.** Record exposure-to-first-visible-observation, exposure-to-stable-sharp-surface, and pilot-motion-to-display separately. Scene delta timestamps do not necessarily date every visible surface: an old surface can be reoptimized and published now. Store last observed/validated age separately from optimization time.
6. **Average FPS misses the uncomfortable frames.** Report display frame-time p50/p95/p99/p99.9, missed-deadline fraction, longest run of misses, upload stalls, and correction-induced screen motion under simultaneous tracking/training. Report visual factors/s and fresh geometry updates/s separately from propagated poses/s. Add stereo disparity and foreground-edge stability; PSNR alone does not see binocular disagreement.
7. **The environment is too forgiving.** These recordings cannot validate real exposure, noise, sensor readout, moving objects or lens calibration. Use a recorded physical Microduck sequence and a public sensor dataset for transfer, plus controlled synthetic corruptions. Never quietly fit scale or calibration on evaluation targets and then label the result deployable.

### The standard eye-check protocol

For each candidate, save the same six reference moments used here plus one difficult moment selected **before** seeing that candidate's scores. Include turns, revisits, a near foreground object and a newly revealed edge. Render truth-left/candidate-right with the same calibrated camera; then do left-eye and right-eye offsets at actual headset IPD, and a small ±3 cm head translation. Save whole frames at 1:1 scale before adding crops. Both the camera-path and novel-eye tests matter: reconstructing the captured image does not prove correct stereo geometry.

Save three short, synchronized clips: **frozen map with moving head**, **fixed head with map updates**, then **both together**. The first isolates renderer/sort/aliasing artifacts, the second exposes popping and world corrections, and the third is the actual experience. Add a moving foreground object crossing a known wall and a brief sensor dropout. Display frame IDs, source observation ages and actual display times in a diagnostic margin, not over the comparison region. Include speed-accurate playback and slow motion; stills cannot establish smoothness.

Inspect blind A/B clips before opening scores. Record a short statement about book labels, floor grain, single versus doubled edges, holes, stereo disagreement and map transitions. Then examine metrics to explain disagreements. Use three repeated runs where scheduling or optimization is nondeterministic, identical input/frame availability schedules, and per-sequence/per-event results; do not average a failed tour into a successful sweep. The owner need only inspect the decisive sheets/clips, not every log. Headset acceptance remains necessary after offline checks, but no live drive was used for this review.

For “plain improvements,” require a clear visual gain and no material regression in the comfort, freshness, coverage and resource measures. Define tolerances before a comparison and estimate run-to-run noise; a hundredth of a dB is not a product gain. A weighted aggregate score must not buy better texture by hiding holes or more missed display frames.

## 6. Ordered next steps and decision criteria

These are proposed gates, not claims that the current system passes. Use the same frozen reference recordings and actual GL renderer throughout. Every GPU experiment holds `ceres.lock`, stays below 8 GB, and ends within 20 minutes; split experiments into independent short jobs. Continue offline until headset inspection is deliberately scheduled by the owner.

| Order | Concrete work | Visual pass criterion | Numeric pass criterion / decision |
|---|---|---|---|
| **1. Establish trustworthy measurement** | Add source frame identity and exposure/arrival/publication/display timestamps to the offline recording path; preserve camera model, exposure/readout status and coordinate revisions. Make identical causal frame schedules and true-pose GL evaluation reusable. | Static truth matches decoded input in geometry; matching thin edges do not jump because of timestamp selection. Both eye images have correct projection and orientation. | New sim recordings have **100% exact frame-ID matches**, no future observations in a prefix snapshot, and calibration reprojection residual **≤0.5 px median, ≤1 px p95** on the known static scene. Keep codec/render differences as a separately measured floor. If physical exposure time cannot be obtained, explicitly quantify delay/jitter instead of calling it exact. |
| **2. Prove the shared compute envelope** | Run actual 90 Hz stereo rendering under controlled tracking and training load; sweep occupied count, allocated capacity and viewpoint. Trace long kernels, uploads and publication. | Frozen-map head sweeps remain smooth and stable; no shimmer bursts or black frames when training starts. | **<0.1% missed 11.11 ms deadlines**, no consecutive misses in the short reference runs, **<8 GB** aggregate Ito GPU residency. Record p99/p99.9 and longest stall. Select the cap/model schedule from this trace, not isolated FPS. A production headset/compositor check must later confirm it. |
| **3. Isolate the mapping ceiling** | Under that budget, run the 2×2 intervention: true/estimated pose × true/estimated depth, with identical observations and prefix availability. Optimize real Gaussians and retain the input images. | Oracle inputs produce sharper book labels/poster edges, without a missing floor or moving walls. Stereo foreground silhouettes agree. | With true inputs, target **≥3 dB full-frame gain** over main on fixed informative views within **1 s**; common-observed-region holes do not increase by >**0.5 percentage point**. Meet step 2 simultaneously. Use each oracle replacement's marginal gain to choose whether depth, pose or optimization is next. |
| **4. Fix the estimator's actual error budget** | Calibrate timing/extrinsics; compare current fusion, a sensor-aided feature/VIO baseline where raw IMU exists, and DPVO as a bounded challenger. Start without map alignment. Measure estimates as available at the time. | Static shelf/door edges stop sliding or doubling during turns; no apparent scale breathing when the floor exits view. | On both sequences, **halve causal p95 rotation error** from the frozen baseline, no increase in p95 translation error, and no added pose-publication delay. For sharp-map admission, require supported cross-view reprojection **≤2 px p95** or explicitly exclude uncertain measurements while retaining common-region coverage. Validate on an independent motion sequence; if timing uncertainty dominates, fix the source rather than retune visual weights. |
| **5. Make optimization persistent and causal** | Combine the winning estimator with bounded local training, retained observations, submap consolidation, per-frame exposure and coherent publication. Add a revisit and intentional coordinate correction. | The first bookshelf remains sharp after its estimator keyframe retires; returning to it does not double it. No correction makes a stationary pilot's world jump. | Full-frame quality **≥2 dB above main on both complete recordings**, LPIPS improves on predefined textured regions, common-region coverage stays within **0.5 point**, and p95 first-visible freshness does not regress by >**10 ms**. Pass step 2 and keep steady memory bounded over a **10-minute offline loop**. Check that all visible changes carry the correct observation age. |
| **6. Spend on the demonstrated bottleneck** | If initialization dominates, run proposal 3's 20-window feed-forward trial. If measured metric geometry dominates, run proposal 2's sensor degradation intervention and then plan a hardware recording. Do not begin both full integrations. | Improvement persists at near-eye offsets, on turns, and away from the floor; no plausible but wrong furniture or surfaces appear. | Apply the proposal-specific ≥2 dB/2× convergence or ≥50% disparity-error reduction gate, plus step 2. If neither passes, keep the measured baseline and reject the dependency/hardware change; do not ship a prettier offline-only result. |
| **7. Establish live-world behavior** | Offline replay: a moving object, exposure change, blur/readout perturbations, 0.5 s input loss, no-floor interval, and reconnection/relocalization. Use a recorded physical Microduck sequence before release. | Moving subjects do not leave permanent ghosts; unknown regions are recognizable; loss/recovery keeps a stable fallback and returns without a world jump. | At **30 FPS**, aim for first reliable event depiction **≤150 ms p95** and no unsupported moving trace older than **0.5 s** after a visible removal. At lower rates, report the sampling-limited age explicitly. No tracking/training fault blocks controls or breaches step 2. Physical timing and scale must be validated, not borrowed from sim. |
| **8. Pilot acceptance** | Present blinded candidate/main clips first; then, with the owner's authorization for that session, run the headset comparison including the liked smooth waddle. | Owner prefers sharpness and continuity without added discomfort; labels/edges remain readable during actual piloting and both eyes agree. | Meet the display/freshness gates under the real runtime; log identical routes and settings. Record subjective comfort separately from frame metrics. A visually worse or less comfortable result fails even if its mean PSNR improves. |

**The immediate action is steps 1–3, not another wholesale tracker replacement.** They answer the missing question: with information and scheduling controlled, can an online optimized Gaussian map become sharp quickly enough for this pilot view? The results then determine whether Ito should spend its next month on estimation, initialization, sensor quality or the optimizer itself.

## 7. Scope and reproducibility of this review

Completed: code/history/plan audit; primary-source research and licence checks; four short current-main offline SLAM replays; six matched-camera visual sheets rendered through Ito GL; inspection of all six sheets and the historical online mapper example; extraction of historical mapper metrics with source hashes. The committed scripts are standalone offline diagnostic drivers, not unit tests or new product paths.

Not established here: physical Microduck calibration/readout/exposure quality; shared-GPU 90 Hz headset performance; temporal popping measurements; an online optimized map trained with current `fff0439` poses; feed-forward initializer performance on these recordings; commercial clearance of every transitive dependency. These are identified experiments, not results silently assumed by the recommendations.

To reproduce the capture procedure in this machine's existing Windows SLAM environment, from `work/bigpic` in WSL (models/environment are read-only):

```sh
export PYTHONPATH='C:\Users\me\Projects\ito\work\bigpic'
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
export CUPY_CACHE_DIR='C:\Users\me\Projects\ito\work\bigpic-out\cupy-cache'
export WSLENV=PYTHONPATH:PYTHONUTF8:PYTHONDONTWRITEBYTECODE:CUPY_CACHE_DIR
flock /mnt/c/Users/me/Projects/ito/ceres.lock \
  /mnt/c/Users/me/Projects/ito/venv-slam/Scripts/python.exe \
  e2e/big_picture_capture.py \
  --models 'C:\Users\me\Projects\ito\work\pp\models' --at 6,15,23 \
  C:/Users/me/Projects/ito/work/rf-run/out/sweep180/rec \
  --out C:/Users/me/Projects/ito/work/bigpic-out/sweep180-L1 \
  --pose L1 --truth C:/Users/me/Projects/ito/work/rf-run/out/sweep180/truth.jsonl \
  --limit 24
```

Repeat for L4; for tour use `--at 12,50,76 --limit 78` and the tour recording/truth/output paths. The snapshot files record which frame actually crossed each target. The pinned baseline was `fff0439`; timing-dependent frame selection need not reproduce byte-identical snapshots.

Then render the six sheets with the isolated image environment (NumPy, Pillow, MuJoCo 3.3.7, ModernGL, PyOpenGL, plyfile, SciPy and pydantic), using the unchanged recorded gsplat checkpoints:

```sh
PYTHONDONTWRITEBYTECODE=1 flock /mnt/c/Users/me/Projects/ito/ceres.lock \
  ../bigpic-out/venv/bin/python e2e/big_picture_images.py \
  --work /mnt/c/Users/me/Projects/ito/work --out docs/big-picture/img
```

The manifest records the saved room/checkpoint/snapshot hashes and software renderer. JPEG sheets are each below 1 MB; uncompressed pairs remain in `work/bigpic-out/rendered`. No shared recordings, mapper worktrees, model environments, simulator installations or running owner processes were modified.
