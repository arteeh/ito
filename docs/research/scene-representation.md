# Scene representation and depth source for Ito

Research report, October 2026. Question from the owner: "Heard of Depth Anything? What if instead
of splats/points we went for generating a 3D scene of meshes/textures based on image+depth data?"

Written without a GPU and without the owner's PC: everything below comes from reading Ito's code,
papers, repositories and licence files, plus one small CPU-only numerical check (section 1.3).
Speeds marked *est.* are my estimates for an RTX 4070 Ti, not measurements. Section 6 lists what I
could not verify.

## TL;DR

1. **The representation is not why the scene looks bad today.** Doubled walls come from poses
   (MASt3R tracking loss, map restarts, heading corrections), and a TSDF mesh fed the same poses
   smears the two walls into one thick blurry wall instead of doubling them. Blur comes from how
   Ito turns pointmaps into splats: fixed 3 px isotropic discs from a 512 px wide image, opacity
   0.85, colour from one pixel, never optimised (`ito/reconstruction/slam.py`,
   `rgbd.py:integrate_points`). Neither is fixed by switching to meshes.
2. **Yes to a mesh, but only for the newest frame.** The best way to show the newest camera view at
   once, sharp and stable, is to draw that frame as a textured depth mesh: a grid displaced by
   its depth map, textured with the full resolution camera image, torn at depth edges. This is
   what headset passthrough does. Behind it, a persistent memory of the room fills the holes the
   pilot sees when their head moves away from the camera.
3. **No to a mesh for the memory.** For the persistent room, Gaussians optimised against the
   camera images (gsplat, Apache 2.0) beat a fused TSDF mesh on sharpness at the offsets a pilot
   reaches (up to 0.5 m), reuse the renderer Ito already has, and need no texture atlas. A TSDF
   mesh is the fallback if optimised Gaussians shimmer in the headset.
4. **Depth source:** Video Depth Anything Small (Apache 2.0, temporally consistent) for the live
   layer, put in metres by the Microduck's 8x8 ToF and the robot's own pose. Depth Anything 3
   Small/Base and MapAnything (Apache variant) are the commercial-safe candidates for multi-view
   depth and pose when MASt3R is replaced. Prompt Depth Anything was trained on 192x256 LiDAR
   prompts; an 8x8 ToF is far outside that, so use the ToF as a scale anchor, not as a prompt.
5. **Licences matter now.** MASt3R and MASt3R-SLAM are CC BY-NC-SA 4.0; Ito's Windows release
   already ships them. Several "Large" checkpoints (DA V2 B/L, VDA B/L, DA3 L/G, UniDepth, Pi3,
   CUT3R, StreamVGGT, original VGGT) are non-commercial. MonoGS, SuGaR, Gaussian Frosting, 2DGS
   and Inria's 3DGS code are non-commercial research licences. Photo-SLAM is GPL-3.0.

Ranked shortlist (section 4): **(1) live depth-mesh layer over the existing splat memory**,
prototype first; **(2) optimised Gaussian memory with gsplat**, seeded from depth and poses;
**(3) TSDF textured mesh memory**, only if (2) fails on comfort.

## 0. What Ito does today, read from the code

- `ito/reconstruction/slam.py`: MASt3R-SLAM tracker with a 4-keyframe local graph, robot pose
  prior (IMU heading plus head kinematics), heading steering, recall of sector keyframes, restarts
  by cause. Each tracked frame's canonical pointmap (`X_canon`, at MASt3R's 512 px wide crop) is
  placed in Ito's world and handed to `integrate_points` with a footprint of 3 px (more under a
  small budget). No confidence cut.
- `ito/reconstruction/rgbd.py`: power-of-two voxel keys, one splat per occupied cell, isotropic
  scale `0.65 * cell`, opacity 0.85, SH degree 0 from the pixel colour, identity rotation.
  `carve` retires splats in front of or on the newly seen surface. Old maps' splats fade.
- `ito/reconstruction/ring.py`: incremental packets of 4x vec4 records (position+opacity,
  scale+deadline, quaternion, SH DC) into shared memory; latest wins, never blocks.
- `ito/render/renderer.py` + shaders: OpenGL 4.3, GPU key generation and bitonic sort (once per
  stereo frame from the mid eye), instanced EWA quads, SH up to degree 3, fade by deadline. It
  already renders arbitrary anisotropic, rotated, SH-coloured 3DGS (it loads standard 3DGS PLY),
  so an optimised scene needs no renderer change, only a wider record in the ring.
- `ito/render/video.py`: the flat feed on a head-locked panel, shown when SLAM has lost tracking
  for 2 s (`ito/app/pilot.py: TRACKING_LOST`).
- The XR swapchain framebuffers have colour only (`ito/xr/session.py:48`): no depth attachment.
- Microduck camera: IMX219, 62 degree horizontal FoV, 360p/720p/1080p at 30 fps
  (`drivers/microduck/lens.py`); the sim streams 360p by default (`sim.py`), and the sim runs
  robotd with an absent ToF socket (`sim.py:230`).

Consequences: what the pilot sees in the camera's own direction is a 3 px disc cloud resampled at
512 px, not the camera image; the newest frame only appears after MASt3R tracks it (or never, when
it is skipped); and every pose error is written into persistent geometry.

## 1. Depth and geometry sources

### 1.1 Comparison

Speeds: the 4070 Ti is roughly an RTX 3090 for ViT inference with half its memory bandwidth, and
it shares 12 GB with Ito's renderer, the decoder and the tracker. *est.* figures assume FP16 at
the model's native input (about 518 px on the long side; the 640x360 camera frame is resized, so
1080p input costs no more unless tiled).

| Source | Gives | Metric | Temporal / multi-view | Speed on 4070 Ti | Weights licence | For Ito |
|-|-|-|-|-|-|-|
| MASt3R (today, in MASt3R-SLAM) | two-view pointmaps + matches, Sim3 poses | no (own units) | two-view, local graph | paper: ~15 FPS on RTX 4090; Ito's replays show skips and restarts | CC BY-NC-SA 4.0 (code and checkpoint, training data NC) | research baseline only |
| Depth Anything V2 Small / metric indoor Small | per-image relative disparity; metric variant trained on Hypersim | metric variant only | none: flickers frame to frame | *est.* 5 to 10 ms | Small Apache 2.0; B/L CC BY-NC 4.0; metric checkpoints' licence not stated | fallback |
| **Video Depth Anything Small** (+ metric Small) | temporally consistent depth for video; streaming mode | metric variant | temporal; streaming mode costs accuracy (ScanNet d1 0.926 to 0.836) | A100: 7.5 ms/frame FP16 at 518; *est.* 15 to 25 ms | Small Apache 2.0; B/L CC BY-NC 4.0; metric licence not stated | **live layer depth** |
| **Depth Anything 3** Small/Base, Metric-Large, Mono-Large | any-view depth + ray/pose; metric model gives metres via focal | Metric-Large yes; others need scale | multi-view; DA3-Streaming (sliding window, <12 GB) | *est.* Small/Base 20 to 60 ms per 4 to 8 view window | Small, Base, Metric-Large, Mono-Large Apache 2.0; Large, Giant, Nested CC BY-NC 4.0 | **commercial-safe multi-view depth + pose** |
| Prompt Depth Anything | metric depth from image + low-res depth prompt | yes | none | *est.* Small ~10 ms | repo Apache 2.0; per-checkpoint licence not stated on the README | 8x8 ToF is out of distribution (trained with 192x256 LiDAR); later, fine-tune Small on simulated 8x8 prompts |
| MoGe-2 (S/B/L, with normals) | metric point map, normals, FoV from one image | yes | none | MoGe ViT-L ~60 ms on A100/3090 (README); S/B *est.* 10 to 30 ms | code MIT; weights licence not stated in README, could not reach Hugging Face | strong single-image metric alternative, check licence |
| UniDepth V2 | metric depth + camera | yes | none | ViT-L heavy | CC BY-NC 4.0 | out |
| Metric3D v2 | metric depth + normals | yes | none | ViT-L/g heavy | BSD 2-Clause repo | possible, superseded by MoGe-2 / DA3 Metric |
| VGGT | multi-view poses, depth, points, tracks | no | multi-view, feed-forward | 1B params; heavy on a shared 12 GB card | original NC; VGGT-1B-Commercial: commercial use except military, gated form | too heavy |
| Pi3 / Pi3X | multi-view, reference-free poses + points | Pi3X approximate | multi-view | ~1B class | code BSD-3, weights CC BY-NC 4.0 | out |
| **MapAnything** | multi-view metric reconstruction; *accepts* intrinsics, poses, depth as inputs | yes | multi-view | no text figures; claims best speed/memory | `map-anything-apache` Apache 2.0; default CC BY-NC 4.0 | **best fit for robot-pose-prior tracking (#33)** |
| CUT3R, StreamVGGT | streaming recurrent reconstruction | no | streaming | | CC BY-NC-SA 4.0 | out |

### 1.2 Opinions

- **Per-frame depth for what the pilot sees now:** Video Depth Anything Small. It is the only
  Apache-licensed model built for temporal consistency, and flicker in a depth mesh is shimmer in
  the headset. Its streaming mode is weaker than offline; that is acceptable because the live
  layer is replaced every frame and the ToF and the memory's own rendered depth anchor it.
- **Metric scale:** do not trust any monocular metric model alone on a 25 cm tall robot looking at
  furniture from below; their training data are human eye height. Fit scale and shift per frame
  to the 8x8 ToF (1.3), and where the ToF has no return, to the memory's rendered depth.
- **Multi-view depth and pose (when MASt3R goes):** MapAnything (Apache weights) is the most
  interesting because it takes Ito's existing priors (calibrated intrinsics, IMU+FK camera
  rotation, odometry) as *inputs*, which is exactly the help MASt3R gets only by post-hoc steering
  in `slam.py`. DA3 Small/Base is the second option. Both need measurement on the 4070 Ti.
- **Prompt Depth Anything with the ToF:** the prompt branch was trained on ARKit LiDAR at 192x256,
  which is about 770 times more depth samples than 8x8. Out of the box it is out of distribution.
  Because the MuJoCo sim renders perfect depth, simulated 8x8 prompts are free; fine-tuning
  PromptDA Small on them is a credible later step, not a first one.

### 1.3 CPU check: can the 8x8 ToF put relative depth in metres?

VL53L5CX: 8x8 zones over 45x45 degrees, about 4 m range, 15 Hz at 8x8 (ST datasheet). The IMX219
sees 62 x 37.3 degrees at 16:9, so each zone is about 5.6 degrees, 52 px wide at 640 px, and the
ToF covers the central 73% of the image width and all of its height.

I simulated a 4 x 3 x 2.5 m room with a cupboard, camera 25 cm above the floor, 640x360, and a
fake depth network that returns `a * disparity * (1 + 8% smooth warp) + b + noise` with random
`a, b`. A robust least-squares fit of `a, b` to the 64 zone medians (3% ToF range noise) recovers
metric depth with AbsRel 3.0% mean, 4.0% worst over 40 random views; the residual is the warp the
fit cannot model. Conclusion: 8x8 is enough for scale and shift every frame, not for structure.
Not verified: the real sensor's multipath on glossy floors, sunlight, the ToF-to-camera extrinsic
on the Microduck head, and real network errors, which are not a smooth warp.

## 2. Representations

Columns: quality at novel views up to 0.5 m off the camera path; VR comfort; incremental update
cost; 90 Hz stereo render cost; how fast the newest frame shows; motion blur and plain walls;
memory; code maturity and licence.

| Option | 0.5 m novel view | Comfort | Update cost | 90 Hz stereo | Newest frame | Blur / plain walls | Memory | Maturity, licence |
|-|-|-|-|-|-|-|-|-|
| **A. Live depth mesh of newest frame** (grid displaced by depth, camera image as texture) | exact at 0, good to ~0.2 m, disocclusion holes beyond (filled by memory) | stable if depth is temporally consistent; cross-fade frames | one depth inference + one texture upload | trivial (~60k triangles) | **one frame after decode + depth** | blur in, blur out; plain walls fine (depth net handles them, no matching) | one frame | own code, ~150 lines GLSL/Python |
| B. Today: fixed isotropic discs | blurry everywhere | grainy, popping on refresh | cheap | ok (sorted, 1M) | after tracking | MASt3R loses plain walls | 64 B/splat | Ito code |
| **C. Optimised 3DGS memory** (gsplat, depth-initialised, photometric + depth loss on keyframe window) | best of the options when poses are good | risk: view-dependent shimmer and floaters off the path; mitigate with depth loss, SH degree 0 to 1, opacity reset | 30 to 200 ms per keyframe for 50 to 100 iterations at half res *est.* | Ito's renderer, unchanged draw | after optimisation (so pair with A) | blur baked in unless frames are rejected; depth prior keeps plain walls flat | 64 to 240 B/splat | gsplat Apache 2.0, mature; Inria 3DGS code non-commercial |
| C2. 2DGS surfels | similar to C, better geometry, slightly softer | flatter, less shimmer | similar | needs ray-splat intersection shader, or approximate as flat 3DGS | as C | as C | as C | gsplat implements 2DGS (Apache); original 2DGS repo Inria licence |
| **D. TSDF textured mesh memory** (nvblox, Open3D, or own CuPy TSDF; colour per vertex or atlas) | crisp silhouettes, blurry texture with vertex colour (2 cm voxel is ~5 px at 2 m); atlas fixes texture but seams | most stable: no sorting, no view dependence | cheap integration (GPU TSDF is ms per frame); marching cubes per changed block | trivial | after integration | pose drift smears; depth net handles plain walls | voxels + mesh | nvblox Apache 2.0 but Linux/Jetson focused; Open3D MIT, CUDA builds Linux only as far as I know; online atlas texturing is research code |
| E. Mesh-anchored Gaussians (SuGaR, Gaussian Frosting, MeshSplats) | high | good | minutes to hours, offline | moderate | no | n/a | high | SuGaR and Frosting under Inria non-commercial licence; offline only |
| F. Feed-forward Gaussians (AnySplat, DepthSplat, DA3 GS head, Flash-Mono) | good near inputs, degrades off-path | per-window regeneration pops | 0.1 to 1 s per window on large GPUs | Ito's renderer | after window | depends on backbone | per window | AnySplat/DepthSplat code MIT but weights tied to VGGT / DA V2 L (NC); DA3 GS head only in NC Giant model; Flash-Mono (ICLR 2026) has a project page and no code |
| G. Online GS-SLAM (MonoGS, Photo-SLAM, Splat-SLAM, HI-SLAM2) | good on benchmarks | as C | system replaces Ito's tracker | own renderers | after mapping | MonoGS mono and Splat-SLAM run well below camera rate; Photo-SLAM's ORB tracking fails on plain walls and motion blur | high | MonoGS Imperial non-commercial; Photo-SLAM GPL-3.0; Splat-SLAM Apache 2.0 (on DROID-SLAM) |

Notes behind the ranking:

- **The newest view.** Only A shows the camera image itself, at full resolution, one depth
  inference after decode, independent of whether SLAM tracked the frame. Everything else shows a
  reconstruction of it, later. That makes A the cheapest large comfort win available: it replaces
  both "blurry discs" in the forward direction and the 2 s fall-back to the flat panel.
- **Offsets up to 0.5 m.** A depth mesh is exact at the camera pose and degrades with offset:
  around depth edges it opens holes about `offset * (1/z_front - 1/z_back) * f` pixels wide. With
  the pilot 0.3 m off the camera, a chair 1 m in front of a wall 3 m away opens a ~0.2 rad gap.
  Those holes must show the memory, which is why A is a layer over C (or B today), not a
  replacement.
- **Meshes versus Gaussians for the memory.** A TSDF averages colours and depths in voxels: at
  1 to 2 cm voxels and 3.75 mm pixels at 2 m, vertex colour is 3 to 5 times blurrier than the
  camera. A keyframe texture atlas fixes that, but online atlases (view selection, seam blending,
  re-texturing when keyframes change) are a lot of code with no maintained Python library on
  Windows. Optimised Gaussians reach camera sharpness with gsplat and a short loop, and Ito
  already renders them. Meshes win on comfort only if Gaussians shimmer; measure that (section 4).
- **Ghosts.** All options except A write pose errors into persistent geometry. A cannot ghost:
  it is replaced each frame. That is a second reason to do A first: it removes ghosts from the
  part of the view the pilot looks at most, while tracking (issue #33) is fixed separately.

## 3. Fit to Ito: what each option changes

| Option | `ito/reconstruction` | `ito/render` | Reused |
|-|-|-|-|
| A. Live depth mesh | new `depth.py`: depth network in the existing worker process, ToF/prior scale fit, publishes the newest `(depth at 1/4 res, camera pose, capture time)` in a one-slot latest-wins shared buffer beside `output_camera` | new `live.py`: grid VAO, vertex shader samples a depth texture, fragment samples the camera texture (the same decoded frame `VideoPanel` uploads), discards triangles across depth edges, feathers frame borders; add depth attachments to XR swapchain framebuffers and the desktop target; draw live mesh first writing depth, then splats with depth test on and depth write off | renderer, sort, ring, anchor glide, frame/pose timing, the decoded RGB |
| C. Optimised Gaussians | new mapper: keyframe window, gsplat rasterisation and Adam step in torch on CUDA; densify/prune; publish changed Gaussians with full scale, rotation, opacity, SH | none for degree 0; ring record grows past 4 vec4 for SH degree 1+ (`ring.py` and `update.glsl` assume stride 4) | renderer as is, PLY loader for offline results, budget/fade/eviction policy |
| C2. 2DGS | as C with gsplat's 2DGS mode | either render as flat 3DGS (set third scale tiny: no change) or add a ray-splat intersection shader | as C |
| D. TSDF mesh | new TSDF integrator (CuPy or torch voxel hashing; nvblox is not a Windows option) + marching cubes per block + colour | new mesh renderer with incremental vertex buffers | carve logic becomes TSDF free-space; little else |
| F. Feed-forward GS | new model runtime per window | none | renderer |
| G. GS-SLAM system | replaces the whole worker, loses the robot-prior handling in `slam.py` | none or own renderer | little |

The narrow boundary AGENTS.md asks for already exists: reconstruction publishes Gaussians, the
renderer draws them. A adds one more narrow output (newest depth + pose) and one more draw call.

## 4. Recommendation

### Ranked shortlist

1. **Live depth-mesh layer over the existing splat memory.** Biggest comfort gain per line of
   code; shows the newest view in one depth inference; immune to ghosts; works when SLAM is lost
   (the robot's own pose places it), which removes most fall-backs to the flat panel. Depth from
   Video Depth Anything Small, scaled by ToF and the memory's rendered depth. Prototype first.
2. **Optimised Gaussian memory (gsplat).** Replace fixed discs with depth-initialised Gaussians
   refined on a keyframe window. Do it offline from recordings first (the agreed plan's step 2,
   with poses from truth, then from SLAM), then online at keyframe rate.
3. **TSDF textured mesh memory.** Only if (2) fails the shimmer criterion in the headset, or if
   tracking stays too noisy for photometric optimisation to converge.

Not recommended now: SuGaR/Frosting (offline, non-commercial), feed-forward Gaussians (weights
tied to non-commercial backbones, heavy, pop per window; revisit when Flash-Mono or DA3 ship
Apache code and weights), adopting a GS-SLAM system wholesale (loses Ito's robot-prior tracking;
licences).

### Prototype plan for (1) on the RTX 4070 Ti

Order: oracle first, so representation and depth are judged separately from tracking, using the
truth benchmark's oracle ladder (issue #32).

1. **Rung 0, oracle depth and pose.** MuJoCo Microduck room, 720p camera. Live layer from truth
   depth and truth camera pose, over today's splat memory (also from truth). This bounds what the
   layer can do.
2. **Rung 1, estimated depth, true pose.** Video Depth Anything Small (streaming) and, as a
   comparison, Depth Anything V2 Small metric indoor and MoGe-2 ViT-S, all scaled per frame by a
   simulated 8x8 ToF (rendered from MuJoCo depth with the VL53L5CX zone layout, 3% noise, 4 m
   range) and the memory's depth.
3. **Rung 2, estimated depth and pose.** Same, with today's MASt3R-SLAM poses and the robot prior
   when tracking is lost.
4. **Headset run.** SteamVR/Virtual Desktop on the owner's PC, 5 minutes of the scripted look
   around plus free piloting.

Changes: `ito/reconstruction/depth.py`, `ito/render/live.py`, depth attachments in
`ito/xr/session.py` and the desktop target, depth test in `GaussianRenderer.draw`, a simulated
ToF in the Microduck sim driver, and the benchmark's new metrics below in `e2e/`.

### Pass criteria (against the truth benchmark)

Views: *current* = the newest camera pose; *held-out* = truth renders at poses of frames withheld
from the reconstruction; *offset* = the current pose moved 0.1, 0.25 and 0.5 m sideways/up and
turned 15 degrees. Baseline = today's splats on the same recording.

| Metric | Pass at rung 0 (oracle) | Pass at rung 1 (est. depth) | Pass at rung 2 (est. pose) |
|-|-|-|-|
| PSNR / SSIM / LPIPS, current view | >= 30 dB / >= 0.90 / <= 0.10 | >= 26 dB / >= 0.85 / <= 0.15 | >= 26 dB / >= 0.85 / <= 0.15 (the layer ignores pose error at offset 0) |
| PSNR / LPIPS, 0.25 m offset | baseline + 4 dB / baseline - 0.10 | baseline + 3 dB / baseline - 0.08 | baseline + 2 dB / no worse LPIPS |
| PSNR / LPIPS, 0.5 m offset | no worse than baseline | no worse than baseline | no worse than baseline |
| Coverage (pixels drawn), 0.5 m offset | >= 95% | >= 95% | >= 90% |
| Ghost fraction (splats > 5 cm off any surface) | unchanged (the layer adds none) | unchanged | unchanged |
| Live surface error, p90, 0.3 to 3 m | < 1 cm | < 5% of depth | < 5% of depth |
| Static flicker: robot and pilot still, mean per-pixel change between display frames | < 1 grey level | < 2 grey levels | < 2 grey levels |
| Latency, frame decoded to drawn in the live layer, p95 | <= 1 display frame | <= 35 ms | <= 35 ms |
| Render, 90 Hz stereo at the headset's render target, 1M splats + live layer, depth net and SLAM running | GPU frame time <= 8 ms, zero dropped frames in 5 min | same | same |
| VRAM, whole app + worker | <= 10 GB of 12 | same | same |

The absolute PSNR thresholds are my starting guesses; recalibrate once the benchmark reports the
baseline. The relative ones (beat baseline at 0 and 0.25 m, never worse at 0.5 m) are the real
gate. Static flicker is not in issue #32 yet; it is the comfort metric that decides between
option 2 and option 3 later and should be added there.

### Then (2)

Offline first, from recordings made with `e2e/slam_drive.py` in its record mode: gsplat with truth poses and depth
init (oracle), then with SLAM poses. Pass if held-out PSNR beats today's splats by >= 4 dB and
LPIPS by >= 0.1 at 0.25 m offset, static flicker stays under 2 grey levels, and ghost fraction
does not rise. Online after that: optimisation on the last 4 to 8 keyframes in the worker,
50 to 100 iterations per keyframe, publishing through the ring; pass if the memory keeps up with
the keyframe rate and holds the render budget above.

## 5. Tracking and licence notes for the owner

- Doubled walls are a tracking problem (issue #33). The representation work above makes ghosts
  less visible (the live layer covers the forward view) but does not remove them from memory.
- If Ito is ever sold, or shipped on Steam for money, the CC BY-NC-SA MASt3R stack must go. The
  commercial-safe set found here: Video Depth Anything Small, Depth Anything V2 Small,
  Depth Anything 3 Small/Base/Metric-Large/Mono-Large, MapAnything Apache weights, Metric3D v2
  (BSD-2), gsplat, Open3D, nvblox (Linux), VGGT-1B-Commercial (gated, no military use). MoGe-2
  and Prompt Depth Anything need their weight licences confirmed on Hugging Face.

## 6. What I could not verify

- No inference timings on any GPU: every 4070 Ti figure is an estimate from published A100/3090
  numbers. The 12 GB shared budget with MASt3R-SLAM plus a depth net plus gsplat is unmeasured.
- Hugging Face and arXiv were unreachable from the sandbox: weight licences for the metric
  Depth Anything V2/Video checkpoints, MoGe-2 and Prompt Depth Anything are read from GitHub
  READMEs where stated, otherwise marked unknown. Runtime tables of MonoGS, Splat-SLAM, HI-SLAM2
  and Flash-Mono were not re-read; claims about them are qualitative.
- No headset: the shimmer and comfort judgements are from experience with these representations,
  not measured on Ito.
- The ToF check is synthetic (smooth warp error model, ideal zones). The Microduck's ToF
  extrinsics and real behaviour are unknown; the sim has no ToF yet.
- I did not run Ito's renderer or the e2e scripts here; nothing in this report changes code.

## Sources

- Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2 (licence section; metric
  models in `metric_depth/README.md`)
- Video Depth Anything: https://github.com/DepthAnything/Video-Depth-Anything
- Depth Anything 3: https://github.com/ByteDance-Seed/Depth-Anything-3
- Prompt Depth Anything: https://github.com/DepthAnything/PromptDA
- MoGe / MoGe-2: https://github.com/microsoft/MoGe
- UniDepth: https://github.com/lpiccinelli-eth/UniDepth (LICENSE: CC BY-NC 4.0)
- Metric3D: https://github.com/YvanYin/Metric3D (LICENSE: BSD 2-Clause)
- VGGT: https://github.com/facebookresearch/vggt
- Pi3: https://github.com/yyfz/Pi3
- MapAnything: https://github.com/facebookresearch/map-anything
- MASt3R: https://github.com/naver/mast3r (LICENSE, CHECKPOINTS_NOTICE)
- MASt3R-SLAM: https://github.com/rmurai0610/MASt3R-SLAM (LICENSE.md: CC BY-NC-SA 4.0)
- CUT3R: https://github.com/CUT3R/CUT3R ; StreamVGGT: https://github.com/wzzheng/StreamVGGT
- gsplat: https://github.com/nerfstudio-project/gsplat (Apache 2.0)
- 3D Gaussian Splatting: https://github.com/graphdeco-inria/gaussian-splatting (Inria licence)
- 2DGS: https://github.com/hbb1/2d-gaussian-splatting (Inria licence)
- SuGaR: https://github.com/Anttwo/SuGaR (Inria licence); Gaussian Frosting:
  https://github.com/Anttwo/Frosting
- AnySplat: https://github.com/InternRobotics/AnySplat (MIT code)
- DepthSplat: https://github.com/cvg/depthsplat (MIT code)
- Flash-Mono: https://proceedings.iclr.cc/paper_files/paper/2026/hash/5fcacc84d87c97bee5baecb3fb8f6e88-Abstract-Conference.html ,
  project page https://github.com/victkk/flash-mono (no code at time of writing)
- MonoGS: https://github.com/muskie82/MonoGS (Imperial College licence, non-commercial)
- Photo-SLAM: https://github.com/HuajianUP/Photo-SLAM (GPL-3.0)
- Splat-SLAM: https://github.com/google-research/Splat-SLAM (Apache 2.0)
- HI-SLAM2: https://arxiv.org/abs/2411.17982
- nvblox: https://github.com/nvidia-isaac/nvblox (Apache 2.0)
- Open3D: https://github.com/isl-org/Open3D (MIT)
- VL53L5CX: https://www.st.com/en/imaging-and-photonics-solutions/vl53l5cx.html
