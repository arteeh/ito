# Reconstruction alternative 2: a sparse voxel radiance field

The strongest distinct challenger is a **sparse voxel radiance field**: build the remembered scene from small cells containing color and opacity, refine those cells from incoming video, and draw each eye through them in the correct depth order. This could make fine detail and head movement steadier than Ito's current blobs without needing another camera or a large view-generation network. However, I found no evidence that it beats the alternatives already being tried within Ito's live computing budget. Its hardest problems are becoming sharp quickly, fitting enough detail in memory, and providing a freely distributable implementation. I recommend a short, staged rejection experiment before considering a reconstruction rewrite.

Research date: 13 October 2026. Checkout `07ca8a3`; measured main baseline `316a556`; round 1 read at `87d0f90`. This report changes no product code. I read the requested evidence, inspected all eleven committed comparison sheets, checked primary papers/code/licences, and did CPU arithmetic. **No GPU workload, simulator, Ito process, model inference, or timing experiment was run.** All timings below are published results with their scope stated, or explicitly proposed budgets.

Prior proposals: [architecture review](../big-picture.md), [depth-mesh issue #47](https://github.com/arteeh/ito/issues/47), and [round 1 at its reviewed revision](https://github.com/arteeh/ito/blob/87d0f90085749a2daba578730541122bf3686414/docs/research/recon-alt-1.md).

## What the local evidence establishes

The [true-pose sweep](../big-picture/img/sweep180-main-L1.jpg) loses book labels and floor grain despite having correct camera poses. The [measured sweep](../measure-img/sweep-reference.jpg) additionally shifts the bookshelf and loses image borders. In the [near-wall stereo pair](../measure-img/tour-stereo.jpg), the left eye contains a large bookshelf where truth contains mostly wall. That needs better geometry and registration; changing the drawing primitive alone cannot fix it.

The [optimized sweep](../big-picture/img/sweep180-optimized.jpg) restores book detail, but the [optimized tour](../big-picture/img/tour-optimized.jpg) has cloudy walls, especially with older fused poses. These are whole-recording, 180-second fits, not causal live results. The [hard frame](../measure-img/hard-526.jpg) and [missed-publication version](../measure-img/hard-526-missed-publication.jpg) show why first useful appearance and steady-state sharpness need separate deadlines.

[Step 1](../measure.md) gives 14.80–22.33 dB for the six camera references, SSIM 0.703–0.804 and LPIPS 0.436–0.603. Its RTX 4070 Ti stereo medians around 3.4–3.7 ms use **360×640 per eye, isolated from reconstruction and the compositor**. They do not prove full-resolution 90 Hz. I also inspected [the results JSON](../measure-results.json): tour 2284's left/right PSNR is 13.71/18.98 dB, and its visible observation-age p95 is about 23.7/23.9 seconds. Publishing a new map does not make every visible surface new.

I would not attribute those stereo errors to Gaussian sorting without an intervention. The evidence already demonstrates pose/depth errors; it does not isolate sorting as their cause. The voxel proposal removes a separate potential source of view instability and must earn its cost experimentally.

## Why this is a different direction

A **radiance field** describes how much light comes from a location toward an eye, and how much that location blocks what is behind it. A **voxel** is a small volume cell. Store cells only where observations support scene content, subdividing them where finer detail is useful. Render the volume directly; do not turn it into Gaussians or a mesh.

| Existing work | What changes here |
| --- | --- |
| #38–#46: optimize persistent Gaussians | Optimize a disjoint spatial partition and its density/color values. Cells have fixed spatial support; surfaces change through density values and subdivision. Visibility follows actual ray/cell intersections rather than projected Gaussian centers. |
| #47: newest RGB-D mesh patches | No image-plane triangles, patch replacement, or texture atlas. Multiple observations constrain one volumetric field, including empty space and occlusion along rays. |
| Round 1: choose remembered source images per eye | Source images supervise reconstruction but are not the renderer's appearance representation. No per-eye photograph selection, second image layer, panorama, or learned layered view compositor. |
| Main's existing voxel hash in `rgbd.py` | That hash chooses/replaces Gaussian samples. Here the cells themselves form the rendered volume; changing the hash resolution in main would not implement this proposal. |

**There is real overlap with the nine-step plan:** independent motion estimation, retained evidence, persistent optimization, resource limits and dynamic invalidation are still necessary. I claim no novelty for those. The distinct hypothesis is that an explicitly partitioned volume gives better stereo visibility and stable fine detail for the same *total* budget than a field of overlapping Gaussian primitives. If it only reproduces the optimized mapper's results with more code, it loses.

Questioning the lasting-model assumption does not produce another eligible winner here. Rendering observed images directly is already round 1. Without either remembered image rays or remembered spatial appearance, looking away from the current narrow camera must show missing information or generated guesses. A lasting model is not intrinsically the goal, but a bounded volume is the strongest distinct way I found to retain that evidence. It need not reconstruct the entire building, recognize objects, or serve navigation.

## Research beyond the existing shortlist

These sources informed the decision; they are not all proposed dependencies.

| Direction and primary evidence | Implication for Ito |
| --- | --- |
| [SVRaster, CVPR 2025](https://svraster.github.io/) | Adaptive radiance voxels are a credible alternative to both Gaussians and neural networks. This is the selected representation family. Its full offline pipeline is not an online robot mapper. |
| [NeRF-SLAM](https://arxiv.org/abs/2210.13641) and its [implementation](https://github.com/ToniRV/NeRF-SLAM) | Monocular pose/depth uncertainty can supervise a continuously fitted radiance field. This supports feasibility in principle, not coexistence with full-resolution headset rendering. The reference launch examples include a multi-GPU configuration. |
| [DTAM](https://www.doc.ic.ac.uk/~ajd/Publications/newcombe_etal_iccv2011.pdf), [TANDEM](https://github.com/tum-vision/tandem) | Temporal images can yield dense geometry without a depth sensor. TANDEM fuses inferred depth into a signed-distance volume. These are useful geometry references, but a textured surface/mesh solution would substantially revisit #47, and camera rotation alone still provides no triangulation baseline. |
| [SNeRG](https://phog.github.io/snerg/), [SMERF](https://smerf-3d.github.io/), [MobileNeRF](https://mobile-nerf.github.io/) | Compiling a learned scene into a cheap display representation is attractive. SMERF first trains a Zip-NeRF teacher; MobileNeRF bakes learned features into polygon textures. Fast playback does not establish fast incorporation of a newly seen obstacle. No such offline teacher is included here. |
| [Microsoft's hologram-stability guidance](https://learn.microsoft.com/en-us/windows/mixed-reality/develop/advanced-concepts/hologram-stability), [OpenXR frame submission](https://github.com/KhronosGroup/OpenXR-Guide/blob/main/chapters/frame_submission.md) | Correct pose time, scene depth and consistent frame delivery are separate from map quality. Reprojection can correct small pose errors; it does not observe hidden surfaces. These principles apply to every candidate, not a new reconstruction alternative. |
| [NVIDIA DLSS frame-generation research](https://research.nvidia.com/labs/adlr/DLSS4/) | Generating intermediate video frames addresses temporal sampling. It does not by itself produce two geometrically correct novel eye views. Applying it to the robot video leaves the reconstruction problem intact. |
| [Teleoperation and Visualization Interfaces for Remote Intervention in Space](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.747917/full) | Model-mediated teleoperation can provide a useful local display, but the study distinguishes perceived workload from task performance. A comfortable simulated scene still needs checks against what actually happened remotely. |
| [Genie 3](https://deepmind.google/models/genie/) | Interactive world generation questions the need to reconstruct at all. Its advertised 720p, 20–24 FPS generated worlds do not supply trustworthy unseen robot observations or establish a local <8 GB stereo implementation. This fails the evidential requirement before performance is considered. |
| [LiteVoxel](https://arxiv.org/html/2511.02510v1), [SpaRV project](https://jianhengliu.github.io/Projects/SpaRV/) | Follow-up work exposes voxel memory/flat-region problems and the value of measured geometry. LiteVoxel reports lower peak memory in offline experiments; no released implementation was verified. SpaRV depends on LiDAR, reports a 24 GB setup, and lists paper/code as forthcoming. Neither is an available solution to Microduck's mono case. |

### What the strongest paper actually demonstrates

[SVRaster's paper, Tables 1, 2 and 4](https://arxiv.org/html/2412.04459v2) reports, on Mip-NeRF360 and RTX 3090 Ti:

| Configuration | Single-view FPS | Scene training | PSNR |
| --- | ---: | ---: | ---: |
| Reference 3DGS | 131 | 24 min | 27.45 dB |
| SVRaster base | 121 | 15 min | 27.33 dB |
| SVRaster fast rendering | 258 | 9 min | 26.87 dB |
| SVRaster fast training | 131 | 4.5 min | 27.08 dB |

The base voxel method's reported peak GPU memory is 3.9 GB versus 1.8 GB for 3DGS; model sizes are 1.8 versus 0.7 GB. Standard views contain roughly 1–1.6 million pixels. Table 2's corrected LPIPS values favor voxels, 0.219 versus 0.257. The paper also shows exposure-related brightness boundaries and floaters. None of these numbers measures live stereo or Ito.

Its important rendering property is direction-dependent Morton ordering: an ordering of disjoint cells that follows each ray through space. This removes incorrect primitive-order changes for a fixed field. It does **not** prevent flicker from optimization, changing resolution, wrong poses, or wrong density.

My inference: this is enough evidence for a challenger, not a superiority claim. Even the fast variant's 258 FPS is about 3.88 ms per published single view. Doubling that already spends about 7.75 ms before larger eye images, online learning or presentation; this arithmetic is a warning, not a stereo timing prediction.

Better view consistency is also not exclusive to voxels. [StopThePop](https://github.com/r4dl/StopThePop) and [VRSplat](https://arxiv.org/abs/2505.10144) address rendering consistency within the Gaussian family. These belong to the existing direction, so I am not offering them as new alternatives. They do strengthen the rejection criterion: if the voxel benefit is only a sorting improvement attainable with a smaller renderer change, the wholesale rewrite is unjustified.

## Concrete design for Ito

The following is an Ito adaptation to be evaluated. It is not a claim that SVRaster already implements streaming reconstruction, the caps, dynamic handling or the GL renderer described here.

```text
mono RGB + source identity/calibration + robot camera motion [+ optional ranges]
                            |
                   timestamped observation join
                    /                     \
        CPU motion / sparse tracking       newest-frame depth worker
            |                       Depth Anything V2 metric Small
     immediate pose                       |
            |                   RGB + depth prior + validity
            |                              |
            |              seed/update bounded radiance voxels
            |                 /                       \
            |       immediate evidence          local refinement
            |                 \                       /
            +---------- complete scene/pose revision --------+
                                                            |
predicted headset eyes --> per-eye volume rendering at 90 Hz -+
                                      |
                           OpenXR / desktop / ImGui
```

### Inputs, geometry and motion

Keep the protocol/link and source joins. A motion worker publishes the current robot camera anchor independently of reconstruction. Reuse `MotionBuffer` and the local fusion boundary, with CPU OpenCV feature tracks, robust relative-pose estimation and sparse triangulation replacing the MASt3R frontend on this candidate path. Fit only a small local window. Do not align the camera to an immature volume to make the volume appear self-consistent.

Robot kinematics/odometry supplies the immediate metric prior; visual tracks test local alignment and depth scale when translation makes it observable. Raw IMU is optional, and an odometry estimate already using that IMU must not be counted twice. Without robot pose, sparse visual motion and learned metric depth provide a weaker initialization through the same boundary. There is no mandatory ToF, floor-height rule or sim-truth input.

Select **Depth Anything V2 Metric Hypersim Small**, 24.8M parameters, FP16, one current image per inference. Start at short side 294, giving approximately **294×518 width×height** for the portrait recording, preserving its aspect ratio and transformed calibration. Keep original RGB at capture resolution. This is a metric initialization prior, not measured geometry or calibrated confidence. Its indoor training and frame-independent depth are important risks. Do not train a replacement model as part of this proposal.

Use cross-frame reprojection residuals, sparse triangulation, blur and timing uncertainty to decide whether an observation may refine existing static cells. Model predictions agreeing over several frames are correlated evidence. Start with ≤2 source-pixel p95 reprojection error for static admission; measure the admitted coverage as well as rejected error. Optional calibrated ToF constrains depth over its angular footprints and valid return ranges, never by pretending that 64 zones locate pixel-precise edges.

The physical camera's current [receipt timestamp](../../drivers/microduck/camera.py) is not an exposure timestamp. Constant offset fitting cannot remove variable camera buffering or unknown rolling shutter. The observation join must preserve this uncertainty. A representation change cannot compensate for misregistered measurements by decree.

### One volume, with a bounded amount of remembered evidence

Use a world-aligned adaptive octree: repeatedly divide a cube into eight children, allocating only useful cells. Begin with a 16 m root around startup; grow the root by powers of two when observed content crosses it, without changing physical cell coordinates. Root growth and reindexing happen in the worker, with the previous complete scene still available. Never choose bounds from the future trajectory.

Initial limits are **two million resident leaf cells**, at most **250,000 actively optimized cells** per working set, and **96 retained source frames** in CPU memory. These are hypotheses to test. Prefer the current neighborhood and previously observed directions around the pilot; evict redundant, unsupported or distant detail before useful angular coverage. Retained geometry is bounded local memory, not an unbounded room archive.

A cell carries spatial index/level, density samples, RGB appearance, source support, last-observed time, geometric uncertainty and a static/provisional state. Begin with view-independent RGB. Permit degree-1 spherical harmonics—12 numbers describing color versus viewing direction—only after several translated views support them. Do not let view-dependent color disguise wrong depth in a pure pan. The [reference model-creation interface](https://github.com/NVlabs/svraster/blob/3503131b5558f5e5eaaab01881129a5f3bd79ad3/articles/model_creation_tutor.md) supports supplied points/cells, color and density; it also demonstrates that overlapping parent/child cells break the expected ordering.

**Keep a single disjoint leaf partition.** A parent and its children never render simultaneously. Static and transient evidence use the same spatial partition rather than two overlapping volumes drawn in an arbitrary order. A moving foreground can occupy nearer cells while the previously observed background remains behind it. Within a contradicted cell, current transient evidence supersedes the old visible parameters; parked history is not automatically valid again when the transient expires.

Geometry uncertainty is metadata, not a broad translucent cloud displayed as an object. Initialize a narrow provisional density band around inferred depth, color it from the current image, and mark its support honestly. Very uncertain depth or registration uses the existing framed video fallback. Empty allocated space is distinct from **observed free space**: an unallocated region is unknown, not evidence that the robot can move through it.

### Updating appearance without waiting for a long fit

Every usable depth result has a cheap insertion path. Back-project supported pixels, initialize new cells, update current color where image correspondence is consistent, and publish affected blocks. This makes first visibility independent of convergence. It does not promise first-frame sharpness: the proposed stable-detail deadline is 0.5–1 second of available observations.

Refine only the active region against a bounded replay of already received images. Start with 2,048 sampled rays per optimizer microbatch, prioritizing new edges and then older supporting views. Optimize density/color with robust RGB residuals, weak depth-prior penalties, supported sparse metric depths, empty-space evidence and edge-aware spatial regularization. Fit per-frame exposure gain/bias so brightness changes do not become geometry. Reject dynamic residuals from static training. No target-eye RGB or future image enters fitting.

An inferred depth alone must not carve away a previously supported obstacle. Keep its free-space penalty soft and stop before its uncertain surface interval; stronger deletion needs consistent registered observations or valid sensor evidence. An occluding foreground does not contradict the remembered surface behind it.

Allocate detail according to **source** sampling and supported multi-view geometry. Aim for approximately one source pixel per near textured cell, with coarser flat/far regions. Subdivision copies the parent's represented field into children before refinement; it must not visibly reset color. Pruning requires sustained low contribution and retained coverage. Current-view opacity alone must not delete surfaces that are merely occluded or behind the pilot.

For scale/pose corrections, keep source IDs and a spatial index of affected blocks. Rebuild/refine corrected blocks against their retained observations in a staging revision. A common coordinate-frame change can transform the eye and map together; an actual relative geometry correction requires rebuilding. Do not shear the whole grid or slowly slide incorrect shelves to hide a coordinate mismatch. If necessary evidence has been evicted, invalidate the affected content and reacquire it.

There is a hard freshness problem here: a newly moving object may require new depth, while a new texture on an existing wall may not. CPU reduced-resolution correspondence can authorize RGB-only updates to existing cells; it must not paint a moving object onto that wall. New or contradictory geometry requests the next depth slot. No future video frame is awaited for interpolation.

### Display and publication

The display receives immutable revisions containing voxel-block changes, topology changes, compatible pose/coordinate revisions and observation ages. Use bounded shared-memory staging and GL uploads initially. Upload into inactive storage with polled fences; swap only after a complete revision arrives. Avoid whole-scene copies at every update. Limit each display upload to 4 MiB, with larger revisions spread across frames; publication delay from this limit is measured, including cold-start views. A stalled publisher leaves the last coherent volume visible.

At every OpenXR deadline, obtain both predicted eye poses, actual eye projections and IPD. Use the existing robot-position/comfort anchor and current pilot head motion. Robot camera rotation places observations and is not added to pilot gaze a second time. Keep the anchor policy identical between candidates so a smoothing change cannot masquerade as a reconstruction improvement.

The proposed production renderer is **OpenGL 4.3 compute**, preserving Windows/Linux and Mesa fallback. Project cells into tiles, construct per-eye lists, order them by ray-direction octant and Morton code, then integrate density/color over actual ray/cell intersections. Tiles crossing a direction-sign boundary must use the appropriate order for each ray. Near-plane clipping and an eye inside a cell need explicit handling. Do not reuse main's midpoint depth sort for stereo.

Budget at most eight million cell/tile references per eye and preallocate sort/scratch capacity. Overflow must be detected before drawing a partial scene: hold the previous usable revision, request supported coarsening, or enter fallback. Do not silently drop foreground cells to meet a timing number. Mixed-resolution boundaries need matching density values and image-based checks; correct ordering alone does not make them continuous.

Start with spatial antialiasing and stable subdivision thresholds, charging its full pixel cost. Avoid temporal accumulation that smears a newly revealed edge. Output per-eye RGB, meaningful depth, supported coverage and observation-age buffers for evaluation. No neural view synthesis, mesh extraction, or second source-image compositor runs in this display path.

**This renderer is substantial new work.** The reference CUDA rasterizer can evaluate the research hypothesis, but its timing cannot be attributed to a GL port. A maintained, permissively licensed implementation meeting all these requirements was not found.

## Components, licences and memory

Non-commercial model permission is acceptable here; it must not be confused with permission to ship research-only source as a general piloting application.

| Piece | Verified licence and intended role | GPU memory |
| --- | --- | --- |
| [NVlabs SVRaster](https://github.com/NVlabs/svraster/blob/3503131b5558f5e5eaaab01881129a5f3bd79ad3/LICENSE) | NVIDIA custom source licence; section 3.3 restricts use to research/evaluation, with inherited Inria terms. Suitable as an experimental reference, **not the proposed freely usable product dependency**. It has no required pretrained reconstruction weights. | Published figures above; do not treat them as the online budget. |
| [SVRaster WebGL viewer](https://github.com/samuelm2/svraster-webgl/blob/f751b4c1a29c4c69a7da28ac0cc2a51fb24c955e/LICENSE) | MIT. Possible GL scaffolding only. Its [README](https://github.com/samuelm2/svraster-webgl/blob/f751b4c1a29c4c69a7da28ac0cc2a51fb24c955e/README.md) explicitly uses distance sorting, not direction-dependent Morton ordering. It is not a ready implementation of this proposal's key benefit. | Must measure after changing the renderer. |
| [Plenoxels/svox2](https://github.com/sxyu/svox2/blob/ee80e2c4df8f29a407fda5729a494be94ccf9234/LICENSE) | BSD-2-Clause. Reusable volume-rendering/optimization reference; its regular sparse grid is not the adaptive rasterizer. No pretrained scene model needed. | No second resident mapper. |
| Production voxel forward/backward kernels | Independently implement the published method; propose MIT for these new kernels and retain compatible reused components' notices. Do not copy/relabel the restricted reference kernels. This work is included in effort below; the product route does not depend on obtaining an exception. | Mapper/scratch allocations below. |
| [Metric Depth Anything V2 Hypersim Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small) | Checkpoint Apache-2.0; [code licence](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/LICENSE) Apache-2.0. Selected weight repository revision `3bc65d4e14a6786a61acec16453c50e12bf5f338`. Base/Large weight terms differ; they are not selected. | 24.8M FP16 parameters ≈47.3 MiB. Allow **1.5 GiB** for the whole worker, not just weights; unmeasured. |
| [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE), [OpenCV](https://opencv.org/license/), [ModernGL](https://github.com/moderngl/moderngl/blob/main/LICENSE) | BSD-style, Apache-2.0, MIT respectively; preserve third-party notices. CUDA runtime retains NVIDIA terms. CPU tracking uses no learned matcher. Existing OpenXR/ImGui/link dependencies stay. | Included in process allocations below. |

The independent implementation requirement materially weakens the recommendation. The MIT viewer does not relicense NVIDIA's trainer. [DirectVoxGO's licence](https://github.com/sunset1995/DirectVoxGO/blob/main/LICENSE) also starts with a research-only restriction above GPL text; I would not assume it solves this packaging problem. The research report can recommend the algorithm while declining to recommend those source distributions for production.

Initial **limits, not measured usage**, for native stereo up to an illustrative 1920×1920 per eye:

| Allocation, including all processes | GiB |
| --- | ---: |
| Depth worker: weights, activations, allocator, context | 1.50 |
| Voxel learner: parameters, gradients/moments, block metadata and training scratch | 1.50 |
| Render volume, inactive changes, stereo binning/sort and upload scratch | 1.00 |
| Eye targets, antialiasing/resolve and UI buffers | 0.75 |
| Decode, retained GPU training images and other Ito allocations | 0.50 |
| Runtime/compositor allowance | 0.75 |
| Unassigned headroom | 0.75 |
| **Total planning envelope** | **6.75 GiB ≈7.25 GB** |

The hard limit is **<8,000,000,000 bytes**, counting cold-start, allocator reservation, rebuilds and recovery, not 8 GiB or Torch allocation alone. Actual runtime eye sizes replace the illustration. The compositor allowance is prudence for sharing this PC, not a claim that its use is controlled by Ito.

CPU arithmetic explains both the opportunity and the trap. Two million cells × 20 float parameters × 16 bytes for parameter/gradient/two Adam moments is about **0.596 GiB**, before indices and scratch; shared corners or a smaller active set reduce it. Two display banks at 64 bytes/cell add **0.238 GiB**. Thus coefficients alone plausibly fit. But rasterization lists, training intermediates, and subdivision copies can dominate. No full-grid densification is allowed to temporarily exceed the cap.

Fine color needs many cells. With `f=532.57`, a 1 mm cell at 0.5 m spans about **1.07 source pixels**. Twenty square metres sampled at 1 mm already need roughly **20 million surface cells**, before thickness or occlusion. Two million cells therefore cannot preserve that detail everywhere. Adaptive allocation is essential, and insufficient retained detail is a legitimate early kill, not justification to quietly expand memory.

## Latency and scheduling at 90 Hz

Separate head response, first event appearance and convergence. The last complete volume can respond to the head while inference stalls; new events cannot become visible before their evidence arrives.

| Display stage | Proposed allocation per stereo frame |
| --- | ---: |
| Late poses, revision polling and bounded upload submission | 0.5 ms |
| Both-eye cell projection, binning and ordering | 1.2 ms |
| Both-eye volume integration and spatial resolve | 3.3 ms |
| ImGui, final composition and submission | 1.0 ms |
| Scheduling and presentation margin | 5.11 ms |
| **Total period** | **11.11 ms** |

The **6 ms application target is unverified and aggressive**. Rendering must be measured at the actual eye resolution, including antialiasing, closest geometry and maximum list size. A 360×640 comparison or a paper's 258 single views/s is insufficient.

Aim for depth inference ≤20 ms p95, ordinarily 10 updates/s, increasing toward 15 only when shared-load measurements allow it. Admit new/changed geometry immediately after the one running depth job; a timer-only 10 Hz admission adds up to 100 ms and is unsuitable for event freshness. Drop obsolete pending requests. Run refinement as approximately ≤1 ms microbatches, initially no more than one per display period, with topology work broken into bounded jobs. If these budgets do not produce sharpness, the representation fails the experiment.

An optimistic native-30-FPS event budget is 33 ms to the next exposure, 40 ms capture/encode/network/decode, 5 ms validation, 20 ms residual wait for running inference, 20 ms new inference, 10 ms seeding/publication, and 11 ms to the next display: **139 ms before remaining presentation costs**. This is allocation arithmetic, not a measured p95 or a sum of guaranteed percentile bounds. Large revisions may take additional upload frames; measure them separately and fail the freshness gate if needed. Exposure-to-visible excludes the initial 33 ms; stable-sharp additionally includes refinement. An existing surface can receive verified color sooner.

Processes, low-priority CUDA streams and average duty limits do not reserve the GPU. Trace individual kernels, allocations, driver scheduling and GL fences while depth, refinement and stereo run together. An inference that repeatedly blocks display is unacceptable even at low frequency. Gate or replace that configuration; do not claim a 20 ms inference fits inside one frame's 5 ms margin. Motion-to-photon needs runtime/scanout evidence and later headset measurement.

## Failure behavior and irreducible risks

| Situation | Required behavior |
| --- | --- |
| Fast pilot turn | Render remembered volume immediately from current eyes. The camera's mechanical turn and new RGB update independently. Newly exposed unsupported directions remain visibly unknown. No automatic scan or forced robot motion. |
| Fast robot turn, blur, uncertain capture time | Propagate the motion prior; reject unreliable static fusion. Preserve last supported cells with their real ages. If anchor uncertainty makes immersive rendering misleading, use the existing framed camera panel. |
| Never-seen region or background revealed by an eye | Stable neutral unknown treatment, excluded from observed coverage. Do not close a doorway or fill a hole with learned room completion. A density prior alone is not evidence that a previously unseen surface was observed. |
| Moving object | Compare image motion to camera-induced motion. Keep inconsistent cells provisional, avoid static optimization there, invalidate contradicted old geometry immediately, and expire unsupported moving traces within 0.5 s. Revalidate parked background before restoring it. Uncertain cases may require the panel; frequent panel use fails. |
| Object changes while out of view | Retain explicit age/uncertainty. The volume cannot know whether the remembered object moved. No claim of current truth for unseen memory. |
| Tracking loss or relocalization | Stop fusion, retain one coherent revision, and keep head tracking independent. Enter the panel when robot-anchor uncertainty is too large; resume only with a coherent map/pose revision. Relative corrections rebuild blocks rather than snap the world. |
| Near-field stereo, reflections, thin leaves/wires | Both eyes intersect the same density field, but incorrect learned depth, coarse cells and view-dependent color can still produce wrong disparity. Preserve actual IPD/head translation; do not shrink them to conceal failure. |
| Worker stall/OOM or raster-list overflow | Keep input/control and display running; bounded last-good scene, then panel with status. Stop repeated failed allocations/restarts. On no-CUDA/Mesa, the fallback and reduced-volume display must work; CPU learned reconstruction at 90 Hz is not promised. |

The recorded FOV is **37.35° horizontal ×62.00° vertical**. A head turning at 90°/s exits the horizontal half-cone in about **207 ms** if the camera has not followed. A different representation cannot make the missing photons exist.

Depth accuracy remains decisive: `disparity = focal_length × IPD / distance`. At 64 mm IPD, a 20% depth overestimate causes about **5.68 source pixels** of disparity error at 1 m and **11.36** at 0.5 m. A pure rotation cannot resolve that uncertainty by multi-view triangulation. The volume can be sharp, correctly ordered, and still geometrically wrong. Test startup pure pans and near objects before any helpful translation or optional ToF.

Other primary risks are density floaters, exposure seams, cell-boundary shimmer, correction rebuild cost and optimization taking too long. The voxel budget may lose to Gaussians because cells spend parameters on empty thickness and require dense color sampling. It may lose to round 1 because learned appearance must converge while cached images are already sharp. Those are central architectural tradeoffs, not polishing tasks.

## What code it would replace

| Ito boundary | Candidate change |
| --- | --- |
| `ito/reconstruction/slam.py`, `mast3r_frontend.py`, `mast3r_runtime.py`, `slam_vision.py` | CPU sparse frontend plus one depth-prior worker on the primary mono path. Remove resident MASt3R from the candidate's budget. |
| `ito/reconstruction/rgbd.py` | Replace Gaussian placement/carving with voxel evidence insertion, contradiction handling and local radiance optimization. Supplied RGB-D uses this same volume without mandatory learned depth. |
| `fusion.py`, `motion.py`, `tracking.py` | Reuse motion/fusion boundaries; adapt factors and uncertainty, publish propagation independently, and remove floor-derived scale from this path. |
| `reconstruction/__init__.py`, `ring.py`, `render/scene.py` | Replace 16-float splat records with versioned voxel-block/topology publication. Preserve bounded queues, failure reporting and isolated workers. |
| `render/renderer.py`, splat/sort/update shaders | New volume renderer and diagnostics buffers. The independent PLY viewer can retain Gaussian drawing. |
| `app/frames.py`, `app/pilot.py`, `xr`, `desktop`, `render/video.py` | Keep joins, safety/input, late predicted-eye loop, ImGui and panel fallback; wire the new scene boundary. No new game engine or UI toolkit. |

This is a replacement reconstruction/display path, not a permanent voxel subsystem beside the complete Gaussian pipeline. During evaluation, keep main intact in its frozen checkout and run separate candidate outputs.

## Judge it with #38; reject it before a rewrite

All experiments here are **future work after the ongoing GPU timing run releases the machine**. Do not run them concurrently. Keep standalone evaluation drivers in `e2e/`, not unit tests. No experimental product mode is added by this research commit.

1. **Preserve the measurement contract.** Reuse `measure_reference.json`, the fixed admissions, causal deadlines, source/calibration checks and audit conventions in `measure.py`, `measure_capture.py`, `measure_source.py` and `measure_audit.py`. Add a separate voxel adapter producing RGB/depth, coverage, source-age and pose/publication traces for `measure_images.py` and the timing format of `measure_display.py`. The frozen main runner intentionally fingerprints product code; do not weaken it or convert voxels into splats to satisfy the old snapshot interface.
2. **First isolate representation and cost.** Give the voxel candidate true poses and depth for already admitted source images, with charged insertion/optimization/publication time and the proposed memory cap. Compare the seven fixed moments: sweep 215/453/**526**/693; tour 363/1501/2284. Include camera views, both eyes at the configured 64 mm IPD and ±3 cm offsets. This oracle isolates the representation and cannot establish monocular feasibility. Future target-eye images never enter the map.
3. **Charge time before judging quality.** Save first useful, 100 ms, 500 ms and 1 s checkpoints after evidence admission. Do not substitute a 15-minute fit. Run the original CUDA reference for an upper-bound rejection experiment if useful, clearly labelled research-only; passing it is not passing the eventual GL renderer. The code inspection found supplied-point/cell initialization, so this test need not begin by building SLAM.
4. **Then replace depth and pose independently.** Run true pose/predicted depth, estimated pose/true depth, then both estimated. No truth-fitted scale, future calibration or retrospective pose replaces the estimate available at the original deadline. Include pure-pan startup, near-floor/near-wall views and no-ToF operation. Keep an oracle pose/depth ablation for diagnosis, not acceptance.
5. **Compare against the real competing investments.** Once available, run the optimized Gaussian mapper, #47 and round 1 with the same admitted RGB, pose/depth priors and total resource limits. Their prior results are not directly comparable to new voxel scores. Beating current main alone does not justify choosing this more expensive rewrite.
6. **Inspect the experience before aggregate scores.** Save unlabelled 1:1 truth/main/candidate sheets, then blind the labels for book labels, floor grain, doubled edges, near-wall stereo and holes. Reuse the three clip types: frozen volume/moving head, fixed head/updating volume, both together, plus the observed-bookshelf versions. Add octant crossings, cell subdivision, a camera inside a coarse cell, exposure changes, a moving foreground, a 0.5 s dropout and a revisit. Headset inspection follows, not precedes, these checks.
7. **Use two workload tracks.** The frozen #38 schedule supplies every sixth image, about 5 Hz; every candidate receives exactly those images, including any color fast path. Its 200 ms sampling interval cannot establish a 150 ms event-freshness target. Native 30 FPS replay is a separate experiment with newest-frame dropping and recorded availability. Legacy recordings substitute source timestamps for missing arrival/exposure times and cannot certify physical event freshness. Capture instrumented Microduck sim data in `ito-microduck` later, then a real-robot recording; sim depth is evaluation-only in the deployable mono run.

Full-frame PSNR/SSIM/LPIPS include holes; also report candidate-independent common-visible masks, previously observed holes and never-seen regions. Opaque neutral fill is not observed coverage. Contribution-weighted source age remains separate from optimization/publication age. Preserve SGBM matched fraction and flag its 128-disparity range limit; use geometric depth/edge diagnostics for near objects beyond that range. Rendering agreement with a wrong field is not geometry accuracy.

Predeclare these gates; their thresholds are engineering decisions, not published comfort guarantees:

| Gate | Continue only if… |
| --- | --- |
| **Oracle rejection, first 3–5 working days** | Within 1 s of available evidence, each sequence gains ≥3 dB mean full-frame PSNR and ≥20% mean LPIPS reduction over main across its predefined camera/eye views; no eye loses >0.5 dB; previously observed holes rise by ≤0.5 percentage point. Book labels/edges visibly improve. If only long offline fitting succeeds, stop. |
| **Early rendering rejection** | First test the likely best CUDA case, then the actual GL candidate. At runtime eye resolution, under concurrent seeding/refinement/depth load: <0.1% missed 11.11 ms frames, no consecutive misses in reference runs, and <8 GB peak total residency. Report p50/p95/p99/p99.9, longest stall, list sizes and uploads. Failure in the favorable reference case is an early stop; success there is only provisional. |
| **Mono usefulness** | With no ToF, retain ≥2 dB mean gain on each sequence and ≥20% LPIPS improvement in predefined textured regions. Halve main's p95 near-field disparity/foreground-edge error without lowering matched fraction. Reject if the gain requires truth, a smaller IPD, extensive input rejection or a helpful pre-scan. |
| **Distinct value over the work already planned** | At equal input/compute, beat the strongest available alternative by ≥1 dB or ≥20% LPIPS on both sequences **and** improve the relevant temporal-edge diagnostic by ≥20%, with no visible popping/coverage/freshness regression. If corrected visibility is not a visible benefit, prefer the simpler competing implementation. |
| **Freshness and continuity** | Native event depiction ≤150 ms p95; first-visible p95 no more than 10 ms worse than main; no contradicted moving trace >0.5 s. Zero future publications/observations, no black missed-publication frame in supported space, ≤1% fallback frames on the ordinary already-observed route. Count fallback pixels honestly. |
| **Sustained use** | Ten-minute replay with bounded memory, revisits, corrections and recovery; three repetitions with no hidden capacity increases. Physical footage and actual-IPD headset A/B must preserve the visual gain and comfort. The pilot's worse experience kills it regardless of mean scores. |

The fastest useful rejection is therefore **oracle-initialized sparse voxels with charged one-second refinement and simultaneous full-resolution stereo**. If that cannot pass, a better depth network or tracker is not the next investment for this direction.

## Runner-ups and effort

**Runner-up 1: an online hash-grid NeRF, periodically compiled for fast display.** This is different from image caches and optimized Gaussians: a compact learned function represents the scene, with a compiled volume/polygon renderer. NeRF-SLAM supports online fitting; SNeRG/SMERF/MobileNeRF support cheaper playback. It loses because the combination adds a second convergence/publication problem: a new event must enter both the learned model and its compiled representation. None of the checked sources establishes that cycle under Ito's shared 90 Hz/<8 GB constraint. The selected voxel approach updates the representation it actually draws.

**Runner-up 2: a model-mediated predictive digital twin.** Use a local physics/scene model to render immediate robot motion, then correct it from observations. This could be excellent in a known factory or a fixed manipulation task. It loses for a Microduck exploring an arbitrary room: the environment is not supplied as CAD, unseen object states are unknown, and model mismatch can give a beautifully steady but false picture. It would optimize a different product requirement. MuJoCo's existing room must not accidentally become a privileged input to this competitor.

Estimate **3–5 engineer-days** for the first research rejection, then **another 1–2 weeks** for mono/concurrency comparisons if it survives. A positive result still leaves approximately **6–9 weeks** for a distributable adaptive forward/backward renderer and GL validation, **4–6 weeks** for bounded streaming mapping, revisions/dynamics/recovery, and **2–3 weeks** for physical timing/calibration and headset acceptance. For one experienced engineer, plan roughly **14–21 weeks total**, with substantial uncertainty; fail the early gates before spending that. This exceeds the existing alternatives largely because no ready FOSS adaptive rasterizer/trainer was verified.

The research decision is conditional: sparse radiance voxels are my one distinct challenger, and have a plausible route to beating current main. They are **not demonstrated to beat the optimized mapper or round 1**, and I would keep those investigations ahead of a wholesale voxel rewrite unless the short comparison produces decisive evidence.
