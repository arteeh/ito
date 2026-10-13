# Reconstruction alternative 1: render from remembered camera views

I would test a **layered camera-view cache with depth-guided reprojection** as Ito's primary view builder. Keep sharp camera images, remember useful earlier views, and use estimated depth to redraw those images from the pilot's current eyes at 90 Hz. This avoids waiting for a detailed room model to converge and should preserve more of the detail main currently blurs. It still needs accurate local motion and depth, and cannot show the truth in places the camera has never seen. My confidence is moderate: the strongest case is sharper, fresher nearby views with bounded compute; the decisive risk is unstable stereo near objects. Test that risk before committing to a reconstruction rewrite.

Research date: 13 October 2026. Checkout: `07ca8a3`; measured product baseline: `316a556`, pinned by [the reference manifest](../../e2e/measure_reference.json). This is a design recommendation, not a measured improvement. Work performed: local code/document review, inspection of all eleven committed comparison sheets, primary-source research, and CPU arithmetic. No GPU workloads, model execution, simulator, or Ito runs were used.

## What the evidence changes

In the [baseline sweep](../measure-img/sweep-reference.jpg), book labels and floor grain disappear, the picture frame becomes lumpy, and black borders grow during the turn. The [true-pose sweep](../big-picture/img/sweep180-main-L1.jpg) remains blurry: fixing tracking alone cannot restore detail discarded by the representation. The [tour stereo sheet](../measure-img/tour-stereo.jpg) is more concerning than its average score: the left eye pulls a large bookshelf into a view that should mostly contain wall. This is a wrong view, not just a soft image.

The [optimized tour](../big-picture/img/tour-optimized.jpg) recovers book detail with true poses, but its wall becomes cloudy; older fused poses bend and smear surfaces. Those checkpoints use whole-recording observations and 180 seconds of optimization. They establish appearance potential, not timely stereo reconstruction. The [hard moment](../measure-img/hard-526.jpg) and its [missed-publication counterpart](../measure-img/hard-526-missed-publication.jpg) show why keeping the last usable observations independently of mapper publication matters.

[Step 1](../measure.md) reports 14.80–22.33 dB PSNR for the six reference camera views, SSIM 0.703–0.804 and LPIPS 0.436–0.603. Its isolated RTX 4070 Ti stereo draws take roughly 3.4–3.7 ms median, **at 360×640 per eye without concurrent reconstruction or a compositor**. This is not a full-resolution 90 Hz result. [The JSON](../measure-results.json) also preserves seconds-old visible observations, and distinguishes causal poses from later refinements. These distinctions carry into this proposal.

The useful change of question is: **what evidence does the next pair of eye images need?** The pilot stays near the robot's current camera position. A globally consistent, editable room is not intrinsically required. A bounded collection of observed images plus sufficiently accurate local geometry may be enough. Geometry remains necessary for stereo and translation; removing a lasting map does not remove that requirement.

The reason this could preserve sharpness is geometric: at the source camera center, a change of viewing direction can reproject the original image without knowing depth. Depth error becomes visible as the target eye moves away from that center. Keeping original pixels avoids a separate appearance-fitting error, while leaving the stereo-depth error fully exposed to evaluation.

## Why this approach, after looking elsewhere

The following sources support parts of the design, not a claim that the assembled system already works on Microduck.

| Field / evidence | Lesson for Ito and limit of the evidence |
| --- | --- |
| [Panoramic View Reconstruction for Stereoscopic Teleoperation, 2016](https://developmental-robotics.jp/wp-content/uploads/publications/Theofilis_Humanoids16.pdf) | Remember camera views and decouple headset viewing from mechanical head motion. Its iCub setup does not establish monocular metric stereo on a walking Microduck. |
| [Layered Depth Images, 1998](https://www.microsoft.com/en-us/research/publication/layered-depth-images/) | An image can retain multiple observed surfaces along a ray, allowing background to reappear when the foreground shifts. This is a compact rendering representation, not a solution to obtaining unseen background. |
| [Unstructured Lumigraph Rendering, 2001](https://www.ics.uci.edu/~gopi/ICS280Win02/UnstructuredLumigraph.pdf) | Select source photographs according to the requested viewpoint and approximate geometry. Preserve image detail instead of baking every observation into one appearance field. Sparse, inaccurate sources still produce seams. |
| [Real-Time View Planning for Unstructured Lumigraph Modeling, 2019](https://doi.org/10.1109/TVCG.2019.2932237) | Incrementally retain views for spatial/angular coverage under a storage limit. Useful memory is not simply the latest N frames. |
| [Meta Passthrough+, 2020](https://ai.meta.com/research/publications/passthrough-real-time-stereoscopic-view-synthesis-for-mobile-mixed-reality/) | A shipping VR system used a coarse geometry proxy, filtered depth, source-image texturing and temporal upsampling; authors report 72 Hz on Snapdragon 835. Its multiple cameras and small camera-to-eye offsets make depth much easier than Ito's case. Borrow the separation of geometry and appearance, not its performance claim. |
| [NeuralPassthrough, 2022](https://github.com/facebookresearch/NeuralPassthrough) | Learned blending can improve near-field reprojection. The released Python implementation precedes the custom C++ real-time optimization, uses RAFT-Stereo, and is archived. It is not a ready mono replacement. |
| [Quark, SIGGRAPH Asia 2024](https://quark-3d.github.io/) | Strong recent evidence that temporary, view-conditioned layers can synthesize high-quality images without keeping a permanent scene. Its reported 1080p/30 FPS is on A100; dynamic examples have multi-view inputs. That does not establish 90 Hz stereo, mono temporal inputs, or <8 GB. |
| [DeepVideoMVS, CVPR 2021](https://github.com/ardaduz/deep-video-mvs) | Posed temporal image pairs and recurrent state can estimate depth without a Gaussian optimizer. Pure rotation supplies no triangulation baseline; the released stack also needs modernization. |
| [Online Video Depth Anything, 2025 preprint](https://arxiv.org/html/2510.09182v1) | Causal depth inference with bounded memory is credible: authors report 42 FPS and 0.45 GB for their context-16 configuration at 280×924 on A100. They also report scale drift and moving-object trails. Their evaluation aligns scale/shift using truth in the first frame; it is not evidence of autonomous metric depth. |
| [NVIDIA late-warp research](https://developer.nvidia.com/blog/nvidia-research-warp-drive-gaming-eliminate-more-than-80-of-the-latency-performance-penalty/) and [AMD FSR 3 integration](https://gpuopen.com/presentations/2023/AMD_FidelityFX_Super_Resolution_3-Overview_and_Integration.pdf) | Late pose sampling and image warping address response latency. Video frame interpolation is a different tool: intermediate frames do not supply the pilot's novel eye rays or unseen surfaces. Do not buffer a future camera frame to manufacture 90 FPS. |
| [HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay) | Interactive generated worlds are relevant to visual continuity, but generation does not establish what is physically behind a robot's unseen corner. I would not make generated detail the pilot's evidence, irrespective of model speed. |

This favors a conventional image renderer with one small depth model. It spends learning on the missing geometric cue and keeps actual camera colors. I would not train a new novel-view network for round 1.

### Relationship to the work already underway

**There is substantive overlap with [#47](https://github.com/arteeh/ito/issues/47):** both retain RGB and estimated depth and can rasterize textured depth grids. The depth model itself is not a new alternative. Merely renaming its patches “layers” would add nothing.

The proposed difference is the rendering and evidence policy:

| Existing proposal | This proposal |
| --- | --- |
| #47 places depth meshes in a world and lets newer patches replace overlapping older ones. | Retain original source views. For each requested view, choose the best supported source locally; a newer blurry or badly registered image does not erase a sharp static observation. |
| Each patch principally supplies its visible surface. | Keep a second **previously observed** surface for disocclusion—the background revealed when the viewpoint moves—and preserve source identity behind every layer. |
| Patch replacement determines appearance. | Use shared source choices for both eyes, temporal hysteresis, and projected-error gates to control source changes and binocular disagreement. |
| Geometry inference precedes a new patch. | Publish new RGB independently; update compatible existing depth through measured image motion, then replace depth asynchronously when inference finishes. |
| #38–#46 accumulate and optimize persistent Gaussians. | No Gaussian training, densification, room-wide appearance fitting or permanent fused surface. Original images remain the appearance representation. |

Independent pose publication, calibration, resource accounting and dynamic evidence are shared necessities already recognized by #38–#46. They are not new contributions claimed here. **Compare this directly with #47 using the same depth and poses.** If source selection, layers and independent RGB publication fail to buy visible stability/freshness, prefer the simpler #47 implementation. This alternative earns a rewrite only through that comparison.

## Concrete Ito design

### Data flow and ownership

```text
mono RGB ~30 Hz + timestamped robot camera motion ~50 Hz [+ optional ranges]
                         |
              frame identity / calibration / time join
                    /              \
     CPU motion + visual checks     newest RGB mailbox -----------------+
          |                         |                                  |
     current pose             asynchronous metric depth, 10–15 Hz       |
     and revisions                  |                                  |
          +----------> bounded source-view cache <--- flow / validity --+
                              |
                    complete revision + provenance
                              |
predicted headset eyes --> source selection + layered reprojection, 90 Hz
                              |
                  existing OpenXR / desktop / ImGui
```

Use a CPU motion/observation worker, one depth worker, and the existing display process. No worker waits for the depth result before publishing motion or decoded RGB. Queues have one newest pending frame; published immutable source slots remain alive until the renderer releases their revision. Sequence numbers and calibration/frame revisions prevent a late result from overwriting newer geometry or being attached to the wrong image.

Begin with bounded shared-memory transfers into GL pixel-buffer objects, polled fences and double buffering. At the recorded resolution, one RGB upload is small; this need not depend on fragile cross-process CUDA/GL handles. Keep image warping and rasterization in GL shaders. Readback of a completed low-resolution depth result is allowed in the worker and charged to latency; the display never waits for it. The reference model's `.cpu().numpy()` is a real transfer, not zero-copy inference.

### The cache

Keep at most **96 original views**: up to 32 recent frames plus 64 retained for useful direction, viewpoint, sharpness and coverage. This is an initial engineering cap, not a demonstrated optimum. A view stores calibrated RGB, axial depth in metres when available, camera pose, source timestamp/ID, invalid/dynamic masks, uncertainty evidence and pose revision. Static coverage can outlive the recent queue; a timestamp does not get refreshed merely because a source was rendered again.

The renderer builds at most two depth layers for an active direction: nearest supported surface, and a useful previously observed background. Store references to original RGB coordinates so cache maintenance does not repeatedly resample color. A newly hidden background is kept; a reliably contradicted surface is invalidated. Unknown background stays unknown. Two layers are a bounded approximation: foliage, fences and several overlapping objects can require more, and will be exposed by the coverage gate.

Select retention by contribution to the current neighborhood and angular coverage, not by a global voxel grid. A 180-degree scan should retain both ends. After robot translation, an old image remains usable only where its projection remains supported; do not drag a panorama along as if everything were infinitely distant. Evict redundant sources first, with an overlap period for any visible source being replaced. No automatic robot scan is required or initiated.

### Geometry and motion

Use **Metric Video Depth Anything Small**, FP16, the causal `video_depth_stream` path, initially at **294×518 width×height** for the portrait input, with RGB retained at capture resolution. Set the short-side input explicitly: upstream's default 518 is a *lower bound*, and would enlarge a portrait image's long side. Preserve intrinsics through resizing; never stretch the portrait frame to square. Aim for 10–15 depth updates/s in steady viewing, subject to the shared-load gate; newly exposed or changing regions request the next available inference instead of waiting for a periodic tick. Cold start and reconnection start from present observations only. Reset temporal features on calibration changes, reconnection or a substantial frame gap; irregular sampling and reset quality must be evaluated explicitly.

This model supplies a **metric prior**, not measured distance or calibrated confidence. Form validity from forward/backward image correspondence, depth reprojection disagreement, blur, boundaries, pose/timestamp uncertainty and available metric observations. Single-frame predictions get a conservative uncertainty floor; agreement between repeated predictions is not independent proof of correctness.

Keep robot camera kinematics/IMU as the immediate pose prior. Reuse the motion buffer and robust local fusion machinery, replacing MASt3R visual factors with OpenCV pyramidal feature tracking, robust geometric fitting and sparse triangulation against past views. Use translation-bearing tracks to constrain metric depth scale against robot motion; refine local relative poses from those tracks in the existing small solver. Pure turns constrain rotation, **not scale**. Use robust PnP only with supported 3D points, and do not count the same robot-derived scale/pose twice as independent measurements. A CPU tracker is an integration to validate, not an assumed accuracy upgrade.

Optional calibrated ToF ranges constrain depth over their actual angular footprints and time intervals. They must not create detailed edges by upsampling 64 cells. Without ToF, startup uses the model prior and openly lower geometric confidence; no floor-height assumption or sim truth enters deployment. For a robot without motion telemetry, the same local visual estimation boundary can supply relative poses, with weaker metric confidence. Neither sensor absence nor model failure changes the protocol into a Microduck-only contract.

Timing remains an upstream limitation. Current physical mediad frames are receipt-stamped. Fitting an offset against robot rotation can reduce a constant delay but cannot remove variable buffering or unknown rolling-shutter distortion. Inflate uncertainty and reject misleading cross-view combinations during these intervals. Camera movement about the neck also includes translation: it is not pure optical-center rotation.

### Rendering the two eyes

At each display deadline, locate both eyes for the runtime's predicted display time. Use the actual per-eye projections/IPD, the current robot camera anchor position, and the pilot's head rotation/translation. Keep the existing comfort motion policy for the initial comparison and log its lag separately. Camera rotation is used to place observations; it must not be added to the pilot's gaze a second time. OpenXR's [frame-submission guide](https://github.com/KhronosGroup/OpenXR-Guide/blob/main/chapters/frame_submission.md) explains why late location and the pose submitted with the rendered image must agree.

Select a small candidate set for each image tile, initially no more than four sources and two visible layers. Score viewing angle, available resolution, projected uncertainty, occlusion, sharpness and age. Compute ownership for the eye pair together, with per-eye visibility tests: shared ownership must not force a surface into an eye where it is occluded. A source change needs a meaningful score improvement and a short hold time, rather than alternating every frame.

Rasterize depth samples as adaptive textured grids using ordinary depth testing. Split at discontinuities and subdivide near silhouettes; never bridge foreground and background with a long triangle. These are disposable projection proxies, not fused world patches. Sample original RGB once, using mipmaps for minification. No learned color synthesis, Gaussian sorting or optimizer is on this path.

Blend overlapping views only where projected edges and depths agree. Starting thresholds are ≤1 source pixel for color blending and ≤2 source pixels for admitting a static cross-view match, scaled to the actual eye projection. Else keep one source while valid; if none is valid, expose uncertainty. A 50–100 ms transition is acceptable only for already aligned static imagery. Dissolving mismatched shelves produces two shelves and is not a stability solution. A common rigid coordinate correction can move cache and eye frame together without visual motion; inconsistent relative corrections require revalidation, not a global cosmetic warp.

New RGB can texture an old depth field only through verified correspondence. Use CPU OpenCV DIS flow at reduced resolution for the dense check and shader reprojection for depth propagation, initially for at most two 30 FPS intervals (67 ms). Flow residuals that disagree with camera-induced motion invalidate moving regions. No future RGB frame is awaited. New regions lacking depth receive the new model result when ready, or appear in the existing framed live-video fallback. Pasting a new image onto unrelated old depth would make apparent freshness dishonest.

## Models, licences and resource budget

| Selected piece | Licence / implementation | Memory evidence or planned allowance |
| --- | --- | --- |
| Metric Video Depth Anything Small, 28.4M parameters | [Code](https://github.com/DepthAnything/Video-Depth-Anything/blob/main/LICENSE) and [specific metric checkpoint](https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small): Apache-2.0. Use the Small checkpoint, not Base/Large. | FP16 parameters alone are approximately 54 MiB. **Allow 2.0 GiB total depth-worker GPU residency**, including cached features, activations, allocator and CUDA context; this is an unmeasured cap. |
| PyTorch inference runtime | [BSD-style licence and third-party notices](https://github.com/pytorch/pytorch/blob/main/LICENSE); CUDA runtime retains NVIDIA's terms. No TensorRT dependency is required by this design. | Included in the worker cap, not added a second time. |
| OpenCV feature tracking, DIS flow, PnP/triangulation | [Apache-2.0 for current OpenCV](https://opencv.org/license/); no model weights. | CPU work; no GPU optical-flow network alongside depth. |
| GL image cache and reprojection | New Ito code using existing [ModernGL, MIT](https://github.com/moderngl/moderngl/blob/main/LICENSE). LDI/ULR/Passthrough+ are algorithmic references, not imported licensed implementations. | Image-cache, scratch and render-target caps below. |
| Existing display/UI/transport | Retain existing dependencies and their notices. No game engine, remote service or new licensed novel-view renderer. | Decoder, GL and runtime allocations counted below. |

The [VDA README](https://github.com/DepthAnything/Video-Depth-Anything) reports **6.8 GB FP16** for Small on a **32-frame 518² A100 workload**. That configuration does not fit this plan with comfortable headroom. Its timing is not a one-frame consumer-GPU latency. The [streaming implementation](https://github.com/DepthAnything/Video-Depth-Anything/blob/main/video_depth_anything/video_depth_stream.py) consumes one current frame and retained hidden states, but upstream labels it experimental and reports an accuracy drop versus offline inference. **No source establishes its peak residency or deadline behavior at the proposed Ito settings.** Memory fit is a first experiment, not a claim derived from weight size.

The trained [oVDA checkpoint](https://huggingface.co/FriedFeid/oVDA) is a useful future substitution within this one boundary, not an additional resident network: Apache-2.0 portions plus Heidelberg NC-SA-UHDV1.0 for the remaining components. Its published low memory is attractive, but it outputs relative depth and needs a separate metric anchor. I selected metric VDA-S to make sensor-free startup concrete; oVDA's 0.45 GB result cannot be attributed to that selection. Non-commercial terms are acceptable for this research; they do not become permissive software terms because Ito is free of charge.

Initial **allocation limits, not measurements**, at up to 720×1280 source RGB and an illustrative 1920×1920 per eye:

| Allocation | GiB allowance |
| --- | ---: |
| Depth worker, all resident GPU allocations | 2.0 |
| 96 original views, depth/masks and color mipmaps | 0.8 |
| Active layers, tile lists, stereo targets and upload staging | 1.0 |
| Decoder and other Ito GPU contexts/buffers | 0.7 |
| Runtime/compositor allowance | 1.0 |
| Unassigned headroom | 1.0 |
| **Total planning envelope** | **6.5 GiB ≈ 6.98 GB** |

CPU arithmetic: 96 × 360 × 640 × `(4×4/3 + 2 + 1 + 1)` bytes is 0.192 GiB for RGBA8 with mipmaps, R16F depth and two byte masks; quadrupling image area gives 0.769 GiB. Extra provenance/layer buffers belong to the scratch allowance. This excludes no hidden second MASt3R or Gaussian optimizer: neither is resident. Charge actual runtime swapchains and all processes to measured total residency; **<8,000,000,000 bytes**, rather than quietly equating 8 GB with 8 GiB, is the hard gate. Runtime-recommended resolutions may exceed the illustrative one and must be tested as supplied.

### Latency at 90 Hz

Each display period is 11.11 ms. The initial engineering budget is:

| Display work | Target wall-clock allocation |
| --- | ---: |
| Late eye locations, revision polling, commands | 0.5 ms |
| Source selection / visibility preparation | 0.7 ms |
| Both-eye layer rasterization and source sampling | 3.0 ms |
| Resolve, validity treatment and aligned transitions | 0.8 ms |
| ImGui and submission overhead | 1.0 ms |
| Scheduling/runtime margin | 5.11 ms |

The 6 ms application target is **unmeasured**. Target depth inference ≤20 ms p95, averaging 10 Hz initially and increasing toward 15 Hz only if display slack permits. A 20 ms inference is not placed in one display frame's “spare” 5 ms. Separate processes and low-priority CUDA streams do not reserve the GPU; measure long kernels, preemption, transfer stalls and concurrent deadlines. If an uninterruptible inference repeatedly makes the display miss, this configuration fails even when average utilization looks low. Reducing inference cadence only helps if the per-inference stall also passes. No inference runs in the render loop.

For new events at 30 FPS, allocate 33 ms sampling wait, 40 ms capture/encode/transport/decode, up to 20 ms waiting for the one running depth job, 20 ms new depth inference, 10 ms validation/publication and 11 ms until the next display frame: approximately 134 ms before remaining presentation costs. This is a proposed **event-to-visible** envelope, not a measured exposure-to-display number or a sum of independently guaranteed p95s. Exposure-to-visible excludes the initial sampling wait. A periodic 10 Hz admission policy could instead add 100 ms and fail; event-priority admissions are required. If frequent changes demand more inference than the display budget permits, record that failure rather than silently promising 150 ms. Measure the complete distribution. Successful color propagation removes depth inference from some RGB updates but does not turn old depth into a fresh measurement.

The pilot's rotational response uses local pose every display frame, independently of that event latency. Actual motion-to-photon still depends on the compositor, headset transport and scanout. It cannot be inferred from GL completion, and 90 Hz pose response does not make a moving person update at 90 Hz.

## Failure behavior and the difficult limits

| Situation | Required behavior and remaining limitation |
| --- | --- |
| Fast head turn, camera lag/blur | Reproject retained clear views immediately. Update camera direction at its actual arrival time; do not wait on mechanical motion. Blur can depict a fresh event but cannot overwrite reliable static detail indiscriminately. Outside retained coverage, show the unknown region. |
| Never observed directions or newly uncovered background | Neutral, stable missing-region treatment with a coverage indication; no stretched edge pixels, invented room or claimed observation coverage. If usable coverage is too small, use the framed live camera panel. Startup cannot be a truthful complete room. |
| Moving person/object | New RGB owns verified changing regions; exclude them from static retention. Stop motion extrapolation beyond the short propagation horizon. Invalidate contradicted history immediately; fade unsupported moving traces within 0.5 s. An object that changed while out of view remains unknowable: old static memory must carry its age. |
| Near object or large head translation | Reproject at actual eye separation while supported. Invalid disocclusions stay invalid. Never silently shrink IPD or freeze head translation to make scores/comfort appear better. If uncertainty dominates, transition to the explicit panel fallback. Frequent fallback on the ordinary route fails the proposal. |
| Robot tracking/timing loss | Stop admitting misregistered sources. Continue local head rendering from the last coherent revision briefly; uncertainty grows with robot movement. If the anchor is no longer trustworthy, use the panel. Reacquire from overlapping observations and publish a coherent revision, not a snapping whole-room correction. |
| Depth worker stall/OOM or no CUDA | Display and controls continue; bounded last-good reprojection where valid, then the existing live panel with status. Repeated restart loops are suppressed. CPU/Mesa must still render the cache/fallback; high-quality learned depth on that machine is not promised. |

A few CPU calculations make the risks concrete. The recorded `f=532.57`, 360×640 camera has **37.35°×62.00°** FOV. At 90°/s, a gaze starting at the camera center exits the horizontal half-cone in **207 ms** if the robot camera has not caught up. No representation can supply an unobserved direction during that interval.

For rectified eyes, disparity is approximately `d = f × IPD / z`. With 64 mm IPD, a 20% depth overestimate causes **5.68 source pixels of stereo-disparity error at 1 m, 11.36 at 0.5 m, and 22.72 at 0.25 m**. Per-eye texture can therefore look sharp while stereo is wrong. The error grows in pixels when magnified into a headset. Pure camera rotation leaves depth unobservable from correspondence, and “metric” learned depth can still be wrong. This is the principal reason confidence is moderate, and the near-floor/near-wall test is first.

Other material risks are source-switch flicker, flow trails, reflections, exposure seams, and variable camera delays. A moving robot can leave the small region in which cached views are useful; an optimized persistent model might then win decisively. The proposal trades room-wide consistency for local appearance/freshness and must pay for that trade in measured coverage, not conceal it behind a vignette.

## What it replaces in Ito

| Current boundary | Proposed change |
| --- | --- |
| `ito/reconstruction/slam.py`, `mast3r_frontend.py`, `mast3r_runtime.py`, `slam_vision.py` on the primary mono path | Replace dense MASt3R tracking/point production with the CPU local estimator and one causal depth worker. Keep main intact for controlled comparisons until the candidate passes. |
| `ito/reconstruction/rgbd.py` map insertion, carving and point-splat lifetimes | Replace with bounded source retention, masks, contradiction handling and observed background-layer references. Optional supplied RGB-D feeds this same cache directly. |
| `ito/reconstruction/fusion.py`, `motion.py`, `tracking.py` | Reuse timestamped motion and robust local optimization concepts; revise visual-factor/scale handling for sparse metric constraints. Remove the floor-scale dependency from the selected path. This is real estimator work. |
| `ito/reconstruction/__init__.py`, `ring.py`, `ito/render/scene.py` | Replace splat-delta publication on this path with source-slot updates and atomic compatible revisions; independent pose/RGB publication. Keep bounded queues, lifecycle reporting and worker isolation. |
| `ito/render/renderer.py`, splat/sort/update shaders | Replace primary scene drawing with source selection, depth-grid projection and image compositing. A PLY viewer may retain its independent Gaussian renderer. |
| `ito/app/frames.py`, `pilot.py`; `ito/xr`; `ito/desktop`; `ito/render/video.py` | Preserve frame joining, input/safety, predicted-eye loop, ImGui and panel fallback. Wire the new scene source and expose age/coverage/failure state. Audit timestamp semantics rather than assuming the current join establishes exposure. |

The driver/link architecture stays useful. Extra sensor observations remain optional. This is a replacement view pipeline, not a mesh layer running alongside a complete Gaussian mapper forever.

## Judge it with #38, and kill it cheaply

All experiments below are **proposed for after the concurrent GPU measurement is finished**. None was run for this report. Keep outputs outside Git except concise result summaries/sheets. Standalone drivers belong in `e2e/`; no unit tests or product experiment switch is added by this research commit.

1. **Preserve the baseline contract.** Reuse the fingerprints, source identity/calibration checks, fixed admissions, causal deadline audit and measurements in `e2e/measure.py`, `measure_capture.py`, `measure_source.py`, `measure_images.py`, `measure_audit.py` and `measure_display.py`. Their renderer/snapshot code is Gaussian-specific: add a narrow candidate adapter producing RGB, observation coverage, depth, source IDs/ages and pose/publication traces. Do not convert the candidate into splats just to fit the harness. The unchanged baseline runner intentionally rejects changed product trees; use a separate candidate command and output directory.
2. **First compare the representation with oracle geometry.** Use identical admitted camera images with true poses/depth for both the image cache and #47's patch policy. Compare against frozen main as well. Reuse the seven moments: sweep 215/453/**526**/693 and tour 363/1501/2284, both eyes at 64 mm and ±3 cm head shifts. No target-eye RGB enters the cache. Oracle results isolate representation and cannot establish mono feasibility.
3. **Replace depth and pose separately, then together.** Run true pose + predicted depth, estimated pose + true depth, and the full mono pipeline. No truth-derived scale fitting, future frames or retrospective pose corrections enter causal snapshots. Include a startup pure pan and near wall before helpful translation/ToF. This is where a source-image copy that only scores well at its own camera should fail.
4. **Keep two workload tracks.** The frozen #38 comparison admits every sixth image, about 5 Hz. Give every candidate the same images, including its RGB fast path. Separately run native 30 FPS replay with recorded arrival times and newest-frame dropping, charging flow, inference, CPU solving and publication. The legacy recordings lack arrival/exposure times: a new instrumented recording is needed for that claim; source timestamps remain only an explicit assumption in legacy replays. At 5 Hz, 150 ms event freshness is sampling-impossible; do not award a candidate extra RGB inputs or infer native freshness from that test.
5. **Show the experience.** Make truth/main/candidate/#47 sheets at 1:1 and the existing three clip types: frozen cache with moving head, fixed head with updates, and both. Include the observed-bookshelf clip. Blind the candidate labels before inspecting book labels, floor grain, single/doubled edges, view boundaries and left/right disagreement. Full-frame PSNR/SSIM/LPIPS include holes. Use candidate-independent common-visibility masks and original observation age. Neutral fallback pixels are not observed coverage.
6. **Then share the GPU and broaden the route.** Reuse the stereo timing format under simultaneous depth, tracking, uploads and rendering at actual runtime resolution, with cold/warm starts and three repetitions. Report p50/p95/p99/p99.9, missed deadlines, longest miss run, peak aggregate residency, decoded-RGB-to-visible and exposure/event timing separately. Add a moving object crossing a remembered wall, 0.5 s loss, timing/pose perturbations, revisits and a ten-minute bounded-memory loop. Later capture those cases through the Microduck sim in `ito-microduck`, then the real robot; sim depth is evaluation-only in the deployable run. Headset A/B remains necessary after these gates.

Predeclare the following decision criteria. Numerical thresholds are engineering choices, not published comfort guarantees.

| Gate | Continue only if… / early kill |
| --- | --- |
| **Oracle value, first 3–5 engineer-days** | Across each sequence's fixed reference views, ≥3 dB mean full-frame PSNR gain and ≥20% mean LPIPS reduction over main, including held-out eyes; no reference eye loses >0.5 dB, and observed-region holes increase by ≤0.5 percentage point. Labels/edges must visibly improve. If even oracle geometry cannot do this, stop. |
| **Added value over #47** | With identical poses/depth, the extra policies reduce p95 temporal edge/residual-change diagnostics by ≥20% or decoded-RGB-to-first-visible p95 by ≥20 ms, while retaining its image quality/coverage within the above regression limits. No visible source-switch shimmer. If it merely matches #47 at higher complexity, stop this separate rewrite. |
| **Mono stereo, following 3–5 days** | Full mono retains ≥2 dB mean gain on each sequence and ≥20% LPIPS improvement in the fixed textured regions. On near-field eyes/offsets, halve main's p95 disparity and foreground-edge errors, with no reduction in SGBM matched fraction; never count unmatched pixels as correct. Inspect both eyes. If gains require oracle poses/depth, ToF, a smaller IPD or hidden targets, stop. |
| **Causality and continuity** | Zero future observations/publications. No all-black missed-publication frame in previously supported space, no growing static ghosts and no source-switch burst worse than main in the three clips. Fallback occupies ≤1% of frames on the ordinary, already-observed reference route. More fallback means failure, even if valid-pixel scores improve. |
| **Shared display and memory** | <0.1% misses of 11.11 ms and no consecutive misses in reference runs at runtime resolution; <8 GB total residency throughout, including warmup/recovery. Reject a model configuration whose individual launches repeatedly break this gate. |
| **Fresh events and transfer** | Native 30 FPS event depiction ≤150 ms p95, and fresh RGB visibility no worse than main by >10 ms. No contradicted moving trace older than 0.5 s. Repeat on a physical recording; guessed exposure times cannot certify this. |
| **Pilot acceptance** | Owner prefers blinded sharpness/continuity and finds no added discomfort at actual IPD/runtime. A better mean score cannot override a worse headset experience. |

SGBM is only a diagnostic. Its 128-disparity search can miss true near-field disparities (the arithmetic above already exceeds it at 25 cm). Keep the existing metric for comparability, explicitly flag out-of-range regions, and additionally check geometric reprojection/edge errors there. Similarly, observation coverage must be separate from any opaque UI or unknown-background fill; otherwise this representation could falsely score 100% coverage.

## Runner-ups and effort

**Runner-up 1: a rotational panorama with remembered video and late timewarp.** This is the cheapest path to sharp head-turn response, supported by the telepresence work above. It loses because one panorama cannot provide correct stereo depth or translational parallax around the near floor, pot and wall. It is useful as an explicitly degraded viewing mode, not the requested primary view.

**Runner-up 2: a Quark-style learned layered novel-view renderer.** It offers a stronger learned answer to occlusions and source blending than my conventional compositor. It loses on proven deployment fit: A100/30 FPS, multi-view dynamic evidence, and no verified public implementation/checkpoint licence or <8 GB consumer configuration from the project page. NeuralPassthrough's released MIT code is another reference, but its stereo dependency and missing real-time C++ path likewise prevent direct adoption. Developing/training a mono replacement would turn Ito into a model-research project.

SHARP was also checked: [code](https://github.com/apple/ml-sharp/blob/main/LICENSE) uses Apple's custom software licence; [weights](https://github.com/apple/ml-sharp/blob/main/LICENSE_MODEL) use the Apple ML Research Model licence, whose definition of research excludes product development. That is narrower than merely non-commercial use. Independently, feeding single-image Gaussian predictions into a persistent mapper substantially overlaps the existing initializer plan; it does not resolve this report's source-selection/stereo question.

Estimate **1–2 engineer-weeks for the decisive offline comparisons**, staged so the oracle failure can stop work in days. If they pass: **4–7 additional weeks** for the bounded cache/compositor, CPU estimator integration, streaming depth publication, recovery and cross-platform validation, plus **1–2 weeks** for physical-camera timing/calibration and headset acceptance: roughly **6–11 weeks total**, with timing and near-field geometry the largest uncertainties. This assumes pretrained weights work without new training. If meaningful gains require training a new model, stop and reassess scope rather than hiding that cost in an integration estimate.

The decision is to test this one view-centric architecture before adopting a wholesale Gaussian reconstruction rewrite. Its success criterion is a better pair of live eye images under the real budget; a convincing reconstructed room viewed offline is insufficient.
