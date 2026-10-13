# Measurement baseline

The product baseline is main `316a556`. The committed
[fingerprint](../e2e/measure_reference.json) pins its `ito` and `drivers` trees and
the manifest of all 3,925 reference files, including every decoded image, both
trajectories, truth logs, camera descriptions and the saved static room. Product
code and the wire protocol are unchanged. All added recording fields are offline
fields; no wire-version bump is needed.

From this worktree in WSL, reproduce the complete baseline with:

```sh
bash e2e/measure_baseline.sh ../m1-out/verified
```

This verifies the frozen inputs, makes exact-ID headless captures, runs three
SLAM repetitions of each sequence, renders the sheets and clips, times 1,000
stereo GL frames per sequence/repetition, and audits every prefix and pose
deadline. CUDA runs hold `ceres.lock`, have a 1,100-second timeout, and reject
an 8 GiB Torch reservation. Image evaluation uses Mesa software rendering.
After an interruption, prefix the same command with `ITO_MEASURE_RESUME=1` to
reuse completed captures whose script/schedule hashes, frame counts, snapshots
and causal traces still validate. Source capture, images, clips, display timing
and the final audit run again.

Nothing drives the desktop, starts the owner's simulator, or installs into its
environment. The existing Windows SLAM environment and models are read only.

The machine-local prerequisites are `work/rf-run`, `work/gs`,
`work/truth/data/scan/scene.mjb`, `venv-slam/Scripts/python.exe`, `work/pp/models`,
and the isolated `work/m1-out/venv`. To create the latter:

```sh
uv venv ../m1-out/venv --python 3.12
uv pip install --python ../m1-out/venv/bin/python -e . pillow scipy scikit-image \
  imageio imageio-ffmpeg opencv-python-headless 'mujoco==3.3.7' ruff
uv pip install --python ../m1-out/venv/bin/python torch torchvision \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python ../m1-out/venv/bin/python lpips
```

Each run saves `packages.txt`. LPIPS weights and CUDA caches stay in `m1-out`.
Override `ITO_MEASURE_SLAM_PYTHON`, `ITO_MEASURE_MODELS`, or `ITO_MEASURE_IPD`
when needed. The baseline uses **64 mm configured IPD**, not a headset measurement.

## What is measured

- **Identity/calibration:** `measure_source.py` rerenders five consecutive camera
  poses around each of the seven moments, encodes H.264, and joins decoded PTS to
  independent source IDs. It never selects a truth frame by image similarity.
  Source rows preserve pinhole calibration, instantaneous simulated exposure,
  global readout, monotonic exposure/arrival/publication/GL-completion times,
  and coordinate/calibration revision 0. A known marker rig checks measured
  centroids in MuJoCo and production GL at both eyes and ±3 cm translations.
  Feature displacement between source and decoded RGB measures the codec's
  geometric floor separately from image quality.
- **Causality:** the frozen schedules admit every sixth recorded frame, about
  5 Hz. Every candidate receives the same admissions; measured processing time
  determines when its result becomes available. A prefix contains only completed
  work at its deadline. This is a controlled queued workload, not the live
  worker's newest-frame dropping policy. Disk writes and diagnostic rendering
  are outside its virtual clock. Models initialize before that clock; first
  inference is charged. The reference repetitions use populated compilation
  caches; the initial cold run is retained separately in `work/m1-out/baseline`.
  The legacy recordings lack arrival times, so
  their frame and motion source-local stamps stand in for availability. This
  assumption is recorded and cannot establish original live causality.
- **Poses:** `poses.jsonl` keeps the latest published backend pose available at
  each deadline separately from the pose refined by the first subsequent
  integration. Missing estimates remain missing. This does not simulate the
  pilot's `AnchorGlide`. Fixed first-frame alignment is shared by all truth
  comparisons; there is no evaluation-image pose fit.
- **Images:** full-frame PSNR, RGB SSIM and AlexNet LPIPS include black holes.
  Coverage means accumulated alpha >0.5. The common visibility mask is derived
  from true depth and the frozen input schedule, independently of the candidate.
  Reprojected depths must agree within 1 cm +0.5% of range. Previously seen
  holes and never-seen pixels are reported separately. Legacy truth is
  thumbnail-matched; its centred rerender differs by one pixel in principal
  point from the original camera. Both comparison panels use the same centred
  projection. The exact-ID capture has its own correctly described calibration.
- **Stereo/edges:** both eye images share the production midpoint sort. SGBM
  disparity error is compared with true depth, with its matched fraction shown
  explicitly; textureless/unmatched pixels are not successes. Foreground edge
  distances sample depth boundaries nearer than 2 m. Temporal clips report
  changes in candidate-minus-truth residuals and edge masks; these are
  diagnostics, not a perceptual comfort score.
- **Timing:** `display.json` reports p50/p95/p99/p99.9, 11.11 ms misses and their
  longest run for 1,000 fenced stereo frames, after ten warmup frames. It records
  occupied splats, sort capacity and upload cost. These are compacted snapshots
  at 360×640 per eye, with no simultaneous SLAM, compositor or scanout. CPU
  sheet-render timing is separately labelled and is not the hardware result.
- **Freshness:** the exact-source RGB-D oracle harness uses an ID-colour pass
  through the same alpha compositor to verify first visible contribution.
  Stable-sharp requires PSNR ≥30 and SSIM ≥0.9 for three consecutive draws;
  failure within that observation's three-draw horizon is censored. An injected
  ±3 cm head translation to GL completion is measured separately. Exposure-to-visible includes offline
  recording/decoding delay and must not be cited as live latency. The old mono
  baseline's physical freshness clocks remain null because exposure and scanout
  were never recorded. Surface-update time and contributing observation index
  are separate arrays in each snapshot. An age-colour compositing pass records
  the contribution-weighted visible observation age per pixel, using the legacy
  capture clock where exposure is unknown. Clip margins label the *newest map*
  observation age explicitly; they do not claim every visible surface is that
  fresh. Visual factors/s, published poses/s,
  and geometry updates/s are separate counters.

## Baseline evidence

The exact-source run matches **35/35 IDs**. H.264 full-frame PSNR has a
40.22 dB median. Decoded feature displacement is **0.037 px median / 0.124 px
p95**; its p99.9 is 13.73 px, so the optical-flow diagnostic does have outliers.
Marker reprojection is **0.106 / 0.283 px** (median/p95) in MuJoCo and
**0.008 / 0.030 px** in production GL, below the 0.5 / 1 px gate.

All 35 stable-sharp measurements are censored. The software-rendered RGB-D
harness measures 4.80 s median exposure-to-first-visible, including the offline
batch capture, and 20.62 ms median injected-head-motion-to-GL completion.
Neither is a live Microduck or motion-to-photon latency claim.

The final audit checks **51 prefix snapshots**, **11,736 pose deadlines** and
**1,098 clip frames**, with no future publications or observations found.
All three runs use the same admissions. The six reference camera scores agree
to the precision shown; the hard event is kept separate.

| Moment | PSNR dB | SSIM | LPIPS | Coverage | Previously seen holes |
| --- | ---: | ---: | ---: | ---: | ---: |
| sweep180 215 | 20.00 | 0.790 | 0.465 | 100.00% | 0.00% |
| sweep180 453 | 17.25 | 0.717 | 0.535 | 96.45% | 1.61% |
| sweep180 526 (hard) | 17.01 | 0.777 | 0.545 | 92.45% | 0.02% |
| sweep180 693 | 14.80 | 0.703 | 0.558 | 89.09% | 0.01% |
| tour 363 | 20.83 | 0.804 | 0.436 | 100.00% | 0.00% |
| tour 1501 | 22.33 | 0.779 | 0.479 | 100.00% | 0.00% |
| tour 2284 | 15.52 | 0.790 | 0.603 | 100.00% | 0.00% |

The hard moment contains frame 523 in all final repetitions, published only
10–15 ms before the deadline. An earlier missed-publication result is retained
below; this event is sensitive to scheduling.

RTX 4070 Ti, **360×640 per eye**, isolated production GL, 1,000 stereo frames
per row. These exclude concurrent reconstruction and headset presentation.

| Sequence / repeat | p50 ms | p95 ms | p99 ms | p99.9 ms | Misses >11.11 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| sweep180 / 1 | 3.38 | 3.64 | 3.81 | 4.85 | 0.00% |
| tour / 1 | 3.72 | 4.09 | 4.32 | 5.19 | 0.00% |
| sweep180 / 2 | 3.38 | 3.60 | 3.79 | 4.91 | 0.00% |
| tour / 2 | 3.67 | 3.99 | 4.13 | 5.18 | 0.00% |
| sweep180 / 3 | 3.38 | 3.58 | 3.73 | 4.84 | 0.00% |
| tour / 3 | 3.68 | 4.01 | 4.21 | 4.97 | 0.00% |

Sweep uses 178,182 occupied splats (262,144 sort capacity); tour uses 477,967
(524,288 capacity). One-time uploads take 9.7–10.0 ms for sweep and 46.1–48.6 ms for tour.

| Sequence | Visual factors/s | Propagated poses/s | Surface updates/s |
| --- | ---: | ---: | ---: |
| sweep180 | 3.905–3.906 | 5.017 | 4.141–4.142 |
| tour | 2.924–2.935 | 4.979–4.982 | 3.488–3.537 |

Peak Torch reservation is 3.80 GiB for sweep and 3.82 GiB for tour; this is
allocator evidence, not a measurement of total process GPU residency.
In repetition 1, contribution-weighted visible observation age has p95 values
of 10.02 s at sweep 453, 10.26 s at sweep 693 and 23.90 s at tour 2284.
That is why newest-map age cannot stand in for visible-surface freshness.
Stereo disparity, foreground-edge diagnostics, pose errors and per-pixel age
percentiles for every eye/translation are retained in the results JSON.


The committed [results](measure-results.json) preserve all three repetitions,
source checks, audit counts and evaluator hashes. Start with the
[sweep sheet](measure-img/sweep-reference.jpg), [tour sheet](measure-img/tour-reference.jpg),
[near-wall stereo](measure-img/tour-stereo.jpg) and [hard moment](measure-img/hard-526.jpg).

The full machine-readable evidence is `work/m1-out/verified`: `audit.json`, per-repeat
`metrics.json`, `display.json`, publication/pose traces, PNG pairs, masks, JPEG
sheets and MP4s. `source/summary.json` is the exact-ID/calibration check.

The clips for repetition 1 are [frozen map/moving head](../../m1-out/verified/repeat-1/frozen-map-moving-head.mp4),
[fixed head/map updates](../../m1-out/verified/repeat-1/fixed-head-map-updates.mp4),
and [both](../../m1-out/verified/repeat-1/both.mp4). Also inspect the
[observed-bookshelf moving-head clip](../../m1-out/verified/repeat-1/bookshelf-frozen-map-moving-head.mp4);
its matching map-update and combined clips are in the same directory.
These bulky outputs stay outside Git.

The six reference moments are sweep frames 215/453/693 and tour
363/1501/2284. Sweep frame **526** was selected as the hard moment before
scoring. Every moment has a camera pair, two eye pairs, and ±3 cm pairs.
Unlabelled 1:1 PNG pairs precede the diagnostic JPEG margins. The three
synchronized two-second clips isolate frozen map/moving head, fixed head/map
updates, and both; each also has 3× slow playback. The hard view starts unknown,
so a second triplet looks at the already observed bookshelf to make frozen-map
head-motion artifacts inspectable.

The inspected images lose book labels and floor grain, blur the pot and ball,
and introduce scalloped wall/floor edges. Sweep has missing upper and lower
borders. Tour's near-wall left eye pulls the bookshelf far into the view while
the right eye exposes a different wall error. At the hard moment, map publication
can change the entire view from black to a blurry slatted foreground object. A separate earlier warm run missed that publication: its
[all-black hard sheet](measure-img/hard-526-missed-publication.jpg) scores 5.92 dB
and has 100% holes in previously seen space. This supplementary failure is
identified separately in the results JSON; it is not one of the final three
repetitions. This is an event failure; averaging it with an easy wall view would conceal it. Filmstrip
inspection establishes visible changes, not headset smoothness.

Remaining limits: old recordings cannot acquire exact source identity or physical
exposure timing retrospectively; the stock Microduck mediad path still exposes
neither. The new source check covers the direct headless H.264 path, not mediad.
Dynamic-object and sensor-dropout acceptance, simultaneous tracking/rendering,
full headset resolution, actual IPD and motion-to-photon remain unestablished.
These are not inferred from static offscreen results.

## Acceptance and verification

| Gate | Result and evidence |
| --- | --- |
| Exact new sim frame IDs | PASS for the direct headless H.264 recording: 35/35; not verified through stock mediad/WebRTC. |
| No future observations | PASS for the controlled replay schedules and clip deadlines; see the frozen audit. Original legacy arrival times remain an explicit assumption. |
| Static calibration ≤0.5 px median / ≤1 px p95 | PASS in MuJoCo and production GL, at the centre, both eyes and ±3 cm. |
| Truth/input geometric agreement without timestamp selection | PASS for the new exact-ID capture. FAIL to establish this retrospectively for the old recordings, which lack IDs and retain a 1 px principal-point mismatch. |
| Both-eye projection | PASS for the measured marker rig at configured 64 mm IPD. Actual-headset IPD/scanout acceptance is unverified. |
| Reproducible baseline numbers, sheets and clips | PASS for the pinned main product trees and frozen reference inputs using the command above. |
| Complete live recording/display instrumentation | FAIL: source identity/exposure in stock mediad, live observation visibility and physical display timing remain unestablished. |

Existing end-to-end checks passed: frame joining (3 μs/frame on the final rerun),
production renderer sorting/compositing/posed views/SH 0–3 under EGL, and real
WebRTC driver/pilot exchange (130 video frames, 213 audio frames, 264 rejected
invalid requests; neutral within 240.6 ms after the harness killed its own pilot).
Ruff and whitespace checks pass. This evidence supports the offline measurement
path; it does not close the live/headset portions of issue #38.
