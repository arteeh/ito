# Robust head-camera tracking for Ito

Research note, 11 October 2026, step 3 of the agreed plan (benchmark, offline splatting, **choose
tracking**, online). Question: how should Ito track the Microduck head camera robustly in real time,
so that the pilot's newest view lands in the right place even during fast head turns, motion blur
and plain walls?

Evidence tags, as in PR #15:

- **[V] verified here**: read in Ito's code at `cabaa4d`, in `origin/fix/imu-slam` (`e957e4a`), in
  Pollen's `microduck` source at the revision `drivers/microduck/setup-sim.sh` pins (`77005fd`), in
  a cloned upstream repository at the commit named, or on PyPI; or computed by the CPU check in the
  appendix.
- **[P] paper, README or owner measurement**: a claim from authors, or a number from the owner's
  runs on Ceres (issue #33), with a link.
- **[E] estimate**: my own arithmetic or judgement; to be confirmed on Ceres.

No NVIDIA GPU and no access to Ceres: nothing ran on CUDA, and nothing here is measured on the real
robot. What I could not verify is listed at the end.

## TL;DR

**Stop asking vision to find the camera. The robot already knows where its head points; make that
the tracker, and let vision correct it.**

1. **The robot's own pose is already better than MASt3R's.** The IMU-prior commit measured the
   head-FK + trunk-IMU camera pose against MuJoCo truth over 800 poses: **p95 1.32 degrees
   rotation error, 1.77 cm/m translation drift** [V, `e957e4a`, `ito/reconstruction/prior.py`].
   MASt3R-SLAM's heading on the same kind of drive: **p95 5.21 degrees**, with 20 map restarts in
   109 s [P, issue #33]. Vision is the weaker sensor for rotation, and rotation is most of what a
   Microduck head does.
2. **The IMU gate failed because it vetoed and overwrote instead of fusing,** on a prior whose
   timestamps are guessed. It replaced MASt3R's rotation after MASt3R had solved rotation,
   translation and scale together, rejected whole frames and whole local solves at a 6 degree gate,
   and counted every rejection as a loss. That is why heading improved while restarts, path error
   and PSNR got worse (section 2).
3. **Timestamps are the cheapest win, and they are broken today.** Every robot sample and every
   frame is stamped on *arrival* in the driver, robot state comes at 20 Hz, and a constant 90 ms is
   subtracted for the camera. My CPU check (appendix) puts the resulting rotation prior error at
   **p95 1.0 to 4.5 degrees, max 5 to 21 degrees** during head turns, depending on jitter. The same
   motion from source-stamped 50 to 100 Hz samples: **p95 0.07 to 0.18 degrees**. robotd already
   publishes `t_ns` on the robot's `CLOCK_MONOTONIC`, which *is* the driver's `perf_counter` on the
   robot; Ito ignores it [V].
4. **Recommendation, ranked first: robot-first tracking.** The driver streams a source-timestamped
   camera pose at the robot's 50 Hz loop rate (head FK on trunk IMU, leg odometry, plus the head IMU
   on real hardware). On the pilot PC a small sliding-window pose graph (CPU, scipy, a few ms) fuses
   it with visual relative poses. The world frame is the robot's odometry frame, so **a map restart
   no longer exists**: when vision fails, its factor is simply missing. The visual front end starts
   as the existing MASt3R tracker with the robot prior added *inside* its Gauss-Newton solve (about
   20 lines), and moves later to a licence-clean front end (DPVO or DROID's frontend, PR #15) behind
   the same interface.
5. **Not recommended:** MASt3R-Fusion as a dependency (no licence file, so all rights reserved; CC
   BY-NC-SA weights; Linux-only GTSAM fork; needs raw high-rate IMU with hardware timestamps), the
   classical VIO stack (sparse features die on plain walls and blur, Linux/ROS, two of four GPL), and
   VGGT-SLAM or feed-forward GS-SLAM for online tracking (too heavy or not mature). MASt3R-Fusion is
   the right *design* to borrow from; VGGT-SLAM fits the offline step of the plan.

## 1. What exists

### 1.1 `ito/reconstruction/slam.py` (MASt3R-SLAM wrapper) [V]

- A **4-keyframe** local graph (`WINDOW = 4`) on MASt3R-SLAM's `FrameTracker`, single-threaded,
  calibrated, `filtering_mode = "recent"`, 5 local-opt iterations. Frames resize to 512 wide.
- The robot pose is used only around the edges: it aligns the first map to gravity and the startup
  heading (`self.world = prior @ inv(pose)`), seeds the solve *only after* a loss (`unmatched_since`)
  or on a recall retry, and checks the solved heading against the robot's (`predicted`).
- Failure handling is a stack of heuristics: a stray beyond `HEADING_TOLERANCE` (12 degrees) turns
  the world a third per frame (`steer`); beyond `HEADING_LOST` (60 degrees) or after `LOSS_SECONDS`
  (1 s) unmatched, a **new local map** starts at the robot's heading. Sector keyframes every 30
  degrees survive restarts for recall (`relocalize`). Restarts are counted by cause.
- Each restart leaves the old map's splats in place while a new world is placed beside them; with
  scale free per map (Sim3), that is the doubled and tilted walls (`carve` retires only what the new
  view sees through).
- What this cannot fix by tuning: MASt3R-SLAM's tracker declares a frame lost when its match
  fraction is low (`tracker.py` line 67 to 70 upstream [V, `e6f4e3d`]). Blank walls, blur and fast
  pans all lower the match fraction, so the frames where tracking matters most are the ones it
  drops. With a strong prior, a low-match frame is still informative about the directions it
  does constrain, but the upstream API throws it away.

Measured on Ceres (issue #33, replay `ITO_SLAM_IMU=0`): slam_hz 6.5, tracked 4.5 Hz, 220 of 908
frames lost, 20 restarts in 109 s, frame p95 406 ms, heading p95 5.21 degrees [P].

### 1.2 The IMU gate, `origin/fix/imu-slam` (`ed00b55`, `e957e4a`) and why it failed

What it did [V, diff of `ed00b55`]:

1. Seeded every measured frame at `predict(prior, previous, yaw_bias)`: robot rotation, previous
   visual translation.
2. After MASt3R tracked, compared the **full SO(3)** rotation with the prior. Over 6 degrees: undo
   the reference pointmap update, set `lost = True`, call `relocalize`, return (frame dropped).
3. Accepted frames: **rotation overwritten** by the prior plus a yaw bias that may drift at most
   0.5 degrees/s within plus or minus 10 degrees; translation and scale kept from vision.
4. After every local graph solve: if any keyframe's rotation moved over 6 degrees, **roll back the
   whole solve**; otherwise keep the solved translations but **put the old rotations back**.

Results on Ceres [P, issue #33 comment]: heading p95 5.21 to 1.47 degrees, but restarts 20 to 27,
lost frames 220 to 329, prediction PSNR 22.75 to 19.02 dB; in the synthetic room camera-centre RMSE
0.27 to 0.60 m, Sim3 scale 0.93 to 0.70, ghosts 77 to 90 %.

Why, from the code [E, each point traced to the diff]:

- **Overwriting rotation breaks the solve it came from.** MASt3R's tracker minimises ray and
  distance residuals over a joint Sim3 (rotation, translation, scale). Swapping the rotation after
  the fact leaves a translation and scale that were optimal only for the discarded rotation, and
  places the frame's pointmap with a pose no residual agreed with. A 3 degree swap moves a surface
  2 m away by about 10 cm: exactly a doubled wall. Step 4 does the same to every keyframe in the
  window, so the next frame tracks against references whose poses contradict their own matches.
  That is the doubled path error and the falling Sim3 scale.
- **The gate trusts the worse sensor at the worst moment.** The prior it trusted is stamped on
  arrival at 20 Hz with a constant 90 ms offset. The appendix shows that prior is several degrees
  off, up to 12 to 21 degrees, *during brisk turns*, which is exactly when MASt3R's rotation is
  most likely correct relative to the previous frame and the gate fires. 63 rejections, each
  turned into a loss and a `relocalize` that shrinks the window to one recalled keyframe
  (`self.frames[:] = [best]`): fewer references, more unmatched frames, more restarts.
- **Hard gates are not fusion.** A 6 degree gate on all three axes is a binary veto with no notion
  of either sensor's uncertainty. Gravity (pitch, roll) from the IMU is excellent; yaw from the IMU
  drifts; translation from vision is up to scale; translation from leg odometry is metric but slips.
  The right tool weighs each direction by its own noise: a least-squares problem, not an if.
- **Scale was left free.** Ito keeps MASt3R's Sim3 scale free per map, and nothing metric
  (odometry translation, ToF) anchors it, so drift in scale reads as drift in path.

The lesson is not "the IMU does not help". It is: **fix the timestamps, then fuse inside the
optimiser with real covariances, then remove the restart mechanism that turns every disagreement
into a new world.**

### 1.3 PR #15 (DROID-SLAM research)

Recommends a DROID-SLAM frontend with robot-prior initialisation and a metric depth prior, BSD-3,
2 to 3 GB VRAM estimated, Windows build effort similar to MASt3R's kernels. I agree with its analysis
of DROID as a *visual front end*, and its point that DROID has no lost state. This note puts that
front end under a robot-first fusion layer, which PR #15 lists only as an optional later step (a
yaw prior factor) [V, PR #15 text].

### 1.4 What the Microduck actually measures [V]

From Ito's driver (`drivers/microduck/adapter.py`, `frames.py`, `camera.py`) and Pollen's source
(`duck-ipc-proto`, `duck-control`, `tof`, `odometry`, `mediad` at `77005fd`):

| Signal | Source, rate | Timestamp available | Used by Ito today |
|:-|:-|:-|:-|
| Trunk IMU: LSM6DSV16X gyro (plus or minus 500 dps) and on-chip SFLP game-rotation quaternion | robotd control loop, **50 Hz**; raw accel exists on the board but is not read (`duck-control/src/imu.rs`) | `RobotState.t_ns`, robot `CLOCK_MONOTONIC` (proto v24) | quaternion only; stamped on driver receive |
| Head joints (neck pitch, head pitch, yaw, roll) and robotd's FK camera pose in trunk (`frames.camera`) | robot.state, 50 Hz max; **Ito subscribes at 20 Hz** (`robot.subscribe {"hz": 20}`) | `t_ns` | yes, receive-stamped |
| Leg odometry: contact-anchor FK, IMU yaw (no magnetometer) | robot.state `odom`, 50 Hz | `t_ns` | position and yaw, receive-stamped |
| **Head IMU** (rigid with the camera) | zero3 board: BMI088 via `tofd`, 100 Hz default, **off by default** (about 4 % of a core); beta board: LSM6DSV16X via robotd, 60 Hz, on by default. `head_imu.stream`, routed over mediad's WebRTC control channel | `HeadImuFrame.t_ns` (gyro, accel, Madgwick quat); mount pose in `frames.head_imu` | **no** |
| ToF: VL53L5CX or VL53L8CX, 8 by 8 zones | `tofd`, 15 Hz, `tof.stream` over mediad | `TofFrame.t_ns` | no; the simulator runs with an absent ToF socket (`sim.py`) |
| Camera: IMX219 (secondhand ID; Pollen says "camera"), 30 fps, H.264 over WebRTC | mediad | **none on the wire**: "Nothing this transport carries is timestamped at source today" (`docs/design/remote-webrtc.md` section 11, which names SLAM and visual-inertial as the reason to fix it) | receive time minus constant `FRAME_DELAY = 0.09` s |
| Safety flags: fallen, limp, picked_up; gait policy | robot.state | `t_ns` | telemetry only |

Board and sensor identities beyond Pollen's source: the press kit lists "a camera, LiDAR and two
IMUs" ([Pollen press kit](https://pollen-robotics.com/microduck/press-kit/)); CNX Software lists
an RK3566 with 1 GB RAM and "2x IMUs, one in the body and one in the head"
([CNX](https://www.cnx-software.com/2026/08/28/microduck-a-duck-like-biped-robot-designed-for-physical-ai-experimentation-and-fun/)).
The RK3566 is the Radxa Zero 3, the `zero3` board in Pollen's code, so expect the BMI088 head IMU,
**off by default** [E].

Three consequences:

- **Ito's driver runs on the robot** (README: `ito-driver-microduck` "On Microduck") next to
  robotd. Python's `perf_counter` on Linux is `clock_gettime(CLOCK_MONOTONIC)` [V, checked here],
  the same clock as `t_ns`. **Robot samples can carry their true sample time with no clock sync at
  all**, and Ito's link `Clock` already maps driver time to pilot time (`ito/link/peer.py`).
- **The camera is the only signal without a source timestamp.** Until mediad dates frames, Ito
  must estimate the frame delay. A constant is wrong whenever encode or decode load changes; an
  online estimate from gyro against visual rotation is standard (VINS-Mono's `td`, MASt3R-Fusion's
  `imu_dt`) and cheap.
- **The IMX219 is a rolling-shutter sensor.** At 90 degrees/s (p95 of the head-turn profile in the
  appendix) and roughly 20 to 30 ms readout, the top and bottom of a frame differ by 2 to 3 degrees
  [E]. Indoor auto-exposure of 10 ms at 720p (about 20 px per degree for a 62 degree lens) smears 18
  px [E]. No vision method tracks well through that; a gyro does not care. The simulator renders
  neither blur nor rolling shutter, so **sim results flatter vision**.

## 2. Options

Scored for the Microduck on a 12 GB RTX 4070 Ti shared with Ito's renderer, Windows first.

| Rank | Option | Fast head turns | Blur | Plain walls | Translation and scale | Cost beside reconstruction | Windows | Licence | Verdict |
|:-|:-|:-|:-|:-|:-|:-|:-|:-|:-|
| **1** | **Robot-first fusion**: source-stamped kinematic + IMU camera pose, sliding-window pose graph with visual relative-pose factors | excellent (gyro/FK at 50 to 100 Hz) | unaffected; blurred frames just add no factor | unaffected; no factor, no restart | metric from leg odometry, drift corrected by vision; ToF anchors scale on hardware | graph on CPU, about 1 to 5 ms per keyframe [E]; front end as chosen | yes (numpy/scipy, already pinned in the `slam` extra) | Ito's own | **adopt** |
| 2 | Same graph, visual factors from **DPVO** (MIT) or **DROID frontend** (BSD-3) instead of MASt3R | as 1 | as 1 | as 1, and DROID's confidence down-weights blank regions | as 1 | DPVO about 4 GB, DROID-VO 2 to 3 GB [P, PR #15] vs MASt3R ViT-L | CUDA kernels to build, as PR #15 describes | permissive | **adopt as phase B**, licence-clean |
| 3 | **MASt3R-Fusion** ([repo](https://github.com/GREAT-WHU/MASt3R-Fusion) `f4469ea`, [arXiv 2509.20757](https://arxiv.org/abs/2509.20757)): MASt3R pointmaps + Sim3 visual factors in a metric SE(3) GTSAM graph with IMU preintegration | good, if IMU raw and stamped | good via IMU | better than MASt3R-SLAM via IMU | metric (IMU) | ViT-L like today plus GTSAM; "8 GB" for long sequences [P]; no fps stated [V, README] | no: modified GTSAM fork built from source, conda/bash only [V] | **no LICENSE file** in the repo (all rights reserved by default) [V]; bundled MASt3R CC BY-NC-SA [V] | **borrow the design, do not depend** |
| 4 | **Classical VIO**: Basalt (BSD-3, [mirror](https://github.com/VladyslavUsenko/basalt-mirror) `0f3b2b5`), OpenVINS (GPL-3, `6948812`), ORB-SLAM3 (GPL-3, last commit 2022-02), Kimera-VIO (BSD-2, `ce8c59b`, 2025-02) [V] | good with a 200+ Hz hardware-synced IMU | partial | **poor**: sparse corners, nothing on a blank wall | metric | CPU, light | none officially; Linux/ROS | 2 of 4 GPL | **no**: needs an IMU and timing the duck does not expose, and solves rotation, which the robot already knows |
| 5 | **DROID-SLAM / DPV-SLAM full** (with backend, loop closure) | vision only | poor | weak (no lost state, but no information either) | up to scale | DROID full 11 GB+ [P, PR #15] | build needed | permissive | no; frontend only, as option 2 |
| 6 | **VGGT-SLAM 2.0** ([repo](https://github.com/MIT-SPARK/VGGT-SLAM) `35327ac`, BSD-2 code) | submap batches, not per frame | poor | better than MASt3R (feed-forward geometry) | SL(4)/Sim3 submaps | VGGT-1B, too heavy beside rendering [E] | unknown | code BSD-2, weights under Meta's VGGT licence (see PR #15) | **offline only**: good candidate for the plan's step 2 (poses for offline splatting) |
| 7 | **Feed-forward GS-SLAM** (Flash-Mono, ICLR 2026 ([proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5fcacc84d87c97bee5baecb3fb8f6e88-Abstract-Conference.html)); FGS-SLAM, VRW 2026 ([abstract](https://pure.bit.edu.cn/en/publications/fgs-slam-a-calibration-free-slam-framework-using-feed-forward-3d-/))) | vision only | poor | learned prior helps | up to scale | unknown | unknown | no public code found [V, search] | **no for tracking**; revisit for reconstruction |
| 0 | **Robot prior only**, no vision (a rung, not a product) | excellent | unaffected | unaffected | odometry drift, yaw drift uncorrected | none | yes | n/a | test rung: measures how much vision must add |

Why rank 1 beats a learned SLAM for *this* robot [E]: the Microduck's camera motion is mostly head
rotation through a known 4-joint chain on an IMU-measured trunk, plus slow walking (at most 0.4 m/s,
`adapter.py`). Pure rotation is exactly what monocular vision cannot triangulate and what blur and
blank walls destroy; it is also what encoders and gyros measure best. What the robot measures badly
(yaw drift over minutes, slip in leg odometry) is what vision corrects well over seconds. That
division of labour is how every robust VIO works; here the "IMU" is a whole kinematic model.

For other Ito robots the same structure holds: a drone has a flight controller's EKF, a car wheel
odometry, a humanoid FK. "The driver streams its best timestamped camera pose with an uncertainty;
the pilot PC fuses vision into it" is robot-agnostic, which a VIO tied to one IMU model is not.

## 3. Recommendation and integration design

### 3.1 Phase A: robot-first tracking with the MASt3R front end

**Robot side (driver, `drivers/microduck`).** No heavy work on the RK3566; robotd already does FK,
IMU fusion and odometry.

- Subscribe robot.state at the loop rate (`robot.subscribe {}`, 50 Hz) and stamp each sample with
  its `t_ns / 1e9` instead of `clock.now()` on arrival. Same clock as the driver's `perf_counter`
  on the robot and in the simulator, so the existing link offset applies unchanged.
- On hardware, subscribe `head_imu.stream` (enable `[head_imu]` on zero3; about 4 % of a core at
  100 Hz) and rotate its gyro into the camera frame with `frames.head_imu`. It sees servo compliance
  and backlash that FK cannot. The simulator has no head IMU (no `tofd` IMU there), so the sim path
  uses FK at 50 Hz, which the appendix shows is already sufficient once stamped at source.
- Send a new protocol message, batched, on the reliable unordered lane that frame metadata uses:

```python
class MotionSample(BaseModel):
    time: Time                     # driver clock, sample time at the source
    camera_pose: Pose              # camera-to-world, world = robot odometry at startup
    angular_velocity: Vector3      # camera frame, rad/s; gyro where the robot has one
    state: Literal["standing", "walking", "turning", "held", "fallen"]

class Motion(Message):
    type: Literal["motion"] = "motion"
    samples: list[MotionSample] = Field(max_length=16)
```

  and in `RobotDescription` the driver's own noise model, so the fusion is robot-agnostic:
  `motion_noise = (rotation_deg_per_s, translation_fraction, gravity_deg)`; Microduck starts from
  the measured 1.32 degree p95 and 1.77 cm/m [V, `e957e4a`], inflated while `held` or `fallen`.
- Frame time stays receive time for now, minus an estimate. Ask Pollen for per-frame capture time
  (`abs-capture-time` RTP extension or a `t_ns` per frame on the control channel; their own
  `remote-webrtc.md` section 11 wants it). Until then the pilot side estimates the delay.

**Pilot side, reconstruction worker (`ito/reconstruction`).** Tracking separates from fusion of
splats; the ring, budget, carve and renderer do not change.

- `motion.py`: `MotionBuffer`, a time-ordered ring of samples fed through a small shared-memory
  ring from the link thread (same pattern as `submit`, never blocking). `at(t)` interpolates SE(3)
  (slerp plus lerp, as `frames.at` does), `between(t0, t1)` returns the relative pose and its
  covariance from the noise model.
- `fusion.py`: `PoseGraph`, a sliding window of the last 12 keyframes plus the current frame.
  Variables: an SE(3) pose per node, one frame-delay offset `td` (bounded, slowly varying), and,
  only while the front end is up to scale, one log-scale per visual window. Factors:
  - robot relative pose between consecutive nodes, from `MotionBuffer.between(t_i + td, t_j + td)`;
  - gravity (pitch, roll) per node from the IMU, tight;
  - a weak absolute yaw and position prior on the oldest node (the window's anchor);
  - visual relative pose between frame and reference keyframe from the front end, information
    scaled by its match fraction, robust (Cauchy) loss.

  Solved with `scipy.optimize.least_squares` on a sparse Jacobian (scipy 1.15.3 is already in the
  `slam` extra; GTSAM has no Windows wheels on PyPI, 4.3.0 ships macOS and manylinux only [V];
  PyPose, Apache-2.0, pure Python, is the fallback if the hand-written residuals grow [V, PyPI]).
  Per frame: a 6-DoF solve of the current node only; per keyframe: the window.
- `tracking.py`: the front-end interface, narrow so MASt3R, DPVO or DROID swap without touching
  fusion:

```python
class VisualResult(NamedTuple):
    relative: np.ndarray | None    # 4x4 reference-keyframe-from-frame, None if no usable match
    information: np.ndarray        # 6x6 (7x7 with scale), from match fraction and residual
    points: "array"                # camera-frame pointmap on the device, for splats
    confidence: "array"
    keyframe: bool

class Frontend(Protocol):
    def track(self, rgb, time, predicted: np.ndarray) -> VisualResult: ...
    def correct(self, keyframe_poses: dict[int, np.ndarray]) -> None: ...  # fused poses back in
```

- `mast3r_frontend.py`: today's `slam.py` minus everything that decides restarts, steering and
  recall. Two upstream-shaped changes: always seed the solve at the fused prediction, and add the
  prior to the tracker's Gauss-Newton normal equations (`FrameTracker.solve` forms `H = A.T @ A`
  and solves by Cholesky in Python [V, upstream `tracker.py` line 156 to 171]; adding
  `J_p.T @ W_p @ J_p` and the matching gradient is about 20 lines). A low match fraction then
  yields a prior-dominated pose instead of `lost`. Its local graph keeps the CUDA solver; after
  each solve, `correct` writes the fused keyframe poses back so MASt3R's references never contradict
  the fusion (the opposite of the rollback in `ed00b55`).
- `slam.py` becomes the composition: `MotionBuffer`, `Frontend`, `PoseGraph`, then
  `integrate_points` and `carve` with the fused pose, exactly as now. **No `restart`**: the world is
  the robot's odometry frame from the first frame; when vision fails, splats placed from the fused
  pose keep coming (MASt3R's mono pointmap scaled to the window), or pause if the oracle ladder shows
  that hurts; either way the pilot's eye keeps moving on the robot pose instead of dropping to the
  flat feed after `TRACKING_LOST`.
- Blur gate [E]: predicted blur = |angular velocity| x exposure x focal length. Frames over about
  8 px still get a pose (from the graph) but neither add a visual factor nor splats. Exposure is not
  on the wire either; start with 10 ms, ask Pollen for the AE value with the frame time.

**Pilot comfort.** The renderer already takes rotation from the pilot's head and only the eye
position from the robot (`ito/app/pilot.py`). Fusion corrections move the world (yaw drift) or the
eye (translation drift); both are eased: yaw at most about 2 degrees/s and position at most about
5 cm/s while the pilot moves, frozen while still [E, from redirected-walking detection thresholds,
Steinicke et al., "Estimation of Detection Thresholds for Redirected Walking Techniques", IEEE TVCG
2010]. `steer`'s third-per-frame ease is the right idea at the wrong place; it moves to the
renderer side of the boundary.

**Where it runs.**

| Robot (RK3566) | Pilot PC CPU (reconstruction process) | Pilot PC GPU |
|:-|:-|:-|
| robotd FK, IMU fusion, odometry (unchanged); head IMU on; driver stamps with `t_ns`, batches `Motion` | `MotionBuffer`, `PoseGraph`, `td` estimate, blur gate | front end (MASt3R now), splat fusion, renderer |

### 3.2 Phase B: licence-clean front end

Once phase A meets its numbers, implement `Frontend` with the DPVO or DROID frontend per PR #15
(robot-delta initialisation becomes the graph's prediction; its confidence weights become the factor
information). That removes CC BY-NC-SA from tracking. Reconstruction depth can stay MASt3R or move
to a metric depth model; that is the reconstruction decision (issue #34), not this one.

### 3.3 What to borrow from MASt3R-Fusion

Its Sim3 visual constraint in Hessian form inside a metric SE(3) graph is precisely the coupling
phase A approximates with relative-pose factors, and its constant `imu_dt` time offset and
"auto calibration" confirm that the delay must be estimated [V, README, `config/base_euroc.yaml`,
`mast3r_fusion/vio_utils.py`]. If phase A's loose coupling leaves accuracy on the table, port that
idea (not the code) into `PoseGraph`.

## 4. Test plan on the truth benchmark

Every rung runs on the same recorded drives (`e2e/slam_drive.py` recordings replayed by
`e2e/slam_replay.py`), with the truth benchmark (issue #32) scoring the scene. New recordings must
store each robot sample's `t_ns` and the MuJoCo truth camera pose per rendered frame (as
`imu_prior.py` already samples [V]).

**Oracle ladder** (pose source, depth source):

| Rung | Pose | Depth | Answers |
|:-|:-|:-|:-|
| L0 | truth | truth | ceiling of the splat pipeline |
| L1 | truth | MASt3R | ceiling for any tracker with today's depth |
| L2 | robot prior as sent today (receive stamps, 20 Hz) | MASt3R | how good the robot alone already is |
| L3 | robot prior, source stamps, 50 Hz | MASt3R | what timestamps alone buy |
| L4 | phase A fused | MASt3R | the recommendation |
| L5 | today's MASt3R-SLAM (`main`) | MASt3R | baseline |
| L6 | `fix/imu-slam` gate | MASt3R | the failed attempt, as a regression check |

**Drives**, each 60 to 120 s, three repeats: (a) standing, head sweeps at full servo speed across a
plain wall; (b) the existing walking tour with body turns; (c) a cluttered room control; (d) pick up
and set down (odometry invalid); (e) (b) with network jitter injected on the driver link (`tc
netem`, 20 ms) to check that source stamps make the result jitter-proof; (f) (a) with synthetic
motion blur (average 4 sub-frame renders per frame in `sim_body`), because the sim otherwise has
none.

**Metrics**: camera rotation error p50/p95/max (degrees) and translation error after a *fixed*
(not Sim3-fitted) alignment at the first frame, since the world is supposed to be the robot's;
relative pose error per second; fraction of frames with a usable pose (within 2 degrees, 5 cm);
map restarts (must be 0 from L4); pilot-view PSNR, SSIM, LPIPS; ghost fraction; eye correction
rate (degrees/s, cm/s, p99); frame-to-pose latency; SLAM Hz, VRAM, and whether the desktop
renderer holds 90 fps alongside.

**Gates for adopting phase A (L4)**: rotation p95 at most 1 degree; translation drift at most 2 %
of path; 0 restarts; PSNR within 1 dB of L1 and at least 2 dB over L5; ghost fraction within 5
points of L1; eye corrections under 2 degrees/s and 5 cm/s at p99; no loss of 90 fps. If L3 already
meets these, ship timestamps first and keep vision for drift only.

**Expected reading [E]**: L3 near L1 on rotation; L2 visibly worse on (a) and (e); L5 worst on (a)
and (f). If L1 itself is poor, tracking is not the bottleneck and issue #34 (splat sharpness) is.

**Real robot, once it arrives**: repeat (a), (b), (d) with the head IMU on and off; check the
estimated `td` is stable; record ToF to test it as the scale anchor.

## 5. Order of work

1. Driver: 50 Hz robot.state, `t_ns` stamps, `Motion` message, noise model in the description;
   recordings carry it. Rung L3 measurable. (Small; touches `drivers/microduck`, `ito/protocol`,
   `ito/app`.)
2. `motion.py`, `fusion.py`, `tracking.py`, MASt3R front end with in-solve prior, no restarts.
   Rungs L4 and L6. (Medium; replaces most of `slam.py`.)
3. `td` online estimate and blur gate. Drive (f).
4. Head IMU and ToF scale on the real Microduck.
5. Phase B front end (DPVO or DROID) behind `Frontend`.

## 6. What I could not verify

- Nothing ran on a GPU: front-end costs, the 1 to 5 ms graph solve, and VRAM numbers are estimates
  or quotes.
- MASt3R-Fusion's real-time rate and accuracy: the README gives no fps, and arXiv was unreachable
  from this sandbox, so I read the README, config and code structure only.
- Which board the shipping Microduck uses (zero3 vs beta) and therefore which head IMU; the IMX219
  identification and its readout time; real exposure times under rkaiq's auto-exposure.
- That robotd in the simulator fills `t_ns` and the frames the way it does on hardware (the pinned
  proto has the fields; I did not run the sim).
- How much mediad's encode and Ito's decode delay actually jitter; the appendix sweeps assumed
  values. The 5 ms jitter row reproduces the measured sim p95 (1.0 vs 1.32 degrees), which suggests
  the model is in the right range.
- Whether the blank-wall and blur behaviour of DPVO or DROID beats MASt3R's inside this graph:
  that is phase B's experiment.

## Appendix: CPU check, rotation prior error from timestamps alone

A camera yawing like a piloted head: minimum-jerk turns of up to 80 degrees in 0.7 to 1.6 s, with
pauses, peak 201 degrees/s, p95 90 degrees/s; 30 fps exposures. *Today*: robot.state at 20 Hz,
stamped on receipt (5 ms plus exponential jitter), frame stamped on receipt (90 ms plus Gaussian
jitter), prior read at receipt minus 90 ms; error is absolute camera yaw at the exposure. *Source
stamped*: gyro at 50 or 100 Hz with 0.1 degrees/sqrt(s) noise, integrated, read at the exposure with
the stated stamp error; error is the frame-to-frame rotation the tracker is seeded with. Script
below; `python3 -I timing.py`, numpy only, seed 0.

| Prior | Jitter (state / frame) | p95 | max |
|:-|:-|:-|:-|
| today, 20 Hz, receive stamps | 5 ms / 5 ms | 0.99 deg | 5.18 deg |
| today, 20 Hz, receive stamps | 15 ms / 20 ms | 2.21 deg | 12.10 deg |
| today, 20 Hz, receive stamps | 30 ms / 40 ms | 4.50 deg | 20.82 deg |
| source stamps, 100 Hz gyro | exposure plus or minus 2 ms | 0.17 deg | 0.98 deg |
| source stamps, 100 Hz gyro | exposure plus or minus 0.5 ms | 0.07 deg | 0.25 deg |
| source stamps, 50 Hz | exposure plus or minus 2 ms | 0.18 deg | 0.69 deg |

```python
"""Rotation prior error from timestamping alone, for a camera yawing like a piloted head."""
import numpy as np
rng = np.random.default_rng(0)
T, dt = 120.0, 1e-4
t = np.arange(0, T, dt)
# Pilot looks: random smooth turns, minimum-jerk segments of 0.7-1.6 s, +-70 deg, pauses between.
yaw = np.zeros_like(t); cur = 0.0; k = 0
while k < len(t):
    dur = rng.uniform(0.7, 1.6); pause = rng.uniform(0.2, 1.5)
    target = np.clip(cur + rng.uniform(-1.4, 1.4), -1.4, 1.4)
    n = int(dur / dt); s = np.linspace(0, 1, n)
    seg = cur + (target - cur) * (10*s**3 - 15*s**4 + 6*s**5)
    yaw[k:k+n] = seg[:len(yaw[k:k+n])]; k += n
    m = int(pause / dt); yaw[k:k+m] = target; k += m; cur = target
rate = np.gradient(yaw, dt)
truth = lambda q: np.interp(q, t, yaw)
frames = np.arange(1, T - 1, 1/30)  # 30 fps exposures
def report(name, err):
    e = np.degrees(np.abs(err)); print(f"{name:58s} median {np.median(e):5.2f}  p95 {np.percentile(e,95):5.2f}  max {e.max():5.2f} deg")
print(f"peak camera yaw rate {np.degrees(np.abs(rate).max()):.0f} deg/s, p95 {np.degrees(np.percentile(np.abs(rate),95)):.0f}")
for state_jit, frame_jit in ((0.005, 0.005), (0.015, 0.02), (0.03, 0.04)):
    # Today: robot.state at 20 Hz stamped at driver receive (sample + 5 ms + jitter);
    # frame stamped at driver receive (exposure + 90 ms + jitter), prior = history(receive - 0.09).
    st = np.arange(0, T, 1/20); srecv = st + 0.005 + rng.exponential(state_jit, len(st))
    order = np.argsort(srecv); srecv, sval = srecv[order], truth(st)[order]
    frecv = frames + 0.09 + rng.normal(0, frame_jit, len(frames))
    report(f"today: 20 Hz state, receive stamps (jitter {state_jit*1e3:.0f}/{frame_jit*1e3:.0f} ms)", np.interp(frecv - 0.09, srecv, sval) - truth(frames))
# Source stamps on one robot clock: head-IMU gyro at 100 Hz integrated, frame exposure time known to +-2 ms.
for hz, stamp_err in ((100, 0.002), (100, 0.0005), (50, 0.002)):
    gt = np.arange(0, T, 1/hz); gyro = np.interp(gt, t, rate) + rng.normal(0, np.radians(0.1)*np.sqrt(hz), len(gt))
    integ = np.concatenate(([0], np.cumsum(gyro[:-1]) / hz))
    # Gyro only carries rotation between frames; anchor to truth at each frame's predecessor (vision does that).
    q = frames + rng.normal(0, stamp_err, len(frames))
    rel = np.interp(q, gt, integ) - np.interp(frames - 1/30, gt, integ)
    report(f"source-stamped {hz} Hz gyro, exposure stamp +-{stamp_err*1e3:.1f} ms, frame-to-frame", rel - (truth(frames) - truth(frames - 1/30)))
```

## Sources

- Ito: `ito/reconstruction/{slam.py,rgbd.py,__init__.py,mast3r_runtime.py}`,
  `drivers/microduck/{adapter.py,frames.py,camera.py,sim.py}`, `ito/link/peer.py`,
  `ito/protocol/__init__.py` at `cabaa4d`; `origin/fix/imu-slam` `ed00b55`, `e957e4a`.
- Issues and PRs: [#32](https://github.com/arteeh/ito/issues/32),
  [#33](https://github.com/arteeh/ito/issues/33) (Round I2 numbers),
  [#34](https://github.com/arteeh/ito/issues/34), [PR #15](https://github.com/arteeh/ito/pull/15).
- Pollen Microduck source, Apache-2.0: https://github.com/pollen-robotics/microduck at `77005fd`
  (`duck-control/src/imu.rs`, `tof/src/imu.rs`, `tof/src/main.rs`, `duck-ipc-proto/src/lib.rs`,
  `odometry/src/lib.rs`, `mediad/src/route.rs`, `docs/design/remote-webrtc.md` section 11,
  `docs/robot/cheatsheet.md`).
- Microduck hardware: https://pollen-robotics.com/microduck/press-kit/,
  https://www.cnx-software.com/2026/08/28/microduck-a-duck-like-biped-robot-designed-for-physical-ai-experimentation-and-fun/,
  https://rcdrone.top/blogs/articles/microduck-hardware-explained-how-to-build-an-original-diy-rl-biped-robot
  (secondhand part numbers).
- MASt3R-SLAM: https://github.com/rmurai0610/MASt3R-SLAM `e6f4e3d` (CC BY-NC-SA 4.0,
  `mast3r_slam/tracker.py`).
- MASt3R-Fusion: https://github.com/GREAT-WHU/MASt3R-Fusion `f4469ea`;
  https://arxiv.org/abs/2509.20757 (v3, November 2025).
- OpenVINS https://github.com/rpng/open_vins `6948812` (GPL-3); Basalt
  https://github.com/VladyslavUsenko/basalt-mirror `0f3b2b5` (BSD-3); ORB-SLAM3
  https://github.com/UZ-SLAMLab/ORB_SLAM3 `4452a3c` (GPL-3, 2022-02); Kimera-VIO
  https://github.com/MIT-SPARK/Kimera-VIO `ce8c59b` (BSD-2).
- DPVO https://github.com/princeton-vl/DPVO `0ac95b6` (MIT); DROID-SLAM
  https://github.com/princeton-vl/DROID-SLAM (BSD-3, per PR #15).
- VGGT-SLAM https://github.com/MIT-SPARK/VGGT-SLAM `35327ac` (BSD-2).
- Flash-Mono: https://proceedings.iclr.cc/paper_files/paper/2026/hash/5fcacc84d87c97bee5baecb3fb8f6e88-Abstract-Conference.html;
  FGS-SLAM: https://pure.bit.edu.cn/en/publications/fgs-slam-a-calibration-free-slam-framework-using-feed-forward-3d-/
- PyPI: https://pypi.org/project/gtsam/ (4.3.0, no Windows wheels), https://pypi.org/project/pypose/
  (0.9.5, Apache-2.0, pure Python).
- Steinicke, Bruder, Jerald, Frenz, Lappe, "Estimation of Detection Thresholds for Redirected
  Walking Techniques", IEEE TVCG 16(1), 2010.
