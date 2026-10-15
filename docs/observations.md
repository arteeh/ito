# Observations (wire 4)

Every camera is named. `Motion.camera` and `SensorObservations.camera` identify the camera they
belong to; `RobotDescription.motion_camera` names the optional motion prior. Selecting another
camera never silently borrows that prior. Camera-only input has no motion or sensor requirement.
Gait labels and contact/floor assumptions belong to the driver, not reconstruction.

All timestamps are source estimates in driver monotonic seconds. The link maps them to the pilot
clock once. `Camera.capture_delay` distinguishes receive-stamped video from exposure timing;
`FrameMetadata.exposure` and `time_sigma` retain known integration time and timing uncertainty.
Microduck WebRTC currently exposes receive time: neither physical exposure/readout nor remote
robot-to-driver clock accuracy has been calibrated. A camera's immutable named `calibration`
references its rectified pinhole intrinsics; changing it requires a new description/connection.
Pixels are `(u right, v down)` with the supplied principal point and focal lengths.

Poses are parent-from-child, metres and unit xyzw quaternions. Camera and world axes are right,
up, back (+X,+Y,+Z); the optical axis is -Z. World origins are arbitrary startup anchors. A world
Y coordinate does not identify a floor. Camera extrinsics map camera to robot; per-frame poses
and optional motion samples map camera to world. The motion noise model and per-sample
`uncertainty_scale` are driver estimates, not truth or guaranteed bounds.

IMU mounts map right-handed sensor axes into the named camera. Angular velocity is rad/s;
specific force is m/s² (a supported sensor reads approximately +g, free fall zero). Optional
orientation maps the reported sensor axes into a +Y-up fusion world with arbitrary yaw. It is
not an odometry heading. Temperature is °C. Sample noise and clock uncertainty travel with it.
Beta robotd already converts chip axes to head axes; the driver uses the corresponding mount.

A range sample retains its source time, sensor-to-camera mount, uncertainty, and each zone's
unit ray, full angular width/height in radians, slant distance in metres and quality. Only valid
zones carry distances. No-return is not a surface and invalid is not free space. Sixty-four zones
never become invented dense depth pixels. The estimator registers their footprints, rejects
mixed-depth patches and stale observations, and uses robust agreement to constrain metric scale.

Missing sensors mean absent tuples/streams, never zero-valued measurements. Raw inertial samples
provide orientation/turn prediction by integrating gyro with near-rest gravity correction; the
estimator does not trust chip-fusion startup tilt. Gaps hold orientation and resume at the next
sample. This does not imply observed translation or full visual-inertial odometry. Camera-only tracking retains an unobserved metric gauge. The live wire rejects v3;
`e2e/observation_recording.py` explicitly migrates archived v3 descriptions and gait uncertainty.

The beta simulation defaults to GC2093 1920×1080/30, an LSM6DSV16X game-rotation approximation
at 60 Hz delivered in 15 Hz bursts, and VL53L8CX 8×8/15 Hz. Its only camera FOV setting is
`--beta-camera-hfov` (`drivers/microduck/lens.py`); this is a simulation assumption until upstream
publishes lens intrinsics. The beta IMU translation is also nominal: camera-coincident because
the available CAD exports only the zero3 HAT position. Its axis mapping is the real reader's
measured mapping. Neither assumption is a physical beta calibration. `--board zero3` selects
the secondary IMX219/BMI088 path. The forks' ordinary body-server defaults remain zero3.

Validation (15 October 2026): [results and source hashes](observations-results.json). One fresh
beta tour supplies all six causal cohorts; live video is 1920×1080, evaluation pixels are
subsampled to 640×360 with matching intrinsics. Full/Input 100%/Max 1M; inherited `a7b8ea0`
mapper, 98,304 render cap for Gaussian arms, uncapped point controls. Each run uses its own
available camera pose. Admissions differ with actual model work. Source truth only evaluates
results. Baseline schema migration and compressed storage do not change its estimator.

| Matched-view sheets (both rows, columns 2–3) | By-eye finding |
|---|---|
| [IMU, 76 s](observations-img/tour-imu-76.jpg), [late turn](observations-img/tour-imu-84.jpg) | IMU avoids the camera-only half-turn failure, but late drift, mottled walls and temporary missing regions remain. No general quality pass. |
| [ToF, 50 s](observations-img/tour-tof-50.jpg), [76 s](observations-img/tour-tof-76.jpg), [late turn](observations-img/tour-tof-84.jpg) | Floor remains blurred; box/wall geometry remains broken. Better metric scale does not establish better appearance. |
| [Contract control, 12 s](observations-img/tour-baseline-12.jpg), [76 s](observations-img/tour-baseline-76.jpg) | Broadly comparable sofa/box blur and wall ghosting to the base; no sharp-edge pass. |
| [Beta default, 12 s](observations-img/tour-default-12.jpg), [76 s](observations-img/tour-default-76.jpg) | Similar broad appearance to the base. Box-side mottling and ghost surfaces remain; no overall visual acceptance claimed. |

Camera-only/+IMU/+ToF rotation p95 is 176.75/18.50/19.10°; median depth error is
.413/.349/.147 m. These locate inspection, not quality gates. Default (motion+IMU+ToF) SIFT
reprojection p95 is **15.63 px**, versus baseline **10.35 px**, on the evaluation grid (multiply
by three for source pixels). Confidence-supported values are 15.66/7.74 px. **The 2 px scene
target is not met.** Full-resolution lens markers independently pass at 1.4 px.

Fresh sensor checks: 60 Hz, near-rest specific force 9.810 m/s², 64 ToF zones; median axial
range-centre error 1.1 cm. Eleven gyro samples reach the configured 500°/s limit; inertial
orientation drift is substantial in difficult tour segments. This is not full VIO or calibrated
SFLP equivalence. Headset, real beta, physical mounts/intrinsics and sustained 30 Hz/VR timing
remain untested. Earlier functional e2e and fork checks passed before the disk incident; their
logs were deleted. The fresh compact evidence does not establish all #42 acceptance conditions.

Second by-eye review (same sheets, 15 October 2026), no number gates:

- Camera-only vs +IMU (12/50/76 s, 84 s turn): equal at 12 and 50 s. At 76 s, +IMU is worse in one
  row (large black hole, ghost box edge). At the 84 s turn camera-only is a smeared mess and +IMU a
  flat wall without the window: both wrong. IMU prevents the rotation blow-up but does not yet look better.
- +IMU vs +IMU+ToF: the 50 s floor is smoother but loses plank detail; 76 s is alike, holes in both;
  at the 84 s turn ToF recovers a wall corner but with a slanted wall and a spurious edge. Mixed.
- Beta default vs `a7b8ea0` (point renders): about the same at 12 and 50 s; at 76 s the default has
  more speckle on the box and wall floaters. Slightly worse, not clearly. The sensor-free wire-4
  control is clearly worse at 76 s (ghost wall slab, duplicated shelf).
- All Gaussian arms at 12 s show the same framing/roll offset from the source view, so it is not
  sensor-specific.

Windows release of `15ad3f1` (`release.ps1 -Branch feat/obs-42`, 7.76 GB): its runtime imports
torch (CUDA available), cupy, mujoco, aiortc, xr, imgui_bundle and ito. The GUI was not
launched because ItoVirtual was not running.

`oracle_capture.py --record-stride 3 --no-live-record` keeps the live beta camera intact;
`observations_suite.sh` checks C: and the total job budget before/after each replay, makes
reports and deletes each arm's depth stacks before starting the next. Contact sheets and small
manifests remain; captured RGB-D and individual video frames are removed after comparison.
