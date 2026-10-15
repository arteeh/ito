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
provide orientation/turn prediction; they do not imply observed translation or full visual-inertial
odometry. Camera-only tracking retains an unobserved metric gauge. The live wire rejects v3;
`e2e/observation_recording.py` explicitly migrates archived v3 descriptions and gait uncertainty.

The beta simulation defaults to GC2093 1920×1080/30, an LSM6DSV16X game-rotation approximation
at 60 Hz delivered in 15 Hz bursts, and VL53L8CX 8×8/15 Hz. Its only camera FOV setting is
`--beta-camera-hfov` (`drivers/microduck/lens.py`); this is a simulation assumption until upstream
publishes lens intrinsics. The beta IMU translation is also nominal: camera-coincident because
the available CAD exports only the zero3 HAT position. Its axis mapping is the real reader's
measured mapping. Neither assumption is a physical beta calibration. `--board zero3` selects
the secondary IMX219/BMI088 path. The forks' ordinary body-server defaults remain zero3.
