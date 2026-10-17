# Ito

Ito is immersive teleoperation software built entirely for piloting robots.

Most teleoperation software treats the pilot experience as secondary. It is usually a basic tool for collecting demonstrations and training robot policies. Ito takes a different approach: it is not designed to train AI. Its sole purpose is to make remotely operating a robot comfortable for the human pilot. We envision a future where people pilot every type of robot from their home or office. This could enable disabled people to act through robots in places their bodies cannot easily take them, and allow people to explore or work in environments that are hostile to humans. Ito is intended to support humanoids, droids, vehicles, mechas, and robot forms that do not fit an existing category. It translates the pilot's tracked pose and controller input into control instructions appropriate to the piloted robot. In the other direction, it translates the robot's sensor input into a comfortable immersive 3D reconstruction of its surroundings.

## Usage

On Windows with an NVIDIA GPU: get the `ito` folder and run `ito.exe` in it. Connect
to a robot by address or a recent robot, or try the simulated robot.

From source (Python 3.12+ and [uv](https://docs.astral.sh/uv/)):

```sh
uv sync
uv run ito                                                # connect screen, as ito.exe
uv run ito --sim                                          # pilot the bundled simulated robot
uv run ito-driver your_robot.adapter:create --port 8080   # on the robot
uv run ito-driver-mujoco --gl osmesa                      # furnished simulation
uv run ito robot-address:8080                             # on the pilot PC
uv run ito robot-address:8080 --mode xr                    # active OpenXR runtime
uv run ito-desktop scene.ply                              # explore a splat scene
```

Drivers listen on all interfaces for LAN or tailnet connections and print their persisted
pairing code at startup. Enter it once on the connect screen: the robot then gives Ito a secret
of its own over the encrypted link and the code is used up. `ito-driver-mujoco --show-code`
shows it again; `--rotate-code` makes a new code and forgets every paired pilot, including
while the driver is running. Use the same `--pairing-file PATH` on each command if you override it.
The bundled simulated robot pairs automatically and has no microphone or speaker.
**Show simulation** opens a separate MuJoCo viewer; closing it leaves piloting connected.

Press **R** to pilot. WASD moves relative to where you look; click or Tab captures mouse-look to aim
its head. The body follows your gaze while moving. Standing still, it turns only far enough
to reach a gaze beyond the head's pan limit.
Differential-drive robots steer into sideways movement. Space stops, **E e-stops**, R resumes, Home recenters, F12 captures a screenshot,
and Escape releases the mouse. Close the window to quit. Gamepad left stick moves, right
stick looks; A resumes, B e-stops, X stops.
Focus loss stops motion. After a lost link Ito reconnects automatically; press R to resume.
The ImGui overlay reports driver state, link RTT, input latency, and capture-to-visible latency.
`--metrics path.jsonl` records these alongside display timing. Ito selects posed RGB-D when the robot advertises depth and camera pose, otherwise monocular
MASt3R-SLAM; `--mode desktop` is the default. `--reconstruction auto|rgbd|slam|video` saves a
per-robot override. Missing CUDA, loading, lost tracking and SLAM failures show a flat live
camera panel in desktop and VR, with the reason in ImGui; controls stay connected.

**Settings** in the main menu saves app-wide choices immediately for the next connection.
Choose MASt3R-SLAM Full (fp32) or Half (bf16 where supported, otherwise fp16), and
100/75/50/25% input pixels. The whole camera frame fits a patch grid up to 512 pixels
on its long side; the overlay shows the active precision. Out-of-memory frames are
dropped at the chosen precision with a warning until memory recovers. Lower settings trade detail for speed and VRAM. Max splats defaults to 1,000,000;
use −/+ to step by 100,000 or type a count on desktop. Reset restores all defaults.

**Diagnostic logging** is off by default. Toggle it in Settings, or set `ITO_DEBUG=1` to
force it on (`0` forces it off). `diagnostics.jsonl` beside the settings file retains up to
8 MiB across four rotating files, with run/build IDs and operational timings and counters.
It excludes pairing codes, credentials, media and input contents; hover its description for the path.

XR uses the active OpenXR runtime (SteamVR, Virtual Desktop/VDXR, or Monado) and OpenGL 4.3.
Use `--reference-space seated` (default) or `standing` after room setup. Each eye renders at
the runtime display rate from the current predicted pose, independently of the robot stream.
Aim a controller and press its trigger to use the world-locked ImGui panel; it appears while
a controller points at its controls, while the robot is not driving (stopped, e-stopped, faulted
or the link down) so it says why, or when pinned with a left stick click. A trigger press that
begins on the panel never reaches the robot. The companion window has the same controls. Left
stick drives; on Touch controllers A resumes, B e-stops, X stops and Y recenters. Other
controllers can use the panel (right menu also e-stops).
Home or the panel recenters: your gaze becomes the robot's current view, and motion stops;
resume explicitly afterward.
Tracking/focus loss disarms input. E-stop and link loss pulse the controllers.
Role-assigned Vive trackers are sent when the runtime supports `XR_HTCX_vive_tracker_interaction`.
F12 saves left-eye, right-eye and panel images. On Linux, `XR_RUNTIME_JSON` selects a runtime.

Comfort and input preferences are saved per driver address and robot name: `--fov 75`,
`--sensitivity 0.0025`, `--invert-y` / `--no-invert-y`, `--move-x move_x`, `--move-y move_y`.
`--camera NAME --cameras N` selects a camera from a driver with N video tracks.
Max splats is global; old per-robot budgets are ignored. Software rendering and OpenGL
buffer limits cap the active count, and the live VRAM manager can lower it further without
changing the saved setting.
`ito-desktop scene.ply` remains a local scene viewer with WASD and Page Up/Down free flight.

`ito.reconstruction.Reconstruction(intrinsics, max_splats=window.max_splats)` accepts synchronized
`submit(rgb_uint8, depth_float32_metres, world_from_camera, capture_time)` frames and is a live
source for `DesktopWindow.run()`. Capture times use the pilot clock (`ito.clock.now`); poses use +Y up,
-Z forward. Use it as a context manager to own its worker process. Input drops when busy;
changed slots coalesce in a bounded shared-memory ring. The map is one persistent set of
Gaussians in the connection's odometry frame; tracking keyframes come and go without it.
An observation adds Gaussians only where the map, rendered from that view, leaves pixels
uncovered, shows a surface clearly in front, or draws it much coarser than the view's pixels;
Gaussians a view repeatedly sees through retire, and matching ones gather support and the
view's light. With CUDA and the `slam` dependencies, patches of up to 24 retained views are
refined photometrically (gsplat) in a bounded share of the worker's time. Under Max splats or
VRAM pressure, unsupported and contradicted Gaussians go before the least recently seen.

Monocular SLAM needs an NVIDIA GPU. Models and their licence notices are included in
`models/` beside the app. Missing or damaged models keep the live flat camera feed usable.
A robot that streams its own camera motion (Microduck does) is tracked robot-first: its
source-stamped 50 Hz camera pose is fused with MASt3R's relative poses in a 12-keyframe
window, in the robot's odometry frame, so there is no map to restart and a frame vision
cannot match still gets a pose; so does one the camera's turn blurs, which adds no splats.
The frame delay is estimated online against the robot's motion. Corrections reach the view at most 2 degrees/s and 5 cm/s,
only while it moves. `--tracking vision` saves a per-robot switch back to MASt3R-SLAM alone
(four keyframes, map restarts) for comparison. Monocular scale is estimated, so
reconstructed distances are not a measurement tool.

On Microduck, from this checkout: `uv run --project drivers/microduck ito-driver-microduck`; connect with `uv run ito <duck-address>:8081` and enter its printed pairing code.
Microduck must have robotd/mediad running and be enabled with `robotctl robot enable`; G/right trigger opens the beak, C crouches.
The driver streams the head camera's pose at robotd's 50 Hz, stamped when robotd read its sensors; The beta's head IMU (robotd) and ToF (tofd) are used when present; `--no-head-imu` and `--no-tof` leave them alone.

Virtual Microduck on Ubuntu: `drivers/microduck/setup-sim.sh /path/to/prefix`, then `/path/to/prefix/microduck-sim --viewer`.
It builds Pollen's stack at the `third_party/` submodule pins (microduck at a release tag, microduck_rl's develop) and runs this checkout's driver.
On Windows, use the prefix `~/.local/share/ito-microduck` in the default WSL distro; `drivers/microduck/microduck-sim.cmd` and `ito-microduck.cmd` launch it from there.
From built checkouts: `uv run --extra microduck python -m drivers.microduck.sim --microduck /path/to/microduck --rl /path/to/microduck_rl --policies /path/to/policies/current [--viewer] [--camera-height 360|720|1080]` (the Microduck beta, with its fisheye lens's datasheet figures until a beta is calibrated; `drivers/microduck/lens.py`).
Enter the printed address and pairing code in Ito; Ctrl+C or closing the viewer stops the complete robot.

## Development and releases

Windows release, from a VS 2022 x64 developer shell with CUDA Toolkit 12.4:
`uv run --python 3.12 --extra slam python release/windows.py` writes the `dist/ito` folder
(Python runtime, `ito.exe`, and `models/` from `ito.build` with its CC BY-NC-SA licences;
`--models DIR` reuses a built bundle).

Use `ito-driver-mujoco --rgb-only` to omit both depth and pose for manual piloting.

The MuJoCo driver needs `libosmesa6` on Debian/Ubuntu for headless software rendering;
use `--gl egl` with GPU drivers. It serves a textured room and a wheeled pan/tilt robot.
Pilot head yaw/pitch drives the head; `PilotState.axes` uses `move_y` for forward and
`move_x` for right turn, both in [-1, 1], with `deadman=true`. Timeout, stop, and e-stop
brake the wheels and hold the head. E-stop stays latched until resume and fresh input.

Supply an MJCF path to pilot another differential-drive robot. `--camera`, `--base`,
`--pan`, `--tilt`, `--left`, and `--right` select its camera, base body, and actuators;
`--wheel-radius` and `--axle-width` set its geometry. Models use +X forward, +Y left,
+Z up, limited head position servos (positive left/up), and wheel velocity servos
(positive forward), all with unit gear. `--help` lists video, speed, and link options.
RGB, optical-axis depth in millimetres, and camera pose share one physics snapshot;
metadata `video_pts` is the source video timestamp in 90 kHz ticks, starting at zero
per connection. Poses use Ito coordinates relative to the base at startup.

Audio uses the system default microphone and speakers (select the headset as the default
output in VR). The shared desktop/VR panel mutes the microphone and speakers independently.
Linux device audio needs PortAudio (`libportaudio2` on Debian/Ubuntu); missing devices leave
piloting available with an audio status line. Both CLIs accept `--audio-source device|none|tone:440`
and `--audio-sink device|none|capture.wav`. MuJoCo has no microphone unless given a source.

Testing is running the software. On the Windows PC, build a branch with `release.ps1 -Branch NAME`,
then in WSL `uv run tools/session.py NAME RUN` drives it against the virtual Microduck with a
pilot script (`tools/scripts/`) and writes `screen.mp4` and `camera.mp4` to `out\sessions\RUN`;
`uv run tools/frame.py VIDEO FRAME [-n COUNT]` saves a frame by number, or COUNT neighbouring frames side by side; steps.txt says which frame each script step starts at.
