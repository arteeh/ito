# Ito

Ito is immersive teleoperation software built entirely for piloting robots.

Most teleoperation software treats the pilot experience as secondary. It is usually a basic tool for collecting demonstrations and training robot policies. Ito takes a different approach: it is not designed to train AI. Its sole purpose is to make remotely operating a robot comfortable for the human pilot. We envision a future where people pilot every type of robot from their home or office. This could enable disabled people to act through robots in places their bodies cannot easily take them, and allow people to explore or work in environments that are hostile to humans. Ito is intended to support humanoids, droids, vehicles, mechas, and robot forms that do not fit an existing category. It translates the pilot's tracked pose and controller input into control instructions appropriate to the piloted robot. In the other direction, it translates the robot's sensor input into a comfortable immersive 3D reconstruction of its surroundings.

## Usage

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run ito-driver your_robot.adapter:create --port 8080   # on the robot
uv run ito-driver-mujoco --gl osmesa                      # furnished simulation
uv run ito robot-address:8080                             # on the pilot PC
uv run ito robot-address:8080 --mode xr                    # active OpenXR runtime
uv run ito-desktop scene.ply                              # explore a splat scene
```

Press **R** to pilot. WASD drives/turns the robot; click or Tab captures mouse-look to aim
its head. Space stops, **E e-stops**, R resumes, Home recenters, F12 captures a screenshot,
and Escape quits. Gamepad left stick drives, right stick looks; A resumes, B e-stops, X stops.
Focus loss stops motion. After a lost link Ito reconnects automatically; press R to resume.
The ImGui overlay reports driver state, link RTT, input latency, and capture-to-visible latency.
`--metrics path.jsonl` records these alongside display timing. Live scenes require posed RGB-D; `--mode desktop` is the default.

XR uses the active OpenXR runtime (SteamVR, Virtual Desktop/VDXR, or Monado) and OpenGL 4.3.
Use `--reference-space seated` (default) or `standing` after room setup. Each eye renders at
the runtime display rate from the current predicted pose, independently of the robot stream.
Aim a controller and press its trigger to use the world-locked ImGui panel; the companion
window has the same controls. Left stick drives; on Touch controllers A resumes, B e-stops,
X stops and Y recenters. Other controllers can use the panel (right menu also e-stops).
Home or the panel recenters position and yaw and stops motion; resume explicitly afterward.
Tracking/focus loss disarms input. E-stop and link loss pulse the controllers.
Role-assigned Vive trackers are sent when the runtime supports `XR_HTCX_vive_tracker_interaction`.
F12 saves left-eye, right-eye and panel images. On Linux, `XR_RUNTIME_JSON` selects a runtime.

XR end-to-end check on Windows: start SteamVR or Virtual Desktop with VDXR set as the active
OpenXR runtime, wear the headset, then run `uv run python e2e/openxr.py` from the checkout.
It launches its own MuJoCo driver and saves eye captures, logs and metrics in `e2e/out/xr`.
Add `--reference-space standing` to check room-scale space. The same script runs on Linux
with a configured OpenXR runtime, including Monado’s simulated HMD.

Comfort and input preferences are saved per driver address and robot name: `--fov 75`,
`--sensitivity 0.0025`, `--invert-y` / `--no-invert-y`, `--move-x move_x`, `--move-y move_y`.
`--camera NAME --cameras N` selects a camera from a driver with N video tracks.
The ImGui panel saves the robot's maximum splat count; lowering it fades excess splats first.
Defaults are 16K for software rendering, 256K for hardware, or 1M with at least 8 GB NVIDIA VRAM.
`ito-desktop scene.ply` remains a local scene viewer with WASD and Page Up/Down free flight.

`ito.reconstruction.Reconstruction(intrinsics, max_splats=window.max_splats)` accepts synchronized
`submit(rgb_uint8, depth_float32_metres, world_from_camera, capture_time)` frames and is a live
source for `DesktopWindow.run()`. Capture times use the pilot monotonic clock; poses use +Y up,
-Z forward. Use it as a context manager to own its worker process. Input drops when busy;
changed slots coalesce in a bounded shared-memory ring. `uv sync --extra cuda` enables CUDA
projection/voxelization on NVIDIA; the default automatically falls back to NumPy on CPU.

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

End-to-end checks (no GPU or headset needed):

```sh
uv run python e2e/webrtc.py
uv run python e2e/lifecycle.py
uv run python e2e/mujoco_driver.py                         # saves RGB-D samples in e2e/out/mujoco
uv run python e2e/reconstruction_faults.py
LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a uv run python e2e/app.py # live MuJoCo, SDL input, reconnect
LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a uv run python e2e/render.py
LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a uv run python e2e/desktop.py
LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a uv run python e2e/stream.py
LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a uv run python e2e/reconstruction.py  # two-minute live room
```
