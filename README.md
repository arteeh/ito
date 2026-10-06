# Ito

Ito is immersive teleoperation software built entirely for piloting robots.

Most teleoperation software treats the pilot experience as secondary. It is usually a basic tool for collecting demonstrations and training robot policies. Ito takes a different approach: it is not designed to train AI. Its sole purpose is to make remotely operating a robot comfortable for the human pilot. We envision a future where people pilot every type of robot from their home or office. This could enable disabled people to act through robots in places their bodies cannot easily take them, and allow people to explore or work in environments that are hostile to humans. Ito is intended to support humanoids, droids, vehicles, mechas, and robot forms that do not fit an existing category. It translates the pilot's tracked pose and controller input into control instructions appropriate to the piloted robot. In the other direction, it translates the robot's sensor input into a comfortable immersive 3D reconstruction of its surroundings.

## Usage

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run ito-driver your_robot.adapter:create --port 8080   # on the robot
uv run ito-driver-mujoco --gl osmesa                      # furnished simulation
uv run ito-link robot-address:8080                        # on the pilot PC
uv run ito-desktop scene.ply                              # explore a splat scene
```

Desktop controls: WASD move, click for mouse-look (Tab releases), Page Up/Down rise/fall,
Home recenter, Space stop, E e-stop, R resume, F12 screenshot, Escape quit. Gamepads work too.

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
LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a uv run python e2e/render.py
LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a uv run python e2e/desktop.py
LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a uv run python e2e/stream.py
```
