# Ito

Ito is immersive teleoperation software built entirely for piloting robots.

Most teleoperation software treats the pilot experience as secondary. It is usually a basic tool for collecting demonstrations and training robot policies. Ito takes a different approach: it is not designed to train AI. Its sole purpose is to make remotely operating a robot comfortable for the human pilot. We envision a future where people pilot every type of robot from their home or office. This could enable disabled people to act through robots in places their bodies cannot easily take them, and allow people to explore or work in environments that are hostile to humans. Ito is intended to support humanoids, droids, vehicles, mechas, and robot forms that do not fit an existing category. It translates the pilot's tracked pose and controller input into control instructions appropriate to the piloted robot. In the other direction, it translates the robot's sensor input into a comfortable immersive 3D reconstruction of its surroundings.

## Usage

Desktop splat viewer (Python 3.12+, `uv`, OpenGL 4.3):

```sh
uv run ito-desktop scene.ply
# Generate a small standard 3DGS scene to explore:
uv run python e2e/sample_scene.py e2e/out/courtyard.ply
uv run ito-desktop e2e/out/courtyard.ply
```

WASD moves; click captures mouse-look and Tab releases it. Page Up/Down rises/falls;
Home recenters. Gamepad sticks move/look, shoulders rise/fall. Space/X requests stop,
E/B requests e-stop, R/A requests resume. F12 saves a screenshot; Escape quits.
The file viewer is offline; robot status and e-stop confirmation come from the driver
when the app supplies them. `--position X Y Z`, `--yaw`, `--pitch`, `--fov`, and
`--speed` place the camera in an existing scene. Poses use +X right, +Y up, -Z forward.

`GaussianRenderer.upload(GaussianBuffer)` accepts contiguous float32 records, including
shared-memory NumPy views; `draw(robot_camera, head, projection, target)` uses fresh poses
without waiting for a scene update. `DesktopWindow.run(source, state=..., on_input=...)`
polls a nonblocking `SceneSource` and exposes pilot input and driver status independently.
The buffer layout and ownership contract are in `ito/render/scene.py`.

End-to-end checks use a real GL window, generated PLYs, screenshots, `xdotool`, and an
SDL virtual gamepad. On Debian/Ubuntu install `xvfb xauth xdotool libgl1 libgl1-mesa-dri`:

```sh
LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a uv run python e2e/render.py
LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a uv run python e2e/desktop.py
LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a uv run python e2e/stream.py
```

Captures and metrics go to `e2e/out/`. The streaming check runs a separate producer
through shared memory, including stalled updates, focus loss and controller unplugging.
