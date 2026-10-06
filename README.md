# Ito

Ito is immersive teleoperation software built entirely for piloting robots.

Most teleoperation software treats the pilot experience as secondary. It is usually a basic tool for collecting demonstrations and training robot policies. Ito takes a different approach: it is not designed to train AI. Its sole purpose is to make remotely operating a robot comfortable for the human pilot. We envision a future where people pilot every type of robot from their home or office. This could enable disabled people to act through robots in places their bodies cannot easily take them, and allow people to explore or work in environments that are hostile to humans. Ito is intended to support humanoids, droids, vehicles, mechas, and robot forms that do not fit an existing category. It translates the pilot's tracked pose and controller input into control instructions appropriate to the piloted robot. In the other direction, it translates the robot's sensor input into a comfortable immersive 3D reconstruction of its surroundings.

## Usage

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run ito-driver your_robot.adapter:create --port 8080   # on the robot
uv run ito-link robot-address:8080                        # on the pilot PC
uv run ito-desktop scene.ply                              # explore a splat scene
```

Desktop controls: WASD move, click for mouse-look (Tab releases), Page Up/Down rise/fall,
Home recenter, Space stop, E e-stop, R resume, F12 screenshot, Escape quit. Gamepads work too.

End-to-end checks (no GPU or headset needed):

```sh
uv run python e2e/webrtc.py
uv run python e2e/lifecycle.py
LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a uv run python e2e/render.py
LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a uv run python e2e/desktop.py
LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a uv run python e2e/stream.py
```
