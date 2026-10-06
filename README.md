# Ito

Ito is immersive teleoperation software built entirely for piloting robots.

Most teleoperation software treats the pilot experience as secondary. It is usually a basic tool for collecting demonstrations and training robot policies. Ito takes a different approach: it is not designed to train AI. Its sole purpose is to make remotely operating a robot comfortable for the human pilot. We envision a future where people pilot every type of robot from their home or office. This could enable disabled people to act through robots in places their bodies cannot easily take them, and allow people to explore or work in environments that are hostile to humans. Ito is intended to support humanoids, droids, vehicles, mechas, and robot forms that do not fit an existing category. It translates the pilot's tracked pose and controller input into control instructions appropriate to the piloted robot. In the other direction, it translates the robot's sensor input into a comfortable immersive 3D reconstruction of its surroundings.

## Usage

Python 3.12+ and [uv](https://docs.astral.sh/uv/) are required.

```sh
uv sync
uv run ito-driver your_robot.adapter:create --host 0.0.0.0 --port 8080
uv run ito-link robot-address:8080
```

The driver command loads a `module:factory` returning `ito.driver.Adapter`;
`--adapter-args` accepts its configuration as a JSON object. The adapter supplies camera/audio
tracks, robot description and nonblocking `apply`/`neutral` commands. It publishes capture
metadata through `publish_frame`. The driver defaults to a 250 ms input timeout and 90 Hz
command limit; stop and e-stop take effect immediately. Resume requires fresh deadman input.

`ito.link.connect(address)` returns an asynchronous `Peer` context manager. Its `tracks` queue
provides media with bounded buffering, `messages` provides validated control/metadata/status,
and `frames` holds the latest metadata by camera. Send `PilotState` or `Command` with `send`;
a false result means the channel is unavailable or backed up, so retry commands and send the
next fresh pilot state. Timestamps use `time.monotonic()`; `peer.clock.remote_to_local` converts
remote capture times. `ito-link` consumes media and prints live link/driver metrics.

Verify with separate driver and pilot processes over real WebRTC (no GPU or headset needed):

```sh
uv run python e2e/webrtc.py
uv run python e2e/lifecycle.py
```

These runs check video, bidirectional audio, pose/depth, malformed-message fuzzing, deadman,
command rate limiting, timeout after killing the pilot, reconnects, the e-stop latch and adapter
failures. Robot fixtures live only in `e2e/`.
