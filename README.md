# Ito

Ito is immersive teleoperation software built entirely for piloting robots.

Most teleoperation software treats the pilot experience as secondary. It is usually a basic tool for collecting demonstrations and training robot policies. Ito takes a different approach: it is not designed to train AI. Its sole purpose is to make remotely operating a robot comfortable for the human pilot. We envision a future where people pilot every type of robot from their home or office. This could enable disabled people to act through robots in places their bodies cannot easily take them, and allow people to explore or work in environments that are hostile to humans. Ito is intended to support humanoids, droids, vehicles, mechas, and robot forms that do not fit an existing category. It translates the pilot's tracked pose and controller input into control instructions appropriate to the piloted robot. In the other direction, it translates the robot's sensor input into a comfortable immersive 3D reconstruction of its surroundings.

## Use

Run `ito.exe` from a Windows build and connect to a robot by its address. Ito shows the robot's surroundings as a mesh: its ToF ranges, placed by the robot's own pose and painted with its camera frames. `--mode xr` uses the active OpenXR runtime.

Press **R** to pilot. WASD moves where you look; click or Tab for mouse-look. Space stops, **E e-stops**, Home recenters, F12 saves a screenshot, Escape releases the mouse. Gamepad: left stick moves, right stick looks, A resumes, B e-stops, X stops. In VR, aim at the panel and pull the trigger; A resumes, B e-stops, X stops, Y recenters.

## Virtual Microduck

In WSL on the Windows PC, from this checkout:

```sh
tools/sim.sh   # robotd, tofd, mediad and Ito's driver in Docker, and the MuJoCo body with its viewer
```

Connect Ito to `localhost:8081` with the pairing code it prints. Closing the viewer stops the sim.

## Development

```sh
tools/build.sh [REF]                              # Windows build, into ~\Projects\ito\builds
uv run --extra sim tools/session.py BUILD NAME    # a scripted, recorded session against the sim
```

No automated tests and no docs besides this README: test by running Ito and watching it.