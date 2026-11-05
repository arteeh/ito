# Ito

Ito is immersive teleoperation software built entirely for piloting robots.

Most teleoperation software treats the pilot experience as secondary. It is usually a basic tool for collecting demonstrations and training robot policies. Ito takes a different approach: it is not designed to train AI. Its sole purpose is to make remotely operating a robot comfortable for the human pilot. We envision a future where people pilot every type of robot from their home or office. This could enable disabled people to act through robots in places their bodies cannot easily take them, and allow people to explore or work in environments that are hostile to humans. Ito is intended to support humanoids, droids, vehicles, mechas, and robot forms that do not fit an existing category. It translates the pilot's tracked pose and controller input into control instructions appropriate to the piloted robot. In the other direction, it translates the robot's sensor input into a comfortable immersive 3D reconstruction of its surroundings.

## Use

Run `ito.exe` from a Windows build (it needs a GPU with Vulkan 1.3) and connect to a robot by its address. Ito shows the robot's surroundings as the reconstruction chosen in Settings (below); by default a mesh of its ToF ranges, placed by the robot's own pose and painted with its camera frames. `--mode xr` uses the active OpenXR runtime.

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
python tools/shaders.py                           # GLSL in ito/shaders to SPIR-V (builds do it)
uv run --extra sim tools/session.py BUILD NAME    # a scripted, recorded session against the sim
```

No automated tests and no docs besides this README: test by running Ito and watching it.

### Reconstruction methods

Settings → **Reconstruction** chooses how the robot's surroundings are built, for a whole
connection (it applies at the next one):

- **Guessed depth**: the ToF mesh; beyond the ToF's ~45° the renderer guesses depth from the
  surfaces it measured and the floor. No model, nothing extra loaded.
- **Prompt Depth Anything** (Small, Large): a metric depth map from each camera frame, prompted
  with the ToF's ranges.
- **Depth Anything 3** (Small, Base, Large, Giant, Metric): a depth map from each frame alone;
  the ToF fixes its scale (Metric's only checked).
- **Prior Depth Anything** (Small, Base, Large): the ToF's real readings completed into a dense
  metric depth map.
- **MapAnything**: a metric depth map from each frame and its lens.
- **MASt3R-SLAM**: its own camera poses and map from the camera alone, drawn as Gaussian splats
  (~11 GB VRAM at its defaults; its own settings below the choice).

The depth models run in a lower-priority process at their own rate and need an NVIDIA GPU with
CUDA; their maps feed the ToF mesh (live as a dense mesh, kept as keyframes), the ToF's real zone
centres check their scale and, inside its view, anchor them (**Inside** / **Outside**: how much
a map counts there, as a share of a ToF reading). Every model ships in the release's `models`
folder (`ito/recon/*/__init__.py` names them, their revisions and licences; some are
non-commercial); nothing is downloaded at runtime, and a missing model is an error, never another
method in its place. None of this has been tested with a real robot or a real headset.
