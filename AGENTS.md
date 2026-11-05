# Ito, for agents

What Ito is and why: the intro of [README.md](README.md). The 3D reconstruction is a choice in Settings, one per session (`ito/recon`): the ToF mesh (the robot's ToF ranges fused into a mesh, placed by the robot's own pose, painted with its camera frames) with guessed depth beyond the ToF's view or a depth model's maps (Prompt Depth Anything, Depth Anything 3, Prior Depth Anything, MapAnything), or MASt3R-SLAM's Gaussian splats.

## Where things are

- `ito/app` pilot app entry, menus, the link loop (`pilot.py`) · `ito/desktop` window, input, overlay, recorder · `ito/xr` OpenXR · `ito/recon` the reconstruction methods, one package each (`__init__` light: label, sizes, weights; `model.py` the network a depth method runs; `mast3r_slam` its worker, splat renderer and bundle build) · `ito/tofmesh` the mesh: fusion and its worker process, the depth models' process (`depth.py`), rendering · `ito/_vendor` third-party inference code, each with its NOTICE · `ito/render` Vulkan (`vk.py`), camera frames from the decoder (`camera.py`, `d3d12.py`), pose maths, the flat video panel, GPU and display timing · `ito/shaders` GLSL, compiled to SPIR-V by `tools/shaders.py` · `ito/ui` theme
- `ito/protocol` wire messages · `ito/link` WebRTC · `ito/driver` what a robot driver builds on
- `drivers/microduck` the Microduck driver; the virtual Microduck: `robot.py` and `Dockerfile` (robot side, run by `compose.yaml`), `sim.py` and `sim_body.py` (the MuJoCo body, on the host)
- `third_party/` microduck, microduck_rl and gst-plugins-rs pins · `release/` Windows packaging · `tools/` build and session scripts

## Building and running

- Source lives in WSL on ceres: clone `~/Projects/ito/main`, worktrees `~/Projects/ito/work/<name>` (`git submodule update --init --depth 1` in each). `tools/build.sh [REF]` builds Windows release `%USERPROFILE%\Projects\ito\builds\<branch>\ito`.
- `tools/sim.sh` starts the virtual Microduck; `uv run --extra sim tools/session.py BUILD NAME` runs a build against it with a pilot script (`tools/scripts/`) and records the window (`screen.mp4`, logs). One virtual Microduck at a time: compose project `ito`, fixed ports.
