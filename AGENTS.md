# Ito, for agents

What Ito is and why: the intro of [README.md](README.md). The plan being built: [docs/big-picture.md](docs/big-picture.md).

## Where things are

- `ito/app` pilot app entry, menus, the link loop (`pilot.py`) · `ito/desktop` window, input, overlay, recorder · `ito/xr` OpenXR · `ito/render` splat renderer · `ito/ui` theme
- `ito/reconstruction` worker process: robot-first SLAM (`slam.py`, `fusion.py`, `mast3r_frontend.py`), RGB-D fusion (`rgbd.py`), VRAM (`memory.py`)
- `ito/protocol` wire messages · `ito/link` WebRTC · `ito/driver` what every robot driver builds on
- `drivers/microduck` Microduck driver and its simulator (`sim.py`, `setup-sim.sh`) · `drivers/mujoco` generic MJCF robot
- `third_party/` microduck and microduck_rl pins · `release/` Windows packaging · `tools/` session recorder and frame viewer

## How to work

- Source work happens in WSL on ceres: clone `~/Projects/ito/main`, worktrees `~/Projects/ito/work/<name>`. Windows only builds: `release.ps1 -Branch <branch>` → `C:\Users\me\Projects\ito\builds\<branch>\ito`.
- Do not write (automated) tests. Test your work simply by running the software: `uv run tools/session.py BUILD NAME` records a scripted session against the virtual Microduck (`tools/scripts/`; every run of a script walks the same path), `uv run tools/frame.py VIDEO FRAME [-n N]` shows a frame and its neighbours (frame numbers per step in `steps.txt`). `session.py --env ITO_DUMP_MAP=<dir>` saves the map as it goes and `tools/mapview.py` (a build's Python) draws such maps side by side from fixed places in the sim room. Compare against main's build by eye. No unit tests, no numeric gates.
- Never change the shared sim install (`~/.local/share/ito-microduck`) from a branch; use `--sim-source`/`--sim-prefix`.
- At most 50 GB of disk per job; delete what you record once judged.
- Branch + PR with `Closes #N` (GitHub project board arteeh/projects/1). Agents never merge, and never open, comment on or edit issues or PRs; report findings in your final message.
- Commits: author and committer `arteeh <35239587+arteeh@users.noreply.github.com>`, no trailers. No secrets, no model weights.
- Comments only where code would mislead without them (units, frames, platform quirks).
