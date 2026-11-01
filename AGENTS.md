# Ito, for agents

What Ito is and why: the intro of [README.md](README.md). This is the ToF mesh version: the robot's ToF ranges fused into a mesh, placed by the robot's own pose, painted with its camera frames. No SLAM, no AI models.

## Where things are

- `ito/app` pilot app entry, menus, the link loop (`pilot.py`) · `ito/desktop` window, input, overlay, recorder · `ito/xr` OpenXR · `ito/tofmesh` the mesh: fusion and its worker process, rendering · `ito/render` pose maths, the flat video panel, GPU and display timing · `ito/ui` theme
- `ito/protocol` wire messages · `ito/link` WebRTC · `ito/driver` what a robot driver builds on
- `drivers/microduck` the Microduck driver; the virtual Microduck: `robot.py` and `Dockerfile` (robot side, run by `compose.yaml`), `sim.py` and `sim_body.py` (the MuJoCo body, on the host)
- `third_party/` microduck, microduck_rl and gst-plugins-rs pins · `release/` Windows packaging · `tools/` build and session scripts

## How to work

- Source lives in WSL on ceres: clone `~/Projects/ito/main`, worktrees `~/Projects/ito/work/<name>` (`git submodule update --init --depth 1` in each). `tools/build.sh [REF]` builds Windows release `%USERPROFILE%\Projects\ito\builds\<branch>\ito`.
- No automated tests. Test by running the software: `uv run --extra sim tools/session.py BUILD NAME` runs a build against the virtual Microduck with a pilot script (`tools/scripts/`) and records the window; watch `screen.mp4` and read the logs. No numeric gates, no test-only scripts or flags.
- One virtual Microduck at a time: compose project `ito`, fixed ports.
- No Markdown files besides README.md and AGENTS.md, and no other docs. Below its intro, README.md stays short.
- No dead code, and as few flags and environment switches as possible.
- At most 50 GB of disk per job; delete what you record once judged.
- Branch + PR with `Closes #N` (GitHub project board arteeh/projects/1). Agents never merge, and never open, comment on or edit issues or PRs; report findings in your final message.
- Commits: author and committer `arteeh <35239587+arteeh@users.noreply.github.com>`, no trailers. No secrets, no model weights.
- Comments only where code would mislead without them (units, frames, platform quirks).
