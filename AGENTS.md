# Ito — instructions for agents

Ito is immersive teleoperation: one pilot, one robot. The pilot wears a VR headset; Ito sends their
tracked pose and controller input to the robot, and turns the robot's camera feed into a live 3D
Gaussian-splat scene rendered at headset rate. The pilot's view never waits on the robot. Comfort
is the most important property of the product. "Pilot"/"piloting" is the word, used consistently —
not "operator", not "user" when the pilot is meant.

## Rules

- **No unit tests.** Verify like a user: end-to-end runs, fuzzing, driving the real programs.
  Launch the app and a driver, inject input (xdotool, a simulated OpenXR runtime, playwright-cli for
  anything browser-based), take screenshots, read logs and metrics. Those scripts live in `e2e/`,
  each runnable with one command.
- **Docs to a minimum.** Let the code explain the software: clear names, small modules, a short
  header comment where the *why* is not obvious. No `docs/` tree, no design essays. The README is
  the pitch plus usage. This file is the only other prose.
- **No prototypes.** No "v0.1", no "for now", no "TODO: later", no mock paths pretending to be the
  product. Every feature that lands is complete, handles its failures, and is something a pilot
  would use. If it cannot be finished, it does not land.
- **Out of scope:** fleets, a central server, session management, robot lobbies, more than one
  pilot. One Ito instance pilots exactly one robot.
- Keep the code base small. Reach for maintained libraries before writing infrastructure.
- Small, coherent commits with descriptive messages. Never commit secrets or model weights.
- Priorities, in order: pilot comfort, low pilot→robot→pilot latency, long-term maintainability.
  Hot paths (frame decode, Gaussian upload, sort, draw) stay on the GPU and out of Python loops;
  boundaries (protocol, reconstruction backend, renderer input) are narrow so any one piece can be
  swapped for a faster implementation without touching the rest.
- UI is Dear ImGui (imgui-bundle): a window in desktop mode, and the same UI drawn to a texture on
  a panel in VR. No hand-built widget toolkit.

## Commits

Every commit says who made it:

- Author and committer: `Eris <339081058+eris-shaped@users.noreply.github.com>`.
- `Co-authored-by: arteeh <35239587+arteeh@users.noreply.github.com>` on every commit: the project
  owner.
- One more `Co-authored-by` per coding agent that actually wrote part of the change:
  `Claude Opus 5.5 <noreply@anthropic.com>` or `Codex <noreply@openai.com>`. Never for a tool that
  did not contribute.

GitHub noreply addresses only; never a personal email or an internal host name. A local
`commit-msg` hook adds the trailers (the agent's from `ITO_AGENT_COAUTHOR`), so agents leave them
out of their commit messages.

## Architecture

Two programs and the protocol between them.

```
 robot computer                                    pilot PC (Windows/Linux, NVIDIA first)
┌──────────────────────┐   WebRTC (one session)   ┌───────────────────────────────────────────┐
│ ito-driver-<robot>   │ ── video (+depth/pose) ─►│ ito                                       │
│  adapter: Ito ⇄ SDK  │ ── audio ──────────────► │  link ──► reconstruction (own process,    │
│  safety: deadman,    │ ── telemetry ──────────► │           GPU, 5–30 Hz) ──► splat buffer  │
│  e-stop, neutral     │ ◄── pilot pose/input ─── │  renderer (OpenXR or desktop, 90–120 Hz)  │
│                      │ ◄── audio ────────────── │  input (OpenXR / keyboard+mouse+gamepad)  │
└──────────────────────┘                          └───────────────────────────────────────────┘
```

- `ito/protocol` — the one language Ito speaks to every robot: robot description (name, cameras
  with intrinsics/extrinsics, capabilities, actuated degrees of freedom), per-frame metadata (capture
  time, camera pose if the robot knows it, optional depth), pilot state (head/hand/tracker poses,
  buttons, axes), commands (stop, e-stop, resume). Versioned and validated on receipt.
- `ito/link` — WebRTC (aiortc) between pilot app and driver. The driver listens; the pilot app
  connects to its address. Media tracks for video/audio; an unreliable unordered datachannel for
  pilot state (newest wins); a reliable one for everything else. Clock-offset estimation so capture
  timestamps are comparable on both ends.
- `ito/driver` — what every robot driver builds on: link server, safety (input timeout → neutral,
  e-stop latch), rate limiting, the adapter interface a robot implements.
- `drivers/<robot>` — one adapter per robot. `mujoco` pilots any MJCF robot in simulation and
  renders its cameras (RGB + depth + ground-truth pose); `microduck` pilots the Pollen Robotics
  Microduck. Later: humanoids, drones, cars, boats.
- `ito/reconstruction` — turns the camera stream into Gaussians in a world frame anchored to where
  the robot started. Backends: posed RGB-D (driver supplies depth and pose: simulation, depth or
  stereo cameras) and monocular dense SLAM (MASt3R-SLAM, CUDA). Keeps a temporal window: recent
  observations dominate, older ones fade, so the scene is "now plus a little memory". Runs in its
  own process and hands Gaussians to the renderer through shared memory. Never blocks rendering.
- `ito/render` — OpenGL Gaussian-splat renderer (GPU depth sort + instanced quads), vendor-neutral
  so NVIDIA and AMD both work. Every display frame it draws the latest scene from the pilot's
  *current* head pose, anchored to the robot camera's latest pose. Overlays: link/latency status,
  robot state, e-stop.
- `ito/xr` — OpenXR session (pyopenxr): stereo swapchains, head/controller/tracker poses, haptics.
  Any OpenXR runtime (SteamVR, Monado); standalone headsets via Virtual Desktop.
- `ito/desktop` — the same renderer in a window with mouse-look, keyboard and gamepad. A real
  pilot mode for people without a headset, and what e2e runs use under Xvfb.
- `ito/app` — wires it together: `ito <driver-address>` on the pilot PC, `ito-driver-<robot>` on
  the robot.

Every stage is asynchronous: pilot input ~60–90 Hz, video ~30–60 Hz, reconstruction as fast as the
model allows, rendering at the display rate. No stage waits for another.

Python 3.12+, managed with `uv`. The AI ecosystem lives in Python, and the Gaussians stay on one
GPU in one process tree instead of being streamed between programs.

## Environment notes

- `uv run ito …` runs the pilot app; `uv run ito-driver-mujoco …` runs a simulated robot.
- The main dev box has no GPU and no headset: desktop mode under `xvfb-run` with Mesa llvmpipe must
  work, and CUDA-only backends must fail with a clear message instead of crashing.
- Monado with its simulated HMD exercises the OpenXR path without hardware.
