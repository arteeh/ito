"""A passive observer in its own process: closing or stalling it cannot stop the robot."""

import asyncio
import contextlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np


async def publish(adapter, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp")

    def snapshot():
        with adapter._lock:
            state = np.concatenate(([adapter.data.time], adapter.data.qpos, adapter.data.qvel))
        with temporary.open("wb") as output:
            np.save(output, state)
        temporary.replace(path)

    failed = False
    while True:
        try:
            await asyncio.to_thread(snapshot)
            failed = False
        except OSError:
            if not failed:
                logging.getLogger(__name__).warning("Simulation viewer snapshot unavailable")
            failed = True
        await asyncio.sleep(1 / 30)


def main():
    # The driver's camera may use OSMesa; this independent process owns a GLFW window.
    os.environ["MUJOCO_GL"] = "glfw"
    import mujoco
    import mujoco.viewer

    model_path, state_path = map(Path, sys.argv[1:])
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    status = state_path.with_suffix(".viewer")
    updated = 0
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as view:
        view.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        view.cam.trackbodyid = model.body("base").id
        view.cam.distance = 2.8
        view.cam.elevation = -25
        last_frame = time.monotonic()
        while view.is_running():
            try:
                state = np.load(state_path, allow_pickle=False)
            except FileNotFoundError:
                state = None
            if state is not None and state[0] != data.time:
                with view.lock():
                    data.time = state[0]
                    data.qpos[:] = state[1 : 1 + model.nq]
                    data.qvel[:] = state[1 + model.nq :]
                    mujoco.mj_forward(model, data)
                view.sync()
                updated += 1
                last_frame = time.monotonic()
                # Report the displayed simulation time without recording poses.
                temporary = status.with_suffix(".tmp-viewer")
                temporary.write_text(json.dumps(dict(frames=updated, simulation_time=data.time)))
                temporary.replace(status)
            if time.monotonic() - last_frame > 5:
                raise RuntimeError("Simulation viewer stopped receiving the robot state")
            time.sleep(1 / 30)
    with contextlib.suppress(FileNotFoundError):
        status.unlink()


if __name__ == "__main__":
    main()
