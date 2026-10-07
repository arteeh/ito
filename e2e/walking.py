"""Real MuJoCo driver: look left, walk there, realign, reverse, and stand still.

LD_LIBRARY_PATH=/opt/data/lib/osmesa uv run python e2e/walking.py
"""

import asyncio
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from mujoco_driver import Pilot

from ito.driver import pairing
from ito.link import connect

OUT = Path("e2e/out/walking")


async def run():
    code_file = OUT / "pairing-code"
    code = pairing.rotate(code_file)
    with (OUT / "driver.log").open("w") as log:
        robot = await asyncio.create_subprocess_exec(
            str(Path(sys.executable).with_name("ito-driver-mujoco")),
            "--port",
            "0",
            "--pairing-file",
            str(code_file),
            stdout=asyncio.subprocess.PIPE,
            stderr=log,
            env=os.environ | {"MUJOCO_GL": "osmesa", "LP_NUM_THREADS": "2"},
        )
        pilot = None
        try:
            async with asyncio.timeout(30):
                line = await robot.stdout.readline()
            assert line.startswith(b"Ito driver listening"), (OUT / "driver.log").read_text()
            peer = await connect(
                line.decode().strip().rsplit(" ", 1)[1], receive_audio=False, code=code
            )
            pilot = Pilot(peer)
            await pilot.drive(1, yaw=math.pi / 2)
            _, status = await pilot.status("active")
            before = status.telemetry
            assert abs(before["base_yaw"]) < 0.03, before
            start = np.array([before["base_x"], before["base_y"]])
            await pilot.drive(4, yaw=math.pi / 2, forward=1)
            _, status = await pilot.status("active")
            after = status.telemetry
            displacement = np.array([after["base_x"], after["base_y"]]) - start
            yaw_error = abs(math.remainder(after["base_yaw"] - math.pi / 2, 2 * math.pi))
            assert displacement[1] > 0.8 and abs(displacement[0]) < displacement[1] * 0.3, after
            assert yaw_error < 0.1 and abs(after["head_pan"]) < 0.1, after
            turn_rates = [
                abs((s.telemetry["right_command"] - s.telemetry["left_command"]) * 0.14 / 0.52)
                for _, s in pilot.statuses
                if "right_command" in s.telemetry
            ]
            assert max(turn_rates) <= 1.200001
            await pilot.drive(1.5, yaw=math.pi / 2, forward=-1)
            _, status = await pilot.status("active")
            reverse = status.telemetry
            assert reverse["base_y"] < after["base_y"] - 0.4, reverse
            # A new gaze without movement must not chase the pilot's head with the base.
            await pilot.drive(1, yaw=0)
            _, status = await pilot.status("active")
            assert abs(status.telemetry["base_yaw"] - reverse["base_yaw"]) < 0.04, status
            report = {
                "look_left_displacement_m": displacement.tolist(),
                "displacement_heading_error_deg": math.degrees(
                    math.atan2(displacement[0], displacement[1])
                ),
                "body_heading_error_deg": math.degrees(yaw_error),
                "max_commanded_turn_rad_s": max(turn_rates),
            }
            (OUT / "summary.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2))
            print("PASS: look-relative forward/reverse, bounded body alignment, stationary look")
        finally:
            if pilot:
                await pilot.close()
            if robot.returncode is None:
                robot.terminate()
                try:
                    await asyncio.wait_for(robot.wait(), 10)
                except TimeoutError:
                    robot.kill()
                    await robot.wait()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    asyncio.run(run())
