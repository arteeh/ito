"""Real MuJoCo driver: look left, walk there, realign, reverse, stand, look past the head,
glance at the edge of its range.

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

from ito import clock
from ito.driver import pairing
from ito.driver.walking import FACING, SOFT_LIMIT
from ito.link import connect
from ito.protocol import Command

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
            assert peer.send(Command(sequence=0, action="resume"))
            # Standing, a look the head can reach leaves the body where it is.
            await pilot.drive(1, yaw=1.0)
            _, status = await pilot.status("active")
            before = status.telemetry
            assert abs(before["head_pan"] - 1.0) < 0.05, before
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
            # A new gaze within the head's range must not chase the pilot's head with the base.
            facing = reverse["base_yaw"]
            await pilot.drive(1, yaw=facing - 1.0)
            _, status = await pilot.status("active")
            assert abs(status.telemetry["base_yaw"] - facing) < 0.04, status
            # Past the pan limit, the standing body turns until it faces the gaze, the head back
            # in the middle of its range, not just until the head can reach.
            limit = next(
                d.maximum for d in peer.description.degrees_of_freedom if d.name == "head_pan_joint"
            )
            beyond = math.radians(30)
            gaze = facing + limit + beyond
            expected = limit + beyond
            following = clock.now()
            await pilot.drive(6, yaw=gaze)
            _, status = await pilot.status("active")
            reached = status.telemetry
            body_turn = math.remainder(reached["base_yaw"] - facing, 2 * math.pi)
            view_error = math.remainder(
                reached["base_yaw"] + reached["head_pan"] - gaze, 2 * math.pi
            )
            assert abs(body_turn - expected) < FACING + math.radians(1), reached
            assert abs(reached["head_pan"]) < FACING + math.radians(1), reached
            assert abs(view_error) < math.radians(3), reached
            follow = [
                s.telemetry for t, s in pilot.statuses if t >= following and s.state == "active"
            ]
            follow_turns = [(f["right_command"] - f["left_command"]) * 0.14 / 0.52 for f in follow]
            # Smooth and bounded: never reverses, never faster than the robot's turn limit.
            assert min(follow_turns) >= -1e-6 and max(follow_turns) <= 1.200001, follow_turns
            overshoot = max(math.remainder(f["base_yaw"] - facing, 2 * math.pi) for f in follow)
            assert overshoot - expected < math.radians(2), overshoot
            # Looking back where the body faced before turns it back to face there: a walk then
            # starts the way the pilot looks (#30: a body left turned walked off diagonally).
            ahead = facing + beyond
            await pilot.drive(6, yaw=ahead)
            _, status = await pilot.status("active")
            settled = status.telemetry
            back = math.remainder(settled["base_yaw"] - ahead, 2 * math.pi)
            assert abs(back) < FACING + math.radians(1), settled
            assert abs(settled["head_pan"]) < FACING + math.radians(1), settled
            looked = math.remainder(settled["base_yaw"] + settled["head_pan"] - ahead, 2 * math.pi)
            assert abs(looked) < math.radians(3), settled
            # A glance just past the soft limit, still within the head's reach, turns the body
            # to face it too, gentler than a look past the limit.
            facing = settled["base_yaw"]
            glance = facing + (SOFT_LIMIT + 1) / 2 * limit
            glancing = clock.now()
            await pilot.drive(5, yaw=glance)
            _, status = await pilot.status("active")
            glanced = status.telemetry
            glance_turn = math.remainder(glanced["base_yaw"] - facing, 2 * math.pi)
            glance_pan_error = glanced["head_pan"]
            assert abs(glance_pan_error) < FACING + math.radians(1), glanced
            glance_turns = [
                (s.telemetry["right_command"] - s.telemetry["left_command"]) * 0.14 / 0.52
                for t, s in pilot.statuses
                if t >= glancing and s.state == "active"
            ]
            assert 0 < max(glance_turns) < max(follow_turns), (glance_turns, follow_turns)
            report = {
                "look_left_displacement_m": displacement.tolist(),
                "displacement_heading_error_deg": math.degrees(
                    math.atan2(displacement[0], displacement[1])
                ),
                "body_heading_error_deg": math.degrees(yaw_error),
                "max_commanded_turn_rad_s": max(turn_rates),
                "gaze_beyond_pan_limit_deg": math.degrees(beyond),
                "standing_body_turn_deg": math.degrees(body_turn),
                "standing_view_error_deg": math.degrees(view_error),
                "standing_turn_overshoot_deg": math.degrees(overshoot - expected),
                "max_follow_turn_rad_s": max(follow_turns),
                "body_error_after_looking_back_deg": math.degrees(back),
                "glance_body_turn_deg": math.degrees(glance_turn),
                "glance_head_pan_error_deg": math.degrees(glance_pan_error),
                "max_glance_turn_rad_s": max(glance_turns),
            }
            (OUT / "summary.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2))
            print(
                "PASS: look-relative forward/reverse, bounded body alignment, stationary look, "
                "body follows the head near and past its pan limit until it faces the gaze, "
                "and turns back when the pilot looks back"
            )
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
