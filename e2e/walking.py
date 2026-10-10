"""Real MuJoCo driver: look left, walk there, realign, reverse, stand, look past the head.

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
from ito.driver.walking import FOLLOW, RETURN, SETTLE, SOFT_LIMIT
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
            # Standing, a look inside the soft limit leaves the body where it is.
            await pilot.drive(1.5, yaw=1.0)
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
            limit = next(
                d.maximum for d in peer.description.degrees_of_freedom if d.name == "head_pan_joint"
            )
            turn_limit = FOLLOW * 1.2  # of the driver's default turn speed

            def turns_since(since):
                return [
                    (s.telemetry["right_command"] - s.telemetry["left_command"]) * 0.14 / 0.52
                    for t, s in pilot.statuses
                    if t >= since and s.state == "active" and "right_command" in s.telemetry
                ]

            # A glance past the soft limit that moves on before it settles turns nothing.
            glance = facing + (SOFT_LIMIT + 1) / 2 * limit
            await pilot.drive(SETTLE * 0.6, yaw=glance)
            await pilot.drive(1.5, yaw=facing)
            _, status = await pilot.status("active")
            glanced = math.remainder(status.telemetry["base_yaw"] - facing, 2 * math.pi)
            assert abs(glanced) < 0.03, status
            # Past the pan limit and held, the standing body turns slowly, only until the head is
            # back inside its range with room to spare, not round to face the gaze.
            beyond = math.radians(30)
            gaze = facing + limit + beyond
            expected = limit + beyond - RETURN * limit
            following = clock.now()
            await pilot.drive(7, yaw=gaze)
            _, status = await pilot.status("active")
            reached = status.telemetry
            body_turn = math.remainder(reached["base_yaw"] - facing, 2 * math.pi)
            view_error = math.remainder(
                reached["base_yaw"] + reached["head_pan"] - gaze, 2 * math.pi
            )
            assert abs(body_turn - expected) < math.radians(4), reached
            assert abs(reached["head_pan"] - RETURN * limit) < math.radians(4), reached
            assert abs(view_error) < math.radians(3), reached
            follow_turns = turns_since(following)
            # Slow and bounded, never reversing, and only once the gaze has settled.
            assert min(follow_turns) >= -1e-6 and max(follow_turns) <= turn_limit + 1e-6
            moved = [
                t - following
                for t, s in pilot.statuses
                if t >= following
                and abs(math.remainder(s.telemetry.get("base_yaw", facing) - facing, 2 * math.pi))
                > math.radians(2)
            ]
            assert moved and moved[0] >= SETTLE, moved[:3]
            overshoot = max(
                math.remainder(s.telemetry["base_yaw"] - facing, 2 * math.pi)
                for t, s in pilot.statuses
                if t >= following and s.state == "active"
            )
            assert overshoot - expected < math.radians(2), overshoot
            # Looking back within range afterwards leaves the body where it is.
            ahead = facing + beyond
            await pilot.drive(2, yaw=ahead)
            _, status = await pilot.status("active")
            settled = status.telemetry
            stayed = math.remainder(settled["base_yaw"] - reached["base_yaw"], 2 * math.pi)
            assert abs(stayed) < 0.03, settled
            looked = math.remainder(settled["base_yaw"] + settled["head_pan"] - ahead, 2 * math.pi)
            assert abs(looked) < math.radians(3), settled
            # A look held just past the soft limit, within the head's reach, turns the body a
            # little, as slowly, and leaves the head the same room.
            facing = settled["base_yaw"]
            near = facing + (SOFT_LIMIT + 1) / 2 * limit
            nearing = clock.now()
            await pilot.drive(5, yaw=near)
            _, status = await pilot.status("active")
            neared = status.telemetry
            near_turn = math.remainder(neared["base_yaw"] - facing, 2 * math.pi)
            assert abs(near_turn - ((SOFT_LIMIT + 1) / 2 - RETURN) * limit) < math.radians(3), (
                neared
            )
            assert max(turns_since(nearing)) <= turn_limit + 1e-6
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
                "standing_turn_began_after_s": moved[0],
                "max_follow_turn_rad_s": max(follow_turns),
                "body_turn_after_looking_back_deg": math.degrees(stayed),
                "glance_body_turn_deg": math.degrees(glanced),
                "near_limit_body_turn_deg": math.degrees(near_turn),
            }
            (OUT / "summary.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2))
            print(
                "PASS: look-relative forward/reverse, bounded body alignment, stationary look, "
                "a settled gaze near or past the pan limit turns the body slowly, only until "
                "the head has room again; a glance turns nothing"
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
