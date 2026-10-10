"""Pollen's real Microduck stack: walk forward, back, sideways and turn, measured in MuJoCo.

Also checks the camera pose robotd's head FK reports against where the pilot looks: its
heading, and its pitch against the pilot's tilt standing and walking.

Runs inside the simulator environment (see drivers/microduck/setup-sim.sh):
PYTHONPATH=. /opt/ito/stack/ito-venv/bin/python e2e/microduck_walking.py \
    --microduck /opt/ito/stack/microduck --rl /opt/ito/stack/microduck_rl \
    --policies /opt/ito/stack/policies/current
"""

import argparse
import asyncio
import json
import math
import tempfile
from pathlib import Path

from drivers.microduck.sim import port, simulation
from ito import clock
from ito.link import connect
from ito.protocol import Command, PilotState, Pose, Status


def head(yaw=0.0, tilt=0.0):
    """Ito's head pose: yaw about up, then tilt (up positive) about the head's right axis."""
    cy, sy, cp, sp = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(tilt / 2), math.sin(tilt / 2)
    return Pose(orientation=(cy * sp, sy * cp, -sy * sp, cy * cp))


def heading(quat):
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class Probe:
    def __init__(self, peer, body):
        self.peer, self.body = peer, body
        self.telemetry = {}
        self.samples = []
        self.sequence = 0
        self.tasks = [asyncio.create_task(self.messages()), asyncio.create_task(self.video())]

    async def messages(self):
        while True:
            message = await self.peer.messages.get()
            if isinstance(message, Status):
                self.telemetry = message.telemetry

    async def video(self):
        track = await self.peer.tracks.get()
        while True:
            await track.recv()

    def truth(self):
        self.body.write('{"op":"read"}\n')
        self.body.flush()
        physical = json.loads(self.body.readline())
        x, y, z = physical["trunk"]
        return x, y, z, heading(physical["imu"]["quat"])

    async def drive(self, label, duration, *, yaw=0.0, tilt=0.0, forward=0.0, right=0.0):
        start = clock.now()
        next_sample = start
        first = len(self.samples)
        while (now := clock.now()) < start + duration:
            for task in self.tasks:
                if task.done():
                    task.result()
            self.peer.send(
                PilotState(
                    sequence=self.sequence,
                    capture_time=now,
                    deadman=True,
                    head=head(yaw, tilt),
                    axes={"move_y": forward, "move_x": right},
                )
            )
            self.sequence += 1
            if now >= next_sample:
                next_sample += 0.1
                x, y, z, body_yaw = self.truth()
                t = self.telemetry
                self.samples.append(
                    {
                        "label": label,
                        "t": now - start,
                        "x": x,
                        "y": y,
                        "z": z,
                        "yaw": body_yaw,
                        "tilt": tilt,
                        **{
                            k: t.get(k)
                            for k in (
                                "policy",
                                "base_yaw",
                                "head_yaw",
                                "head_yaw_target",
                                "head_pitch",
                                "head_pitch_target",
                                "neck_pitch",
                                "neck_pitch_target",
                                "head_roll",
                                "head_roll_target",
                                "imu_quat_0",
                                "imu_quat_1",
                                "imu_quat_2",
                                "imu_quat_3",
                                "camera_yaw",
                                "camera_pitch",
                                "camera_roll",
                                "requested_vx",
                                "requested_vy",
                                "requested_vyaw",
                                "applied_vx",
                                "applied_vy",
                                "applied_vyaw",
                                "fallen",
                            )
                        },
                    }
                )
            await asyncio.sleep(1 / 60)
        return self.samples[first:]


def change(segment):
    first, last = segment[0], segment[-1]
    dx, dy = last["x"] - first["x"], last["y"] - first["y"]
    c, s = math.cos(first["yaw"]), math.sin(first["yaw"])
    return {
        "forward_m": dx * c + dy * s,
        "left_m": -dx * s + dy * c,
        "turn_deg": math.degrees(math.remainder(last["yaw"] - first["yaw"], 2 * math.pi)),
        "seconds": last["t"] - first["t"],
    }


async def run(sim, out):
    peer = await connect(f"127.0.0.1:{sim.port}", receive_audio=False, code=sim.code)
    probe = Probe(peer, sim.body)
    try:
        assert peer.send(Command(sequence=0, action="resume"))
        # Pollen's body boots seated; robotd stands it up with its sit-stand policy.
        async with asyncio.timeout(60):
            while True:
                segment = await probe.drive("rise", 1)
                if min(s["z"] for s in segment) > 0.095 and segment[-1]["policy"] == "walk":
                    break
        report = {}
        checks = {}

        async def leg(label, duration, *, gaze=0.0, tilt=0.0, **keys):
            """Gaze is relative to the body's heading when the leg starts."""
            heading = probe.telemetry["base_yaw"]
            segment = await probe.drive(label, duration, yaw=heading + gaze, tilt=tilt, **keys)
            moved = change(segment)
            last = segment[-1]
            # The view direction robotd's head FK reports, against where the pilot looks.
            moved["camera_vs_gaze_deg"] = math.degrees(
                math.remainder(last["camera_yaw"] - heading - gaze, 2 * math.pi)
            )
            # Camera pitch against the pilot's tilt, once the head has settled: the mean over
            # the leg's second half spans several gait periods, so the sway averages out.
            settled = [s["camera_pitch"] for s in segment[len(segment) // 2 :]]
            moved["camera_vs_tilt_deg"] = math.degrees(sum(settled) / len(settled) - tilt)
            # The worst the camera strays from the pilot's tilt while a stop or start settles.
            early = [s["camera_pitch"] - tilt for s in segment if s["t"] < 1.5]
            moved["camera_vs_tilt_early_worst_deg"] = math.degrees(max(early, key=abs))
            moved["head_yaw_deg"] = math.degrees(last["head_yaw"])
            moved["camera_roll_deg"] = math.degrees(last["camera_roll"])
            moved["fallen"] = any(s["fallen"] for s in segment)
            report[label] = moved
            return moved

        looked = await leg("level_standing", 3)
        checks["standing, the camera is level when the pilot's gaze is"] = (
            abs(looked["camera_vs_tilt_deg"]) < 1
        )
        looked = await leg("tilt_down_standing", 3, tilt=math.radians(-25))
        checks["the camera tilts down with the pilot"] = abs(looked["camera_vs_tilt_deg"]) < 1
        looked = await leg("tilt_up_standing", 3, tilt=math.radians(15))
        checks["the camera tilts up with the pilot"] = abs(looked["camera_vs_tilt_deg"]) < 1
        walked = await leg("forward", 5, forward=1)
        checks["forward walks"] = walked["forward_m"] > 0.4
        checks["walking, the camera is level when the pilot's gaze is"] = (
            abs(walked["camera_vs_tilt_deg"]) < 1
        )
        await leg("settle", 2)
        walked = await leg("backward", 5, forward=-1)
        checks["backward walks"] = walked["forward_m"] < -0.4
        stopped = await leg("settle", 2)
        # The policy drops the neck as the robot stops; the head must follow it without the
        # camera dipping (it sank 13 degrees for a second when one trim served every gait).
        checks["stopping a backward walk keeps the camera level"] = (
            abs(stopped["camera_vs_tilt_early_worst_deg"]) < 7
        )
        looked = await leg("look_60_standing", 3, gaze=math.radians(60))
        checks["a reachable look leaves the body"] = abs(looked["turn_deg"]) < 5
        checks["the camera looks where the pilot looks"] = abs(looked["camera_vs_gaze_deg"]) < 8
        looked = await leg("look_130_standing", 8, gaze=math.radians(130))
        # The head reaches 80 degrees. Once the gaze has settled the body turns slowly, until
        # the head is back at 70% of its reach (56 degrees), not round to face the gaze.
        checks["past the head's reach the body turns until the head has room"] = (
            62 < looked["turn_deg"] < 80 and 45 < looked["head_yaw_deg"] < 66
        )
        checks["and the camera then looks there"] = abs(looked["camera_vs_gaze_deg"]) < 15
        # Looking as far the other way turns the body back past where it started, again only
        # until the head has room (the route that left a body turned on Ceres, #30).
        looked = await leg("look_back_standing", 9, gaze=math.radians(-130))
        checks["looking back the other way turns the body back"] = (
            -80 < looked["turn_deg"] < -62 and -66 < looked["head_yaw_deg"] < -45
        )
        await leg("face", 2)
        walked = await leg("right", 6, right=1)
        checks["D walks to the right"] = walked["left_m"] < -0.3 and walked["turn_deg"] < -45
        await leg("face", 2)
        walked = await leg("left", 6, right=-1)
        checks["A walks to the left"] = walked["left_m"] > 0.3 and walked["turn_deg"] > 45
        await leg("face", 2)
        walked = await leg("walk_looking_left", 6, gaze=1.0, forward=1)
        travel = math.degrees(math.atan2(walked["left_m"], walked["forward_m"]))
        walked["travel_deg"] = travel
        checks["walking follows the gaze"] = 35 < walked["turn_deg"] < 75 and travel > 20
        checks["never fell"] = not any(r["fallen"] for r in report.values())
        await probe.drive("stop", 1)
        report["checks"] = checks
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        failed = [name for name, ok in checks.items() if not ok]
        assert not failed, failed
        print(
            "PASS: Microduck walks forward, back and sideways, turns past the head's reach, "
            "and its camera pose follows the gaze"
        )
    finally:
        (out / "samples.json").write_text(json.dumps(probe.samples) + "\n")
        for task in probe.tasks:
            task.cancel()
        await asyncio.gather(*probe.tasks, return_exceptions=True)
        await peer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microduck", type=Path, required=True)
    parser.add_argument("--rl", type=Path, required=True)
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("e2e/out/microduck-walking"))
    parser.add_argument("--gl", choices=("osmesa", "egl"), help="head camera renderer")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ito-duck-walk-") as temporary:
        with simulation(
            args.microduck,
            args.rl,
            args.policies,
            logs=args.out,
            pairing_file=Path(temporary) / "pairing-code",
            host="127.0.0.1",
            driver_port=int(port()),
            gl=args.gl,
        ) as sim:
            asyncio.run(run(sim, args.out))


if __name__ == "__main__":
    main()
