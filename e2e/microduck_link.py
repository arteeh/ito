"""Real headless pilot: python e2e/microduck_link.py ADDRESS --code CODE.

Sends a headset's 90 Hz of head poses and checks they reach the robot at that rate: the driver
applies them at >= 72 Hz and robot.look answers >= 72 times a second (#31).

Against the bedrock/WSL simulator instead of an address, inside the simulator environment:
python e2e/microduck_link.py --microduck M --rl RL --policies P
"""

import argparse
import asyncio
import contextlib
import json
import math
import statistics
import tempfile
from pathlib import Path

from ito import clock
from ito.link.pairing import PairingError
from ito.link.signaling import connect
from ito.protocol import Command, PilotState, Pose, Status


async def run(args):
    wrong = "111111" if args.code.replace(" ", "") != "111111" else "222222"
    try:
        refused = await connect(args.address, code=wrong, receive_audio=False)
    except PairingError:
        pass
    else:
        await refused.close()
        raise AssertionError("wrong pairing code accepted")
    peer = await connect(args.address, code=args.code, receive_audio=False)
    frames = 0
    statuses = []
    latencies = []
    rates = []  # (pilot_input_hz, look_hz, look_ms) while the head sweeps
    latest = None
    sequence = 0
    moving = sweeping = False
    head_yaw = 0.0
    first_move = None
    response_ms = None

    async def video():
        nonlocal frames
        track = await peer.tracks.get()
        assert track.kind == "video"
        while True:
            frame = await track.recv()
            assert frame.width > 0 and frame.height > 0
            if frames == 0:
                rgb = frame.to_ndarray(format="rgb24")
                assert rgb.std() > 10, "camera has no scene"
            frames += 1

    async def telemetry():
        nonlocal latest, response_ms
        while True:
            message = await peer.messages.get()
            if isinstance(message, Status):
                latest = message
                statuses.append(message.model_dump())
                t = message.telemetry
                if "pilot_input_latency_ms" in t:
                    latencies.append(t["pilot_input_latency_ms"])
                if sweeping and "pilot_input_hz" in t:
                    rates.append((t["pilot_input_hz"], t.get("look_hz", 0), t.get("look_ms", 0)))
                if first_move and response_ms is None and t.get("applied_vx", 0) > 0.02:
                    response_ms = (clock.now() - first_move) * 1000

    async def input_loop():
        nonlocal sequence, first_move
        deadline = clock.now()
        while True:
            now = clock.now()
            yaw = head_yaw + (0.3 * math.sin(2 * now) if sweeping else 0.0)
            if moving and first_move is None:
                first_move = now
            assert peer.send(
                PilotState(
                    sequence=sequence,
                    capture_time=now,
                    deadman=True,
                    axes={"move_y": 0.7 if moving else 0.0},
                    head=Pose(orientation=(0, math.sin(yaw / 2), 0, math.cos(yaw / 2))),
                )
            )
            sequence += 1
            deadline = max(deadline + 1 / args.rate, now)
            await asyncio.sleep(max(0, deadline - clock.now()))

    async def until(predicate):
        try:
            async with asyncio.timeout(20):
                while not predicate():
                    assert peer.connected, "link disconnected"
                    for task in tasks:
                        if task.done():
                            task.result()
                    await asyncio.sleep(0.01)
        except TimeoutError:
            raise AssertionError(f"Timed out: frames={frames}, status={latest}") from None

    tasks = [asyncio.create_task(f()) for f in (video, telemetry, input_loop)]
    try:
        peer.send(Command(sequence=0, action="resume"))
        await until(lambda: frames >= 30 and latest and latest.telemetry.get("healthy"))
        baseline = latest.telemetry["head_yaw"]
        moving, head_yaw = True, 0.45
        await until(
            lambda: (
                latest.telemetry.get("applied_vx", 0) > 0.05
                and abs(latest.telemetry["head_yaw"] - baseline) > 0.15
            )
        )
        await asyncio.sleep(3)
        walking = latest.telemetry.copy()
        moving, sweeping = False, True
        await asyncio.sleep(6)
        sweeping = False
        # The rates are over a one-second window: skip the first second of the sweep.
        swept = rates[len(rates) // 6 :]
        applied_hz = statistics.median(r[0] for r in swept)
        look_hz = statistics.median(r[1] for r in swept)
        peer.send(Command(sequence=1, action="e-stop"))
        await until(
            lambda: (
                latest.state == "e-stopped"
                and all(
                    abs(latest.telemetry.get(f"applied_{axis}", 1)) < 0.001
                    for axis in ("vx", "vy", "vyaw")
                )
            )
        )
        # Keep sending walking input to prove the latch wins.
        await asyncio.sleep(1)
        assert latest.state == "e-stopped"
        assert latest.telemetry["requested_vx"] == 0
        for task in tasks:
            if task.done():
                task.result()
        report = {
            "address": args.address,
            "paired": True,
            "wrong_code_rejected": True,
            "frames": frames,
            "telemetry_messages": len(statuses),
            "pilot_input_latency_ms_median": statistics.median(latencies),
            "pilot_input_latency_ms_max": max(latencies),
            "pilot_send_hz": args.rate,
            "pose_applied_hz_median": applied_hz,
            "pose_applied_hz_p5": sorted(r[0] for r in swept)[len(swept) // 20],
            "robot_look_hz_median": look_hz,
            "robot_look_round_trip_ms_median": statistics.median(r[2] for r in swept),
            "walk_to_telemetry_ms": response_ms,
            "head_yaw_before": baseline,
            "head_yaw_after": walking["head_yaw"],
            "applied_vx": walking["applied_vx"],
            "e_stop": latest.state,
            "ice": peer.pc.iceConnectionState,
            "rtt_ms": peer.clock.rtt * 1000,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if args.rate >= 72:
            assert applied_hz >= 72 and look_hz >= 72, (applied_hz, look_hz)
    finally:
        peer.send(Command(sequence=2, action="e-stop"))
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await peer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("address", nargs="?")
    parser.add_argument("--code")
    parser.add_argument("--rate", type=float, default=90, help="pilot poses per second")
    parser.add_argument("--microduck", type=Path, help="start the simulator from these trees")
    parser.add_argument("--rl", type=Path)
    parser.add_argument("--policies", type=Path)
    parser.add_argument("--output", type=Path, default=Path("e2e/out/microduck-link.json"))
    args = parser.parse_args()
    if not args.microduck:
        if not args.address or not args.code:
            parser.error("give ADDRESS and --code, or the simulator's --microduck/--rl/--policies")
        asyncio.run(run(args))
        return
    from drivers.microduck.sim import port, simulation

    with tempfile.TemporaryDirectory(prefix="ito-duck-link-") as temporary:
        with simulation(
            args.microduck,
            args.rl,
            args.policies,
            logs=args.output.parent,
            pairing_file=Path(temporary) / "pairing-code",
            host="127.0.0.1",
            driver_port=int(port()),
        ) as sim:
            args.address, args.code = f"127.0.0.1:{sim.port}", sim.code
            asyncio.run(run(args))


if __name__ == "__main__":
    main()
