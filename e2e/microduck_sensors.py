"""Drive the real sim/daemons/pilot link and check head IMU and ToF streams.

GST_PLUGIN_PATH=~/.local/share/ito-microduck/gst-plugins-rs/target/debug \\
flock ~/Projects/ito/ceres.lock uv run --extra microduck python e2e/microduck_sensors.py \\
    --microduck ~/.local/share/ito-microduck/microduck \\
    --rl ~/.local/share/ito-microduck/microduck_rl \\
    --policies ~/.local/share/ito-microduck/policies/current
"""

import argparse
import asyncio
import contextlib
import json
import math
import statistics
from pathlib import Path

from drivers.microduck.sim import arguments, port, simulation
from ito import clock
from ito.link.signaling import connect
from ito.protocol import Command, Motion, PilotState, Pose, Status


async def collect(path, method, frames):
    read, write = await asyncio.open_unix_connection(str(path))
    try:
        write.write(
            (
                json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": {}}) + "\n"
            ).encode()
        )
        await write.drain()
        response = json.loads(await read.readline())
        assert "result" in response, response
        while line := await read.readline():
            frames.append((clock.now(), json.loads(line)["params"]))
    finally:
        write.close()
        await write.wait_closed()


async def run(sim, out):
    imu, depth, states, motion, status = [], [], [], [], []
    tasks = [
        asyncio.create_task(collect(path, method, rows))
        for path, method, rows in (
            (sim.tof_socket, "head_imu.stream", imu),
            (sim.tof_socket, "tof.stream", depth),
            (sim.robot_socket, "robot.subscribe", states),
        )
    ]
    peer = await connect(f"127.0.0.1:{sim.port}", code=sim.code, receive_audio=False)
    try:

        async def receive():
            while True:
                message = await peer.messages.get()
                if isinstance(message, Motion):
                    motion.extend(message.samples)
                if isinstance(message, Status):
                    status.append(message)

        tasks.append(asyncio.create_task(receive()))
        start = clock.now()
        sequence = 0
        while clock.now() - start < 12:
            elapsed = clock.now() - start
            yaw = 0.5 * math.sin((elapsed - 4) * 2) if elapsed > 4 else 0.0
            peer.send(
                PilotState(
                    sequence=sequence,
                    capture_time=clock.now(),
                    deadman=True,
                    head=Pose(orientation=(0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2))),
                )
            )
            sequence += 1
            for task in tasks:
                if task.done():
                    task.result()
            await asyncio.sleep(1 / 90)
        assert states and all(r[1].get("frames", {}).get("head_imu") for r in states)
        steady = [f for t, f in imu if start + 1 < t < start + 3]
        turned = [f for t, f in imu if t > start + 5]
        assert len(steady) > 150, len(steady)
        force = statistics.median(math.sqrt(sum(v * v for v in f["accel"])) for f in steady)
        assert 8 < force < 12, force
        dt = [(b["t_ns"] - a["t_ns"]) / 1e9 for a, b in zip(steady, steady[1:], strict=False)]
        assert abs(statistics.median(dt) - 0.01) < 0.0001, dt[:20]
        spin = max(math.sqrt(sum(v * v for v in f["gyro"])) for f in turned)
        assert spin > 0.3, spin
        assert depth and all((f["rows"], f["cols"]) == (8, 8) for _, f in depth)
        valid = [
            d
            for _, f in depth
            for d, s in zip(f["distance_mm"], f["status"], strict=True)
            if s == 5
        ]
        assert valid and max(valid) > 500, valid[:20]
        assert motion and any(s.telemetry.get("motion_gyro") == "head IMU" for s in status)
        report = dict(
            imu_frames=len(imu),
            steady_hz=1 / statistics.median(dt),
            steady_specific_force_ms2=force,
            max_turn_gyro_rads=spin,
            tof_frames=len(depth),
            tof_range_mm=[min(valid), max(valid)],
            mount_present=True,
            ito_motion_samples=len(motion),
        )
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        (out / "sensors.jsonl").write_text(
            "".join(json.dumps(dict(arrival=t, **f)) + "\n" for t, f in imu)
        )
        print(json.dumps(report, indent=2))
    finally:
        peer.send(Command(sequence=1, action="e-stop"))
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await peer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    arguments(parser)
    parser.add_argument("--out", type=Path, default=Path("e2e/out/microduck-sensors"))
    args = parser.parse_args()
    with simulation(
        args.microduck,
        args.rl,
        args.policies,
        logs=args.out,
        pairing_file=args.out / "pairing-code",
        host="127.0.0.1",
        driver_port=port(),
        head_imu=True,
        tof_noise=0,
    ) as sim:
        asyncio.run(run(sim, args.out))


if __name__ == "__main__":
    main()
