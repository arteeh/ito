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

from scipy.spatial.transform import Rotation

from drivers.microduck.sim import arguments, port, simulation
from ito import clock
from ito.link.signaling import connect
from ito.protocol import Command, Motion, PilotState, Pose, SensorObservations, Status


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


async def run(sim, out, board):
    imu, depth, states, motion, status, observations = [], [], [], [], [], []
    tasks = [
        asyncio.create_task(collect(path, method, rows))
        for path, method, rows in (
            (sim.robot_socket if board == "beta" else sim.tof_socket, "head_imu.stream", imu),
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
                if isinstance(message, SensorObservations):
                    observations.append(message)
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
        expected_hz = 60 if board == "beta" else 100
        assert len(steady) > expected_hz * 1.3, len(steady)
        force = statistics.median(math.sqrt(sum(v * v for v in f["accel"])) for f in steady)
        assert 8 < force < 12, force
        dt = [(b["t_ns"] - a["t_ns"]) / 1e9 for a, b in zip(steady, steady[1:], strict=False)]
        assert abs(statistics.median(dt) - 1 / expected_hz) < 0.0001, dt[:20]
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
        assert any(s.imu for s in observations), "Ito received no raw IMU observations"
        assert any(s.ranges for s in observations), "Ito received no ToF observations"
        camera = peer.description.cameras[0]
        observed_steady = [
            sample
            for batch in observations
            for sample in batch.imu
            if start + 1 < sample.time < start + 3
        ]
        assert observed_steady, "No source-timed steady IMU samples reached Ito"
        camera_force = [
            Rotation.from_quat(sample.mount.orientation).apply(sample.specific_force)
            for sample in observed_steady
        ]
        up_force = statistics.median(force[1] for force in camera_force)
        assert up_force > 8, camera_force[:4]
        # FIFO delivery must retain its source stamps rather than stamping all four at wake.
        arrivals = [t for t, _ in imu if start + 1 < t < start + 3]
        burst_sizes = []
        for arrival in arrivals:
            if burst_sizes and arrival - burst_sizes[-1][0] < 0.01:
                burst_sizes[-1][1] += 1
            else:
                burst_sizes.append([arrival, 1])
        if board == "beta":
            assert 3 <= statistics.median(size for _, size in burst_sizes) <= 5, burst_sizes
        if board == "beta":
            assert (camera.intrinsics.width, camera.intrinsics.height) == (1920, 1080)
        report = dict(
            board=board,
            camera=camera.model_dump(),
            raw_imu_received=sum(len(s.imu) for s in observations),
            ranges_received=sum(len(s.ranges) for s in observations),
            imu_frames=len(imu),
            steady_hz=1 / statistics.median(dt),
            steady_specific_force_ms2=force,
            camera_up_force_ms2=up_force,
            median_burst_samples=statistics.median(size for _, size in burst_sizes),
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


async def camera_only(sim, out):
    peer = await connect(f"127.0.0.1:{sim.port}", code=sim.code, receive_audio=False)
    try:
        assert peer.description.motion_camera is None
        assert peer.description.motion_noise is None
        track = await asyncio.wait_for(peer.tracks.get(), timeout=20)
        assert track.kind == "video"
        frames = 0
        began = clock.now()
        while clock.now() - began < 8:
            frame = await asyncio.wait_for(track.recv(), timeout=10)
            image = frame.to_ndarray(format="rgb24")
            assert image.std() > 10
            assert (frame.width, frame.height) == (1920, 1080)
            frames += 1
            while not peer.messages.empty():
                message = peer.messages.get_nowait()
                assert not isinstance(message, (Motion, SensorObservations)), message
        assert frames > 5, frames
        report = dict(mode="camera-only", frames=frames, motion_prior=False, sensor_batches=0)
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        peer.send(Command(sequence=1, action="e-stop"))
        await peer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    arguments(parser)
    parser.add_argument("--board", choices=("beta", "zero3"), default="beta")
    parser.add_argument("--out", type=Path, default=Path("e2e/out/microduck-sensors"))
    parser.add_argument("--camera-only", action="store_true")
    args = parser.parse_args()
    with simulation(
        args.microduck,
        args.rl,
        args.policies,
        logs=args.out,
        pairing_file=args.out / "pairing-code",
        host="127.0.0.1",
        driver_port=port(),
        board=args.board,
        head_imu=not args.camera_only,
        motion_prior=not args.camera_only,
        tof=not args.camera_only,
        tof_noise=0,
    ) as sim:
        asyncio.run(
            camera_only(sim, args.out) if args.camera_only else run(sim, args.out, args.board)
        )


if __name__ == "__main__":
    main()
