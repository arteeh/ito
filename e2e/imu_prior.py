"""Measure the real Microduck pose stream against MuJoCo over a walking/head-sweep drive.

Run with PYOPENGL_PLATFORM=osmesa in the Microduck simulator environment,
with --microduck, --rl and --policies.
Truth is independent MuJoCo FK at sampled physical joints, never robotd FK.
"""

import argparse
import asyncio
import contextlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from drivers.microduck import frames
from drivers.microduck.adapter import FRAME_DELAY
from drivers.microduck.sim import port, simulation
from ito import clock
from ito.app.frames import camera_matrix
from ito.link import connect
from ito.protocol import Command, FrameMetadata, PilotState, Pose
from ito.reconstruction.prior import rotation_error, turn


async def capture(sim, args):
    import mujoco
    from mjlab_microduck.sim.body_server import Body, World

    world = World(args.rl / "src/mjlab_microduck/robot/microduck/scene.xml")
    body = Body(world, 0)
    model, data = world.model, world.data
    camera_id = model.camera("head_camera").id
    # Pollen's Camera fixes the MJCF's backward camera by a local X half-turn.
    fixed = np.zeros(4)
    mujoco.mju_mulQuat(fixed, model.cam_quat[camera_id], np.array([0.0, 1.0, 0.0, 0.0]))
    model.cam_quat[camera_id] = fixed
    axes = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
    upright = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    truth, priors, latency, bases = [], [], [], []
    peer = await connect(f"127.0.0.1:{sim.port}", receive_audio=False, code=sim.code)

    async def messages():
        while True:
            message = await peer.messages.get()
            if isinstance(message, FrameMetadata) and message.camera_pose is not None:
                priors.append(
                    (message.capture_time - FRAME_DELAY, camera_matrix(message.camera_pose))
                )
                latency.append(clock.now() - message.capture_time)

    async def video():
        track = await peer.tracks.get()
        while True:
            await track.recv()

    tasks = [asyncio.create_task(messages()), asyncio.create_task(video())]
    start = clock.now()
    sequence = 0
    try:
        peer.send(Command(sequence=0, action="resume"))
        while (now := clock.now()) - start < args.seconds:
            t = now - start
            yaw = 0.9 * math.sin(max(0, t - 8) * 0.5)
            peer.send(
                PilotState(
                    sequence=sequence,
                    capture_time=now,
                    deadman=True,
                    head=Pose(orientation=(0, math.sin(yaw / 2), 0, math.cos(yaw / 2))),
                    axes={"move_y": 0.6 if 10 < t < args.seconds - 4 else 0.0},
                )
            )
            sequence += 1
            before = clock.now()
            sim.body.write('{"op":"read"}\n')
            sim.body.flush()
            physical = json.loads(sim.body.readline())
            stamp = (before + clock.now()) / 2
            data.qpos[body.trunk : body.trunk + 3] = physical["trunk"]
            data.qpos[body.trunk + 3 : body.trunk + 7] = physical["imu"]["quat"]
            data.qpos[body.qpos_adr] = np.array(physical["positions"])[body.to_wire]
            mujoco.mj_forward(model, data)
            pose = np.eye(4)
            pose[:3, :3] = axes @ data.cam_xmat[camera_id].reshape(3, 3) @ upright
            pose[:3, 3] = axes @ data.cam_xpos[camera_id]
            truth.append((stamp, pose))
            bases.append((stamp, *physical["trunk"]))
            for task in tasks:
                if task.done():
                    task.result()
            await asyncio.sleep(0.02)
    finally:
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await peer.close()
    return truth, priors, latency, bases


def report(truth, priors, latency, bases, out):
    from ito.render.pose import quaternion

    np.savez_compressed(
        out / "raw.npz",
        truth_times=[t for t, _ in truth],
        truth=[p for _, p in truth],
        prior_times=[t for t, _ in priors],
        prior=[p for _, p in priors],
        latency=latency,
        bases=bases,
    )
    times = np.array([t for t, _ in truth])
    pt = np.array([t for t, _ in priors])
    pp = np.array([p for _, p in priors])
    valid = (pt > times[0] + 8) & (pt < times[-1] - 0.3)
    pt, pp = pt[valid], pp[valid]
    if len(pp) < 100:
        raise RuntimeError("Fewer than 100 timestamp-matched camera priors; check driver logs")
    history = [(t, p[:3, 3], quaternion(p[:3, :3]), 0) for t, p in truth]

    def matched(lag):
        return np.array(
            [
                camera_matrix(Pose(position=p, orientation=q))
                for p, q, _ in (frames.at(history, t - lag) for t in pt)
            ]
        )

    # Both worlds have gravity, but robotd anchors yaw/XY when the driver starts.
    raw = matched(0)
    yaw = math.atan2(pp[0, 0, 2], pp[0, 2, 2]) - math.atan2(raw[0, 0, 2], raw[0, 2, 2])
    align = np.eye(4)
    align[:3, :3] = turn(yaw)
    align[:3, 3] = pp[0, :3, 3] - align[:3, :3] @ raw[0, :3, 3]
    target = align @ raw
    rotation = np.degrees(
        [rotation_error(a[:3, :3], b[:3, :3]) for a, b in zip(pp, target, strict=True)]
    )
    translation = np.linalg.norm(pp[:, :3, 3] - target[:, :3, 3], axis=1)
    # Camera path includes head motion; report base-independent camera travel explicitly.
    distance = np.linalg.norm(np.diff(target[:, :3, 3], axis=0), axis=1).sum()
    bases = np.asarray(bases)
    # One-second horizontal samples remove gait sway from distance travelled.
    route_times = np.arange(pt[0], pt[-1], 1.0)
    route = np.column_stack([np.interp(route_times, bases[:, 0], bases[:, i]) for i in (1, 2)])
    base_distance = np.linalg.norm(np.diff(route, axis=0), axis=1).sum()
    if base_distance < 0.2:
        raise RuntimeError("Drive travelled less than 20 cm; cannot measure odometry drift")
    lags = np.linspace(-0.2, 0.2, 81)
    errors = [
        np.mean(
            [
                rotation_error(a[:3, :3], b[:3, :3]) ** 2
                for a, b in zip(pp, align @ matched(lag), strict=True)
            ]
        )
        for lag in lags
    ]
    summary = dict(
        samples=len(pp),
        rotation_median_deg=float(np.median(rotation)),
        rotation_p95_deg=float(np.percentile(rotation, 95)),
        rotation_max_deg=float(max(rotation)),
        translation_p95_m=float(np.percentile(translation, 95)),
        camera_path_m=float(distance),
        base_path_m=float(base_distance),
        endpoint_drift_m_per_m=float(translation[-1] / base_distance),
        residual_pose_lag_ms=float(lags[np.argmin(errors)] * 1000),
        configured_frame_delay_ms=FRAME_DELAY * 1000,
        metadata_delivery_p95_ms=float(np.percentile(latency, 95) * 1000),
        alignment="first moving sample yaw and position only; no scale/tilt fit",
        timing="truth socket midpoint; metadata timestamp is driver receive time, not exposure",
    )
    np.savez_compressed(out / "poses.npz", times=pt, prior=pp, truth=target)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("microduck", "rl", "policies"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("e2e/out/imu_prior"))
    parser.add_argument("--seconds", type=float, default=35)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.rl.resolve() / "src"))
    with simulation(
        args.microduck,
        args.rl,
        args.policies,
        logs=args.out,
        pairing_file=args.out / "pairing-code",
        host="127.0.0.1",
        driver_port=int(port()),
        gl="osmesa",
    ) as sim:
        result = asyncio.run(capture(sim, args))
    report(*result, args.out)


if __name__ == "__main__":
    main()
