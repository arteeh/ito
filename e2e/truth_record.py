"""Record source truth while piloting Pollen's stack; run in the sim distro's Ito venv.

PYTHONPATH=. /opt/ito/stack/ito-venv/bin/python e2e/truth_record.py --route short --out PATH
The checkout is used directly: no installed distro files are patched.
"""

import argparse
import asyncio
import json
import math
import os
import time
from pathlib import Path

from microduck_walking import Probe
from PIL import Image

from drivers.microduck.sim import port, simulation
from ito.link import connect
from ito.protocol import Command

ROUTES = {
    "scan": [
        (0, 0, "", 3),
        (80, 30, "", 3),
        (-80, -35, "", 3),
        (180, 0, "", 4),
        (270, 30, "", 3),
        (360, -35, "", 3),
        (360, 0, "", 3),
    ],
    "short": [
        (0, 0, "", 2),
        (0, 0, "w", 3),
        (0, 0, "", 1),
        (90, 0, "", 2),
        (-90, 0, "", 2),
        (0, 0, "", 1),
        (0, 0, "s", 3),
        (0, 0, "", 2),
    ],
    "tour": [
        (0, 0, "", 4),
        (90, 0, "", 4),
        (180, 0, "", 4),
        (90, 0, "", 3),
        (0, 0, "", 3),
        (-90, 0, "", 4),
        (0, 0, "", 3),
        (0, 0, "w", 6),
        (0, 20, "", 2),
        (0, -25, "", 2),
        (0, 0, "", 2),
        (60, 0, "w", 5),
        (60, 0, "s", 4),
        (-60, 0, "w", 6),
        (-150, 0, "", 4),
        (-150, 0, "w", 5),
        (0, 0, "", 4),
    ],
    "smoke": [(0, 0, "", 2), (30, 10, "", 2)],
}


class RecordingProbe(Probe):
    async def video(self):
        track = await self.peer.tracks.get()
        count = 0
        while True:
            frame = await track.recv()
            stamp = time.monotonic_ns()
            # Small lossless thumbnails identify source frames after transport compression.
            image = Image.fromarray(frame.to_ndarray(format="rgb24"))
            image.thumbnail((64, 64))
            name = f"live/{count:06d}.png"
            image.save(self.out / name)
            with (self.out / "live.jsonl").open("a") as index:
                index.write(
                    json.dumps(
                        dict(
                            rgb=name,
                            monotonic_ns=stamp,
                            pts=frame.pts,
                            time_base=str(frame.time_base),
                        )
                    )
                    + "\n"
                )
            count += 1


async def run(sim, out, route):
    peer = await connect(f"127.0.0.1:{sim.port}", receive_audio=False, code=sim.code)
    (out / "live").mkdir()
    probe = RecordingProbe(peer, sim.body)
    probe.out = out
    try:
        assert peer.send(Command(sequence=0, action="resume"))
        async with asyncio.timeout(75):
            while True:
                segment = await probe.drive("rise", 1)
                if min(s["z"] for s in segment) > 0.095 and segment[-1]["policy"] == "walk":
                    break
        yaw, tilt = 0.0, 0.0
        legs = []
        for index, (degrees, pitch, keys, duration) in enumerate(ROUTES[route]):
            target = math.radians(degrees)
            dy = math.remainder(target - yaw, math.tau)
            dp = math.radians(pitch) - tilt
            steps = max(1, math.ceil(max(abs(dy), abs(dp)) / math.radians(45) * 20))
            start = time.monotonic_ns()
            for step in range(1, steps + 1):
                await probe.drive(
                    f"turn-{index}",
                    0.05,
                    yaw=yaw + dy * step / steps,
                    tilt=tilt + dp * step / steps,
                    forward=1.0 if keys == "w" else -1.0 if keys == "s" else 0,
                )
            yaw += dy
            tilt += dp
            await probe.drive(
                f"hold-{index}",
                duration,
                yaw=yaw,
                tilt=tilt,
                forward=1.0 if keys == "w" else -1.0 if keys == "s" else 0,
            )
            legs.append(
                dict(
                    index=index,
                    command=ROUTES[route][index],
                    start_ns=start,
                    end_ns=time.monotonic_ns(),
                )
            )
            print(f"{route}: leg {index + 1}/{len(ROUTES[route])}", flush=True)
        (out / "route.json").write_text(json.dumps(dict(route=route, legs=legs), indent=2))
    finally:
        peer.send(Command(sequence=1, action="stop"))
        for task in probe.tasks:
            task.cancel()
        await asyncio.gather(*probe.tasks, return_exceptions=True)
        await peer.close()
        (out / "telemetry.json").write_text(json.dumps(probe.samples))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=ROUTES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    os.environ["ITO_MICRODUCK_RECORD"] = str(out)
    # Durable state stays in the workspace. simulation owns and removes its Unix socket tempdir.
    logs = out.parent / (out.name + "-logs")
    logs.mkdir(exist_ok=False)
    with simulation(
        Path("/opt/ito/stack/microduck"),
        Path("/opt/ito/stack/microduck_rl"),
        Path("/opt/ito/stack/policies/current"),
        logs=logs,
        pairing_file=logs / "pairing",
        driver_port=int(port()),
        viewer=True,
        gl="glfw",
    ) as sim:
        asyncio.run(run(sim, out, args.route))


if __name__ == "__main__":
    main()
