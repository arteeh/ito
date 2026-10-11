"""Measure the real Microduck simulator -> mediad -> WebRTC -> Ito decoder.

python e2e/sharp_camera.py --microduck M --rl RL --policies P --height 1080
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
from aiortc.codecs.h264 import H264Decoder
from aiortc.codecs.vpx import Vp8Decoder

from drivers.microduck.sim import arguments, port, simulation
from ito.link.signaling import connect


async def receive(address, code, height, seconds):
    costs, converts = [], []
    originals = {}
    for cls in (H264Decoder, Vp8Decoder):
        original = cls.decode
        originals[cls] = original

        def timed(self, frame, original=original):
            start = time.perf_counter()
            result = original(self, frame)
            costs.append((time.perf_counter() - start) * 1000)
            return result

        cls.decode = timed
    peer = await connect(address, code=code, receive_audio=False)
    try:
        track = await asyncio.wait_for(peer.tracks.get(), 30)
        first = None
        frames = 0
        while first is None or time.perf_counter() - first < seconds:
            frame = await asyncio.wait_for(track.recv(), 30)
            now = time.perf_counter()
            if first is None:
                first = now
            else:
                frames += 1
            rgb = frame.to_ndarray(format="rgb24")
            converts.append((time.perf_counter() - now) * 1000)
            assert rgb.shape == (height * 16 // 9, height, 3), rgb.shape
        k = peer.description.cameras[0].intrinsics
        assert (k.height, k.width) == rgb.shape[:2]
        return dict(
            height=height,
            width=k.height,
            received_fps=frames / (time.perf_counter() - first),
            decode_p50_ms=float(np.median(costs)),
            decode_p95_ms=float(np.percentile(costs, 95)),
            rgb_p50_ms=float(np.median(converts)),
            rgb_p95_ms=float(np.percentile(converts, 95)),
        )
    finally:
        await peer.close()
        for cls, original in originals.items():
            cls.decode = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    arguments(parser)
    parser.add_argument("--height", type=int, choices=(360, 720, 1080), default=1080)
    parser.add_argument("--seconds", type=float, default=20)
    args = parser.parse_args()
    out = Path(f"e2e/out/sharp/camera{args.height}")
    out.mkdir(parents=True, exist_ok=True)
    with simulation(
        args.microduck,
        args.rl,
        args.policies,
        logs=out,
        pairing_file=out / "pairing",
        host="127.0.0.1",
        driver_port=port(),
        gl="osmesa",
        camera_height=args.height,
    ) as sim:
        result = asyncio.run(receive(f"127.0.0.1:{sim.port}", sim.code, args.height, args.seconds))
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
