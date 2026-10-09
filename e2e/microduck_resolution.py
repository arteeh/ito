"""Check a running simulated Microduck delivers the camera size and lens it was started with.

python e2e/microduck_resolution.py ADDRESS --code CODE --height 720

Start the simulator with `--camera-height`. Connects as a pilot, reads the camera's described
intrinsics and the frames themselves for a few seconds, and checks the frame size, the 62 degree
lens across the long side, and that frames keep arriving.
"""

import argparse
import asyncio
import math
import time

from ito.link.signaling import connect


async def run(args):
    peer = await connect(args.address, code=args.code, receive_audio=False)
    try:
        track = await asyncio.wait_for(peer.tracks.get(), 30)
        frames, first, size = 0, None, None
        deadline = None
        while deadline is None or time.monotonic() < deadline:
            frame = await asyncio.wait_for(track.recv(), 30)
            if first is None:
                first = time.monotonic()
                deadline = first + args.seconds
                size = (frame.width, frame.height)
            else:
                frames += 1
        rate = frames / (time.monotonic() - first)
        (camera,) = peer.description.cameras
        k = camera.intrinsics
        hfov = math.degrees(2 * math.atan(k.width / 2 / k.fx))
        vfov = math.degrees(2 * math.atan(k.height / 2 / k.fy))
        print(f"frame {size[0]}x{size[1]}, intrinsics {k.width}x{k.height}, {rate:.1f} fps")
        print(f"fov {hfov:.1f} x {vfov:.1f} deg (width x height)")
        # The camera is mounted a quarter turn off: delivered frames are tall and thin.
        wide, tall = args.height * 16 // 9, args.height
        assert size == (tall, wide), f"frame {size}, expected {(tall, wide)}"
        assert (k.width, k.height) == size, "intrinsics do not describe the frames"
        assert abs(vfov - 62.0) < 0.2, f"long side spans {vfov:.1f} degrees, not the lens's 62"
        assert rate > 3, f"only {rate:.1f} frames per second"
        print("PASS")
    finally:
        await peer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("address")
    parser.add_argument("--code", required=True)
    parser.add_argument("--height", type=int, choices=(360, 720, 1080), required=True)
    parser.add_argument("--seconds", type=float, default=6)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
