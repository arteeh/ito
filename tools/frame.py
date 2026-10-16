"""Save frames of a video as one PNG:

    uv run tools/frame.py VIDEO SECONDS [-n COUNT] [--every K] [--width W] [-o OUT.png]

One frame by default; -n puts COUNT consecutive frames (every K-th) side by side, wrapping
after --columns, to compare them. OUT defaults to VIDEO-SECONDS.png beside the video.
"""

import argparse
import math
import subprocess
from pathlib import Path

import imageio_ffmpeg


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("seconds", type=float)
    parser.add_argument("-n", "--count", type=int, default=1)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--width", type=int, default=0, help="scale each frame to this width")
    parser.add_argument("-o", "--out", type=Path)
    args = parser.parse_args()
    out = args.out or args.video.with_name(f"{args.video.stem}-{args.seconds:g}.png")
    filters = []
    if args.every > 1:
        filters.append(f"select='not(mod(n\\,{args.every}))'")
    if args.width:
        filters.append(f"scale={args.width}:-2")
    if args.count > 1:
        columns = min(args.columns, args.count)
        rows = math.ceil(args.count / columns)
        filters.append(f"tile={columns}x{rows}:padding=4:color=white")
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error"]
    command += ["-ss", str(args.seconds), "-i", str(args.video)]
    if filters:
        command += ["-vf", ",".join(filters)]
    command += ["-frames:v", "1", "-update", "1", str(out)]
    subprocess.run(command, check=True)
    print(out)


if __name__ == "__main__":
    main()
