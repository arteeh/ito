"""Save frames of a video as one PNG, by frame number (0 = first frame):

    uv run tools/frame.py VIDEO FRAME [-n COUNT] [--every K] [--width W] [-o OUT.png]

One frame by default; -n puts COUNT frames from FRAME on (every K-th) side by side,
left to right then top to bottom, wrapping after --columns. -n 2 shows a frame and the
one after it. Videos from tools/session.py hold every recorded frame once, so neighbouring
numbers are neighbouring frames; steps.txt gives the frame each script step started at.
OUT defaults to VIDEO-FRAME.png beside the video.
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
    parser.add_argument("frame", type=int)
    parser.add_argument("-n", "--count", type=int, default=1)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--width", type=int, default=0, help="scale each frame to this width")
    parser.add_argument("-o", "--out", type=Path)
    args = parser.parse_args()
    first, step = args.frame, max(args.every, 1)
    last = first + (args.count - 1) * step
    out = args.out or args.video.with_name(f"{args.video.stem}-{first}.png")
    filters = [f"select='between(n\\,{first}\\,{last})*not(mod(n-{first}\\,{step}))'"]
    if args.width:
        filters.append(f"scale={args.width}:-2")
    if args.count > 1:
        columns = min(args.columns, args.count)
        rows = math.ceil(args.count / columns)
        filters.append(f"tile={columns}x{rows}:padding=4:color=white")
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error"]
    command += ["-i", str(args.video), "-vf", ",".join(filters), "-fps_mode", "passthrough"]
    command += ["-frames:v", "1", "-update", "1", str(out)]
    out.unlink(missing_ok=True)
    subprocess.run(command, check=True)
    if not out.exists():
        raise SystemExit(f"no frame {first} in {args.video}")
    shown = range(first, last + 1, step)
    print(out, f"frames {shown[0]}..{shown[-1]}" if len(shown) > 1 else f"frame {first}")


if __name__ == "__main__":
    main()
