"""Side by side of one tape window: main's MASt3R-SLAM replay (tools/tapes, its composed mp4:
map | true-pose reference / robot camera | overview) next to tofmesh_replay.py's video.

    uv run --extra microduck python tools/tofmesh_versus.py MAIN.mp4 TOFMESH.mp4 OUT.mp4
        [--fps 15] [--seconds 60]

Out is 1920x1080: top left main's SLAM map, top right the ToF mesh (both from the robot's
camera path), bottom left the robot's camera, bottom right the ToF mesh from behind the robot.
Both inputs must start at the same tape second and run at the same frame rate.
"""

import argparse
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np


def quadrant(frame, row, col):
    h, w = frame.shape[0] // 2, frame.shape[1] // 2
    return frame[row * h : (row + 1) * h, col * w : (col + 1) * w]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("main")
    parser.add_argument("tofmesh")
    parser.add_argument("out")
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--seconds", type=float, default=60)
    args = parser.parse_args()
    a, b = cv2.VideoCapture(args.main), cv2.VideoCapture(args.tofmesh)
    w, h = 960, 540
    proc = subprocess.Popen(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{2 * w}x{2 * h}", "-r", str(args.fps), "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", args.out,
        ],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    for _ in range(int(args.seconds * args.fps)):
        ok_a, fa = a.read()
        ok_b, fb = b.read()
        if not (ok_a and ok_b):
            break
        tiles = [
            (quadrant(fa, 0, 0), "main: MASt3R-SLAM splats (robot camera path)"),
            (quadrant(fb, 0, 1), "feat/tof-mesh: ToF mesh (robot's own pose)"),
            (quadrant(fa, 1, 0), "robot camera"),
            (quadrant(fb, 1, 0), "ToF mesh from behind the robot"),
        ]
        out = np.zeros((2 * h, 2 * w, 3), np.uint8)
        for i, (tile, label) in enumerate(tiles):
            tile = cv2.resize(tile, (w, h), interpolation=cv2.INTER_AREA)
            cv2.rectangle(tile, (0, 0), (w, 30), (0, 0, 0), -1)
            cv2.putText(tile, label, (10, 21), 0, 0.65, (0, 255, 255), 1, cv2.LINE_AA)
            r, c = divmod(i, 2)
            out[r * h : (r + 1) * h, c * w : (c + 1) * w] = tile
        proc.stdin.write(out.tobytes())
    proc.stdin.close()
    proc.wait()


if __name__ == "__main__":
    main()
