import argparse
import logging
import math
from contextlib import nullcontext
from pathlib import Path

import moderngl
import pygame

from ito.render import GaussianFrame, load_ply, pose
from ito.render.pose import validate_pose

from .window import DesktopState, DesktopWindow


class FileScene:
    def __init__(self, path: Path):
        self.frame = GaussianFrame(load_ply(path), revision=0)

    def poll(self) -> GaussianFrame | None:
        frame, self.frame = self.frame, None
        return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Ito desktop Gaussian-splat pilot view")
    parser.add_argument("scene", type=Path, help="standard 3DGS .ply scene")
    parser.add_argument("--size", nargs=2, type=int, default=(1280, 720), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--position", nargs=3, type=float, default=(0, 0, 0), metavar=("X", "Y", "Z"),
                        help="robot-camera anchor in scene coordinates")
    parser.add_argument("--yaw", type=float, default=0, help="anchor yaw in degrees")
    parser.add_argument("--pitch", type=float, default=0, help="anchor pitch in degrees")
    parser.add_argument("--fov", type=float, default=70, help="vertical field of view in degrees")
    parser.add_argument("--speed", type=float, default=1.5, help="movement in scene units per second")
    parser.add_argument("--fps", type=int, default=90)
    parser.add_argument("--frames", type=int, default=0, help="exit after N frames; 0 runs until closed")
    parser.add_argument("--capture-dir", type=Path, default=Path("captures"))
    parser.add_argument("--metrics", type=Path, help="write frame/input metrics as JSON lines")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.frames < 0:
            raise ValueError("--frames must be nonnegative")
        anchor = validate_pose(pose(args.position, math.radians(args.yaw), math.radians(args.pitch)))
        source = FileScene(args.scene)
        if args.metrics:
            args.metrics.parent.mkdir(parents=True, exist_ok=True)
        with (
            args.metrics.open("w") if args.metrics else nullcontext()
        ) as metrics, DesktopWindow(args.size, fps=args.fps, fov=args.fov, speed=args.speed,
                                   capture_dir=args.capture_dir) as window:
            window.run(source, state=lambda: DesktopState(robot_camera=anchor),
                       max_frames=args.frames, metrics=metrics)
    except (OSError, ValueError, RuntimeError, pygame.error, moderngl.Error) as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
