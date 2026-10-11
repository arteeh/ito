"""Run #32's scene_truth benchmark at the full head-camera resolution.

Copy e2e/scene_truth.py from fix/scene-truth beside this script, then:
  uv run --with scipy python e2e/sharp_truth.py --height 1080 --label sharp --budget 400000
The benchmark stays unmodified; only camera resolution and calibration change.
"""

import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path

if "--serve" in sys.argv:
    os.environ["MUJOCO_GL"] = "osmesa" if sys.platform == "linux" else "glfw"
    if sys.platform == "linux":
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--height", type=int, choices=(360, 720, 1080), default=1080)
    parser.add_argument("--serve", action="store_true")
    args, remaining = parser.parse_known_args()
    if not Path(__file__).with_name("scene_truth.py").is_file():
        raise SystemExit("Copy e2e/scene_truth.py from fix/scene-truth beside this script")
    import scene_truth as truth

    truth.SIZE = (args.height * 16 // 9, args.height)
    if args.serve:
        from drivers.mujoco import adapter

        original = adapter.MujocoAdapter

        def camera_adapter(width, height, gl):
            # The live driver's size limit is independent of this offline camera.
            room = original(width=320, height=180, gl=gl)
            room.model.vis.global_.offwidth = max(room.model.vis.global_.offwidth, width)
            room.model.vis.global_.offheight = max(room.model.vis.global_.offheight, height)
            camera = room.description.cameras[0]
            k = camera.intrinsics
            scale = height / 180
            k = k.model_copy(
                update=dict(
                    width=width,
                    height=height,
                    fx=k.fx * scale,
                    fy=k.fy * scale,
                    cx=(k.cx + 0.5) * scale - 0.5,
                    cy=(k.cy + 0.5) * scale - 0.5,
                )
            )
            room._description = room.description.model_copy(
                update=dict(cameras=(camera.model_copy(update=dict(intrinsics=k)),))
            )
            return room

        adapter.MujocoAdapter = camera_adapter
        truth.serve()
    else:

        class Room(truth.Room):
            def __init__(self):
                self.process = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--serve",
                        "--height",
                        str(args.height),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                )
                self.intrinsics, self.limits = pickle.load(self.process.stdout)

        truth.Room = Room
        sys.argv = [sys.argv[0], *remaining]
        truth.main()


if __name__ == "__main__":
    main()
