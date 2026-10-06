import argparse
import asyncio
import json
import logging
import os

from ito.driver.cli import serve

from .adapter import ROOM


def main():
    parser = argparse.ArgumentParser(description="Pilot an MJCF robot over Ito WebRTC")
    parser.add_argument(
        "model", nargs="?", default=str(ROOM), help="MJCF file (default: furnished room)"
    )
    parser.add_argument(
        "--gl", choices=("osmesa", "egl", "glfw"), default=os.environ.get("MUJOCO_GL")
    )
    parser.add_argument("--camera", default="head")
    parser.add_argument("--rgb-only", action="store_true", help="send RGB without depth or pose")
    parser.add_argument("--base", default="base", help="body anchoring the Ito world at startup")
    parser.add_argument("--pan", default="head_pan", help="position servo, positive turns left")
    parser.add_argument("--tilt", default="head_tilt", help="position servo, positive looks up")
    parser.add_argument("--left", default="left_drive", help="left wheel velocity servo")
    parser.add_argument("--right", default="right_drive", help="right wheel velocity servo")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--wheel-radius", type=float, default=0.14)
    parser.add_argument("--axle-width", type=float, default=0.52)
    parser.add_argument("--speed", type=float, default=0.7, help="maximum forward speed in m/s")
    parser.add_argument("--turn-speed", type=float, default=1.2, help="maximum yaw speed in rad/s")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--input-timeout", type=float, default=0.25)
    parser.add_argument("--command-rate", type=float, default=90)
    parser.add_argument("--ice-server", action="append", default=[], help="STUN/TURN URL")
    parser.add_argument("--turn-username")
    parser.add_argument("--turn-credential")
    args = parser.parse_args()
    args.adapter = "drivers.mujoco.adapter:MujocoAdapter"
    args.adapter_args = json.dumps(
        {
            name: getattr(args, name)
            for name in (
                "model",
                "rgb_only",
                "gl",
                "camera",
                "base",
                "pan",
                "tilt",
                "left",
                "right",
                "width",
                "height",
                "fps",
                "wheel_radius",
                "axle_width",
                "speed",
                "turn_speed",
                "input_timeout",
            )
        }
    )
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        pass
    except (ValueError, KeyError, ImportError, AttributeError, OSError, RuntimeError) as exc:
        parser.exit(1, f"ito-driver-mujoco: {exc}\n")


if __name__ == "__main__":
    main()
