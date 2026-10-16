import argparse
import asyncio
import json
import logging

from ito.driver.cli import driver_arguments, pairing_command, serve


def main():
    parser = argparse.ArgumentParser(description="Pilot a Pollen Robotics Microduck")
    parser.add_argument(
        "--robot", default="ws://127.0.0.1:8443", help="robot mediad LAN signalling address"
    )
    imu = parser.add_mutually_exclusive_group()
    imu.add_argument(
        "--head-imu",
        action="store_true",
        default=None,
        help="require the head IMU (read by default when the robot has one, as every beta does)",
    )
    imu.add_argument(
        "--no-head-imu", dest="head_imu", action="store_false", help="leave the head IMU alone"
    )
    parser.add_argument("--no-tof", action="store_true", help="leave the head ToF alone")
    driver_arguments(parser)
    parser.set_defaults(port=8081)
    args = parser.parse_args()
    args.prog = parser.prog
    args.audio_source, args.audio_sink = None, "none"
    args.adapter = "drivers.microduck.adapter:MicroduckAdapter"
    args.adapter_args = json.dumps(
        {
            "robot": args.robot,
            "input_timeout": args.input_timeout,
            "head_imu": args.head_imu,
            "tof": not args.no_tof,
        }
    )
    logging.basicConfig(level=logging.INFO)
    try:
        if not pairing_command(args):
            asyncio.run(serve(args))
    except KeyboardInterrupt:
        pass
    except (ValueError, KeyError, ImportError, AttributeError, OSError, RuntimeError) as exc:
        parser.exit(1, f"ito-driver-microduck: {exc}\n")


if __name__ == "__main__":
    main()
