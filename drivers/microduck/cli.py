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
    driver_arguments(parser)
    # Microduck's console owns port 8080. robotd runs its control loop at 50 Hz, but the
    # adapter sends only the newest input, so passing every pilot sample on keeps the one each
    # tick reads fresh.
    parser.set_defaults(port=8081)
    args = parser.parse_args()
    args.prog = parser.prog
    args.audio_source, args.audio_sink = None, "none"
    args.adapter = "drivers.microduck.adapter:MicroduckAdapter"
    args.adapter_args = json.dumps({"robot": args.robot, "input_timeout": args.input_timeout})
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
