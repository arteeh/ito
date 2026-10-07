import argparse
import asyncio
import json
import logging

from ito.driver.cli import serve


def main():
    parser = argparse.ArgumentParser(description="Pilot a Pollen Robotics Microduck")
    parser.add_argument(
        "--robot", default="ws://127.0.0.1:8443", help="robot mediad LAN signalling address"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--port", type=int, default=8081, help="Ito port (8080 belongs to Microduck's console)"
    )
    parser.add_argument("--input-timeout", type=float, default=0.25)
    parser.add_argument("--ice-server", action="append", default=[])
    parser.add_argument("--turn-username")
    parser.add_argument("--turn-credential")
    args = parser.parse_args()
    args.command_rate = 50
    args.audio_source, args.audio_sink = None, "none"
    args.adapter = "drivers.microduck.adapter:MicroduckAdapter"
    args.adapter_args = json.dumps({"robot": args.robot, "input_timeout": args.input_timeout})
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        pass
    except (ValueError, ImportError, OSError, RuntimeError, TimeoutError) as exc:
        parser.exit(1, f"ito-driver-microduck: {exc}\n")


if __name__ == "__main__":
    main()
