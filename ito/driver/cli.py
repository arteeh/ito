import argparse
import asyncio
import importlib
import json
import logging
import signal
from pathlib import Path

from aiortc import RTCIceServer

from ito.driver import Adapter, Driver, pairing
from ito.link.audio import arguments
from ito.link.pairing import display


def driver_arguments(parser) -> None:
    """Network, safety and pairing options every ito-driver-<robot> shares."""
    parser.add_argument("--host", default="0.0.0.0", help="address to listen on (default: all)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--input-timeout", type=float, default=0.25)
    parser.add_argument("--command-rate", type=float, default=90)
    parser.add_argument("--ice-server", action="append", default=[], help="STUN/TURN URL")
    parser.add_argument("--turn-username")
    parser.add_argument("--turn-credential")
    parser.add_argument("--pairing-file", type=Path, default=pairing.default_path())
    parser.add_argument("--hide-code", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--show-code", action="store_true", help="print the pairing code, exit")
    parser.add_argument(
        "--rotate-code",
        action="store_true",
        help="make a new single-use pairing code and exit; the pilot must enter it again",
    )


def shown(path) -> str:
    code, _ = pairing.ensure(path)
    if pairing.armed(path):
        return f"Pairing code: {display(code)} (the pilot enters it once)"
    return "Pairing code used by a paired pilot; --rotate-code makes a new one"


def pairing_command(args) -> bool:
    """Handle --show-code/--rotate-code; True when the program should exit."""
    if args.rotate_code:
        code = pairing.rotate(args.pairing_file)
        print(f"New pairing code: {display(code)}", flush=True)
    elif args.show_code:
        print(shown(args.pairing_file), flush=True)
    return args.rotate_code or args.show_code


async def serve(args) -> None:
    module, factory_name = args.adapter.rsplit(":", 1)
    factory = getattr(importlib.import_module(module), factory_name)
    options = json.loads(args.adapter_args)
    if not isinstance(options, dict):
        raise ValueError("--adapter-args must be a JSON object")
    adapter = factory(**options)
    if not isinstance(adapter, Adapter):
        raise ValueError("adapter factory must return an ito.driver.Adapter")
    ice_servers = [
        RTCIceServer(urls=url, username=args.turn_username, credential=args.turn_credential)
        for url in args.ice_server
    ]
    driver = Driver(
        adapter,
        audio_source=args.audio_source,
        audio_sink=args.audio_sink,
        input_timeout=args.input_timeout,
        command_rate=args.command_rate,
        ice_servers=ice_servers,
        pairing_file=args.pairing_file,
    )
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stopped.set)
        except NotImplementedError:
            pass
    code = shown(args.pairing_file)
    try:
        address = await driver.start(args.host, args.port)
        print(f"Ito driver listening at {address}", flush=True)
        if not args.hide_code:
            print(code, flush=True)
        failed = asyncio.create_task(driver.failed.wait())
        signalled = asyncio.create_task(stopped.wait())
        await asyncio.wait((failed, signalled), return_when=asyncio.FIRST_COMPLETED)
        failed.cancel()
        signalled.cancel()
        if driver.failed.is_set():
            raise RuntimeError("the safety watchdog stopped; the robot is held neutral")
    finally:
        await driver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve one robot through an Ito adapter")
    parser.add_argument("adapter", nargs="?", help="Python module:factory returning an Adapter")
    parser.add_argument("--adapter-args", default="{}", help="JSON object passed to the factory")
    driver_arguments(parser)
    arguments(parser)
    args = parser.parse_args()
    args.prog = parser.prog
    if not (args.adapter or args.show_code or args.rotate_code):
        parser.error("the adapter argument is required")
    logging.basicConfig(level=logging.INFO)
    try:
        if not pairing_command(args):
            asyncio.run(serve(args))
    except KeyboardInterrupt:
        pass
    except (ValueError, ImportError, AttributeError, OSError, RuntimeError) as exc:
        parser.exit(1, f"ito-driver: {exc}\n")
