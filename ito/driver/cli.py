import argparse
import asyncio
import importlib
import json
import logging
import signal

from aiortc import RTCIceServer

from ito.driver import Adapter, Driver
from ito.link.audio import arguments


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
    )
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stopped.set)
        except NotImplementedError:
            pass
    try:
        address = await driver.start(args.host, args.port)
        print(f"Ito driver listening at {address}", flush=True)
        await stopped.wait()
    finally:
        await driver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve one robot through an Ito adapter")
    parser.add_argument("adapter", help="Python module:factory returning an Adapter")
    parser.add_argument("--adapter-args", default="{}", help="JSON object passed to the factory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--input-timeout", type=float, default=0.25)
    parser.add_argument("--command-rate", type=float, default=90)
    parser.add_argument("--ice-server", action="append", default=[], help="STUN/TURN URL")
    parser.add_argument("--turn-username")
    parser.add_argument("--turn-credential")
    arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        pass
    except (ValueError, ImportError, AttributeError, OSError) as exc:
        parser.exit(1, f"ito-driver: {exc}\n")
