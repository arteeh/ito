"""A live link diagnostic: consume media and report clock/driver status."""

import argparse
import asyncio
import json

import aiohttp
from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError

from ito.link import connect
from ito.protocol import Status


async def monitor(address: str, cameras: int) -> None:
    async with await connect(address, video_tracks=cameras) as peer:
        print(peer.description.model_dump_json(), flush=True)
        counts = {"video": 0, "audio": 0}

        async def consume(track: MediaStreamTrack):
            try:
                while True:
                    await track.recv()
                    counts[track.kind] += 1
            except MediaStreamError:
                return

        async def tracks():
            async with asyncio.TaskGroup() as group:
                while True:
                    group.create_task(consume(await peer.tracks.get()))

        async def report():
            latest_status = None
            while True:
                await asyncio.sleep(1)
                while not peer.messages.empty():
                    message = peer.messages.get_nowait()
                    if isinstance(message, Status):
                        latest_status = message.model_dump()
                print(
                    json.dumps(
                        {
                            "frames": counts,
                            "clock_offset": peer.clock.offset,
                            "rtt": peer.clock.rtt,
                            "rejected": peer.rejected_messages,
                            "driver": latest_status,
                        }
                    ),
                    flush=True,
                )

        async with asyncio.TaskGroup() as group:
            media = group.create_task(tracks())
            metrics = group.create_task(report())
            await peer.closed.wait()
            media.cancel()
            metrics.cancel()
        raise ConnectionError("driver disconnected")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a live Ito driver link")
    parser.add_argument("address")
    parser.add_argument("--cameras", type=int, default=1)
    args = parser.parse_args()
    try:
        asyncio.run(monitor(args.address, args.cameras))
    except KeyboardInterrupt:
        pass
    except (ConnectionError, TimeoutError, OSError, ValueError, aiohttp.ClientError) as exc:
        parser.exit(1, f"ito-link: {exc}\n")


if __name__ == "__main__":
    main()
