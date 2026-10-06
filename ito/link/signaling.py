import asyncio
from collections.abc import Sequence

import aiohttp
from aiortc import MediaStreamTrack, RTCIceServer, RTCSessionDescription
from pydantic import Field, ValidationError

from ito.link.peer import Peer
from ito.protocol import VERSION, Model


class Offer(Model):
    version: int
    type: str = Field(pattern="^(offer|answer)$")
    sdp: str = Field(min_length=1, max_length=200_000)


async def connect(
    address: str,
    *,
    video_tracks: int = 1,
    receive_audio: bool = True,
    audio: MediaStreamTrack | None = None,
    ice_servers: Sequence[RTCIceServer] = (),
    connect_timeout: float = 15,
) -> Peer:
    """Connect directly to a driver; caller consumes tracks without blocking input."""
    if not 0 <= video_tracks <= 16:
        raise ValueError("video_tracks must be between 0 and 16")
    if audio is not None and audio.kind != "audio":
        raise ValueError("pilot outgoing track must be audio")
    if "://" not in address:
        address = "http://" + address
    peer = Peer("pilot", ice_servers=ice_servers)
    try:
        async with asyncio.timeout(connect_timeout):
            for _ in range(video_tracks):
                peer.pc.addTransceiver("video", direction="recvonly")
            if audio is not None:
                peer.pc.addTrack(audio)
            elif receive_audio:
                peer.pc.addTransceiver("audio", direction="recvonly")
            await peer.pc.setLocalDescription(await peer.pc.createOffer())
            async with aiohttp.ClientSession() as session:
                offer = Offer(version=VERSION, type="offer", sdp=peer.pc.localDescription.sdp)
                async with session.post(
                    address.rstrip("/") + "/offer", json=offer.model_dump()
                ) as response:
                    if response.status != 200:
                        raise ConnectionError(
                            f"driver rejected connection (HTTP {response.status})"
                        )
                    answer = Offer.model_validate_json(await response.read())
                    if answer.version != VERSION or answer.type != "answer":
                        raise ConnectionError("incompatible driver answer")
            await peer.pc.setRemoteDescription(RTCSessionDescription(sdp=answer.sdp, type="answer"))
            await peer.ready.wait()
            await peer.robot_received.wait()
            await peer.clock_ready.wait()
        return peer
    except BaseException:
        await peer.close()
        raise


def parse_offer(data: bytes) -> Offer:
    try:
        offer = Offer.model_validate_json(data)
        if offer.version != VERSION or offer.type != "offer":
            raise ValueError("unsupported offer")
        return offer
    except (ValidationError, ValueError) as exc:
        raise ValueError("invalid or unsupported offer") from exc
