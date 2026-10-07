import asyncio
from collections.abc import Sequence

import aiohttp
from aiortc import MediaStreamTrack, RTCIceServer, RTCSessionDescription
from pydantic import Field, ValidationError

from ito.link import pairing
from ito.link.audio import opus
from ito.link.peer import Peer
from ito.protocol import VERSION, Credential, Model, Token


class Offer(Model):
    version: int
    type: str = Field(pattern="^(offer|answer)$")
    sdp: str = Field(min_length=1, max_length=200_000)
    nonce: str | None = Field(default=None, pattern="^[0-9a-f]{32}$")
    proof: str | None = Field(default=None, pattern="^[0-9a-f]{64}$")
    pilot: Token | None = None  # Proves this pilot's secret instead of the pairing code.


class Challenge(Model):
    nonce: str = Field(pattern="^[0-9a-f]{32}$")


async def connect(
    address: str,
    *,
    video_tracks: int = 1,
    receive_audio: bool = True,
    audio_io=None,
    audio: MediaStreamTrack | None = None,
    ice_servers: Sequence[RTCIceServer] = (),
    connect_timeout: float = 15,
    code: str | None = None,
    credential: Credential | None = None,
) -> Peer:
    """Connect directly to a driver; caller consumes tracks without blocking input.

    A credential from an earlier pairing replaces the code. After a code, the returned
    peer holds the driver's new credential: store it, then send Paired so the driver
    retires the code. Raises PairingError when the driver refuses the code or credential
    (or neither was given).
    """
    if not 0 <= video_tracks <= 16:
        raise ValueError("video_tracks must be between 0 and 16")
    if audio is not None and audio.kind != "audio":
        raise ValueError("pilot outgoing track must be audio")
    if "://" not in address:
        address = "http://" + address
    peer = Peer("pilot", ice_servers=ice_servers, audio=audio_io)
    try:
        async with asyncio.timeout(connect_timeout):
            for _ in range(video_tracks):
                peer.pc.addTransceiver("video", direction="recvonly")
            if audio_io is not None:
                audio = audio_io.track  # Devices open once the robot says what it has.
            if audio is not None:
                peer.pc.addTrack(audio)
            elif receive_audio:
                peer.pc.addTransceiver("audio", direction="recvonly")
            opus(peer.pc)
            if not ice_servers:
                from ito.link.routing import direct_interface

                await direct_interface(peer.pc, address)
            await peer.pc.setLocalDescription(await peer.pc.createOffer())
            sdp = peer.pc.localDescription.sdp
            base = address.rstrip("/")
            async with aiohttp.ClientSession() as session:
                nonce = proof = None
                key = credential.secret if credential else code
                if key is not None:
                    async with session.post(base + "/pairing") as response:
                        if response.status != 200:
                            raise ConnectionError(
                                f"driver refused pairing (HTTP {response.status})"
                            )
                        nonce = Challenge.model_validate_json(await response.read()).nonce
                    proof = pairing.proof(key, nonce, "offer", sdp)
                offer = Offer(
                    version=VERSION,
                    type="offer",
                    sdp=sdp,
                    nonce=nonce,
                    proof=proof,
                    pilot=credential.pilot if credential else None,
                )
                async with session.post(
                    base + "/offer", json=offer.model_dump(exclude_none=True)
                ) as response:
                    if response.status in (401, 403):
                        raise pairing.PairingError(await reason(response))
                    if response.status != 200:
                        raise ConnectionError(
                            await reason(response)
                            or f"driver rejected connection (HTTP {response.status})"
                        )
                    answer = Offer.model_validate_json(await response.read())
                    if answer.version != VERSION or answer.type != "answer":
                        raise ConnectionError("incompatible driver answer")
            if key is None or not pairing.valid(key, nonce, "answer", answer.sdp, answer.proof):
                raise pairing.PairingError("The robot could not prove it paired with this pilot")
            await peer.pc.setRemoteDescription(RTCSessionDescription(sdp=answer.sdp, type="answer"))
            await peer.ready.wait()
            await peer.robot_received.wait()
            await peer.clock_ready.wait()
            if credential is None:
                await peer.credential_received.wait()
        return peer
    except BaseException:
        await peer.close()
        raise


async def reason(response) -> str:
    """The driver's plain-text refusal, short enough to show the pilot."""
    if response.content_type != "text/plain":
        return ""
    return (await response.text())[:200].strip()


def parse_offer(data: bytes) -> Offer:
    try:
        offer = Offer.model_validate_json(data)
        if offer.version != VERSION or offer.type != "offer":
            raise ValueError("unsupported offer")
        return offer
    except (ValidationError, ValueError) as exc:
        raise ValueError("invalid or unsupported offer") from exc
