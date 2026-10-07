"""Microduck mediad's LAN signalling and JSON-RPC contract (upstream API 40)."""

import asyncio
import contextlib
import json

import aiohttp
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.sdp import candidate_from_sdp


class Remote:
    def __init__(self, address, notification):
        self.address = address
        self.notification = notification
        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        self.channel = None
        self.video = None
        self.ready = asyncio.Event()
        self.pending = {}
        self.sequence = 0
        self.error = None
        self.task = None
        self.http = None

        @self.pc.on("track")
        def track(value):
            if value.kind == "video" and self.video is None:
                self.video = value
            else:
                value.stop()

        @self.pc.on("datachannel")
        def channel(value):
            if value.label != "control" or self.channel is not None:
                value.close()
                return
            self.channel = value
            value.on("message", self._message)
            value.on("close", lambda: self.fail("Microduck control channel closed"))
            value.on("open", self.ready.set)
            if value.readyState == "open":
                self.ready.set()

        @self.pc.on("connectionstatechange")
        def connection():
            if self.pc.connectionState in {"failed", "disconnected"}:
                self.fail("Microduck WebRTC connection lost")

    def fail(self, reason):
        self.error = RuntimeError(str(reason))
        self.ready.set()
        for future in self.pending.values():
            if not future.done():
                future.set_exception(self.error)

    def check(self):
        if self.error:
            raise self.error
        if not self.channel or self.channel.readyState != "open":
            raise RuntimeError("Microduck control channel is not ready")
        if self.channel.bufferedAmount > 16384:
            raise RuntimeError("Microduck control channel stalled")

    def _message(self, raw):
        try:
            if len(raw) > 1_000_000:
                raise ValueError("oversized Microduck message")
            message = json.loads(raw)
            if message.get("jsonrpc") != "2.0":
                raise ValueError("invalid Microduck JSON-RPC response")
            future = self.pending.get(message.get("id"))
            if future and not future.done():
                result = message.get("result", {})
                if "error" in message or result.get("accepted") is False:
                    future.set_exception(RuntimeError(str(message.get("error", result))))
                else:
                    future.set_result(result)
            elif "method" in message:
                self.notification(message["method"], message.get("params", {}))
        except Exception as exc:
            self.fail(exc)

    async def call(self, method, params=None, deadline=0.5):
        self.check()
        self.sequence += 1
        sequence = self.sequence
        future = asyncio.get_running_loop().create_future()
        self.pending[sequence] = future
        try:
            self.channel.send(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": sequence,
                        "method": method,
                        "params": params or {},
                    }
                )
            )
            async with asyncio.timeout(deadline):
                return await future
        finally:
            self.pending.pop(sequence, None)

    async def start(self):
        self.http = aiohttp.ClientSession()
        self.task = asyncio.create_task(self._signal())
        try:
            async with asyncio.timeout(15):
                await self.ready.wait()
                self.check()
                if self.video is None:
                    raise RuntimeError("Microduck mediad did not offer its RGB camera")
        except Exception as exc:
            raise RuntimeError(
                f"Microduck unavailable at {self.address}: {exc}. "
                "Run on the robot with robotd and mediad running, or set --robot to mediad's "
                "LAN WebSocket address."
            ) from exc

    async def _signal(self):
        try:
            async with self.http.ws_connect(self.address, max_msg_size=1_000_000) as ws:
                session = None
                starting = False
                async for packet in ws:
                    if packet.type != aiohttp.WSMsgType.TEXT:
                        raise RuntimeError("Microduck signalling closed")
                    message = packet.json()
                    kind = message["type"]
                    if kind == "welcome":
                        await ws.send_json({"type": "list"})
                    elif kind == "list" and not starting:
                        producers = message.get("producers", [])
                        if len(producers) != 1:
                            raise RuntimeError("mediad must publish exactly one robot camera")
                        starting = True
                        await ws.send_json({"type": "startSession", "peerId": producers[0]["id"]})
                    elif kind == "sessionStarted":
                        session = message["sessionId"]
                    elif kind == "peer" and message.get("sessionId") == session:
                        if "sdp" in message:
                            await self.pc.setRemoteDescription(
                                RTCSessionDescription(**message["sdp"])
                            )
                            await self.pc.setLocalDescription(await self.pc.createAnswer())
                            await ws.send_json(
                                {
                                    "type": "peer",
                                    "sessionId": session,
                                    "sdp": {
                                        "type": "answer",
                                        "sdp": self.pc.localDescription.sdp,
                                    },
                                }
                            )
                        elif message.get("ice", {}).get("candidate"):
                            ice = message["ice"]
                            candidate = candidate_from_sdp(
                                ice["candidate"].removeprefix("candidate:")
                            )
                            candidate.sdpMLineIndex = ice["sdpMLineIndex"]
                            await self.pc.addIceCandidate(candidate)
                    elif kind in {"error", "endSession"}:
                        raise RuntimeError(str(message))
                raise RuntimeError("Microduck signalling disconnected")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.fail(exc)

    async def close(self):
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
        await self.pc.close()
        if self.http:
            await self.http.close()
