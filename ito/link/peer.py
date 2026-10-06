"""One peer; bounded queues keep network backlog out of the pilot's input path."""

import asyncio
import contextlib
import logging
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCDataChannel,
    RTCIceServer,
    RTCPeerConnection,
)

from ito.link.media import LatestTrack
from ito.protocol import (
    Command,
    FrameMetadata,
    PilotState,
    Ping,
    Pong,
    ProtocolError,
    RobotDescription,
    Status,
    WireMessage,
    decode,
    encode,
)

CONTROL_BUFFER = 2_000_000
log = logging.getLogger(__name__)


@dataclass
class Clock:
    """Offset is remote minus local monotonic time; lowest RTT limits queueing bias."""

    offset: float | None = None
    rtt: float | None = None

    def remote_to_local(self, timestamp: float) -> float:
        if self.offset is None:
            raise RuntimeError("clock has not synchronized")
        return timestamp - self.offset


class Peer:
    def __init__(
        self,
        role: Literal["pilot", "driver"],
        *,
        ice_servers: Sequence[RTCIceServer] = (),
        on_message: Callable[[WireMessage], None] | None = None,
        on_disconnect: Callable[[], None] | None = None,
        on_track: Callable[[MediaStreamTrack], None] | None = None,
        description: RobotDescription | None = None,
    ):
        self.role = role
        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=list(ice_servers)))
        self.on_message = on_message
        self.on_disconnect = on_disconnect
        self.on_track = on_track
        self.description = description
        self.clock = Clock()
        self.last_received = time.monotonic()
        self.rejected_messages = 0
        self.dropped_messages = 0
        self.messages: asyncio.Queue[WireMessage] = asyncio.Queue(maxsize=128)
        self.tracks: asyncio.Queue = asyncio.Queue(maxsize=32)
        self.frames: dict[str, FrameMetadata] = {}
        self._media: list[LatestTrack] = []
        self.control: RTCDataChannel | None = None
        self.pilot: RTCDataChannel | None = None
        self.ready = asyncio.Event()
        self.closed = asyncio.Event()
        self.robot_received = asyncio.Event()
        self.clock_ready = asyncio.Event()
        self._clock_task: asyncio.Task | None = None
        self._closing = False
        self._disconnect_notified = False
        self._close_task: asyncio.Task | None = None
        self._pending_pings: dict[int, float] = {}
        self._samples: deque[tuple[float, float]] = deque(maxlen=8)
        self._pilot_sequence = -1

        @self.pc.on("track")
        def track_received(track):
            if len(self._media) >= 32 or self.tracks.full():
                track.stop()
                self.rejected_messages += 1
            else:
                latest = LatestTrack(track)
                self._media.append(latest)
                if self.on_track:
                    try:
                        self.on_track(latest)
                    except Exception:
                        log.exception("media callback failed")
                        latest.stop()
                        self._notify_disconnect()
                        self._schedule_close()
                else:
                    self.tracks.put_nowait(latest)

        @self.pc.on("connectionstatechange")
        async def connection_changed():
            if self.pc.connectionState in {"failed", "closed"}:
                await self.close()

        if role == "pilot":
            self._bind(self.pc.createDataChannel("control", protocol="ito/1"))
            self._bind(
                self.pc.createDataChannel(
                    "pilot", ordered=False, maxRetransmits=0, protocol="ito/1"
                )
            )
        else:
            self.pc.on("datachannel", self._bind)

    def _bind(self, channel: RTCDataChannel) -> None:
        valid = channel.protocol == "ito/1"
        if channel.label == "pilot":
            valid = valid and not channel.ordered and channel.maxRetransmits == 0
            valid = valid and channel.maxPacketLifeTime is None and self.pilot is None
            if valid:
                self.pilot = channel
        elif channel.label == "control":
            valid = valid and channel.ordered and channel.maxRetransmits is None
            valid = valid and channel.maxPacketLifeTime is None and self.control is None
            if valid:
                self.control = channel
        else:
            valid = False
        if not valid:
            self.rejected_messages += 1
            channel.close()
            return

        channel.on("message", lambda data: self._receive(channel.label, data))

        @channel.on("close")
        def channel_closed():
            self._notify_disconnect()
            self._schedule_close()

        @channel.on("open")
        def channel_opened():
            self._opened()

        self._opened()

    @property
    def connected(self) -> bool:
        return bool(
            not self._closing
            and not self._disconnect_notified
            and self.ready.is_set()
            and self.control
            and self.control.readyState == "open"
            and self.pilot
            and self.pilot.readyState == "open"
        )

    def _notify_disconnect(self) -> None:
        if not self._disconnect_notified:
            self._disconnect_notified = True
            if self.on_disconnect:
                try:
                    self.on_disconnect()
                except Exception:
                    log.exception("disconnect callback failed")

    def _schedule_close(self) -> None:
        if self._close_task is None and not self._closing:
            self._close_task = asyncio.create_task(self.close())

    def _opened(self) -> None:
        if not (self.control and self.pilot):
            return
        if self.control.readyState != "open" or self.pilot.readyState != "open":
            return
        if self.ready.is_set():
            return
        self.ready.set()
        if self.role == "driver" and self.description:
            self.send(self.description)
        self._clock_task = asyncio.create_task(self._synchronize())

    def _receive(self, label: str, data: str | bytes) -> None:
        if self._closing or self._disconnect_notified:
            return
        try:
            message = decode(data)
            if label == "pilot":
                if self.role != "driver" or not isinstance(message, PilotState):
                    raise ProtocolError("wrong message direction or channel")
                if message.sequence <= self._pilot_sequence:
                    self.dropped_messages += 1
                    return
                self._pilot_sequence = message.sequence
            else:
                allowed = (
                    (Command, Ping, Pong)
                    if self.role == "driver"
                    else (RobotDescription, FrameMetadata, Status, Ping, Pong)
                )
                if not isinstance(message, allowed):
                    raise ProtocolError("wrong message direction or channel")
                if isinstance(message, Ping):
                    self.last_received = time.monotonic()
                    received = time.monotonic()
                    self.send(
                        Pong(
                            sequence=message.sequence,
                            sent=message.sent,
                            received=received,
                            replied=time.monotonic(),
                        )
                    )
                    return
                if isinstance(message, Pong):
                    self._clock_sample(message)
                    return
                if isinstance(message, RobotDescription):
                    if self.description is not None:
                        raise ProtocolError("robot description may not change during a connection")
                    self.description = message
                    self.robot_received.set()
                if isinstance(message, FrameMetadata):
                    cameras = (
                        {c.name: c for c in self.description.cameras} if self.description else {}
                    )
                    if message.camera not in cameras:
                        raise ProtocolError("unknown camera")
                    camera = cameras[message.camera]
                    if message.depth and (
                        message.depth.width != camera.intrinsics.width
                        or message.depth.height != camera.intrinsics.height
                    ):
                        raise ProtocolError("depth does not match camera")
                    previous = self.frames.get(message.camera)
                    if previous and message.sequence <= previous.sequence:
                        self.dropped_messages += 1
                        return
                    self.frames[message.camera] = message
            self.last_received = time.monotonic()
            if self.on_message:
                self.on_message(message)
            elif not isinstance(message, PilotState):
                if self.messages.full():
                    self.messages.get_nowait()
                    self.dropped_messages += 1
                self.messages.put_nowait(message)
        except ProtocolError:
            self.rejected_messages += 1
        except Exception:
            log.exception("message callback failed")
            self._notify_disconnect()
            self._schedule_close()

    def send(self, message: WireMessage) -> bool:
        if isinstance(message, PilotState):
            if self.role != "pilot":
                raise ValueError("only a pilot sends pilot state")
            channel = self.pilot
            limit = 0
        else:
            allowed = (
                (Command, Ping, Pong)
                if self.role == "pilot"
                else (RobotDescription, FrameMetadata, Status, Ping, Pong)
            )
            if not isinstance(message, allowed):
                raise ValueError("wrong message direction")
            channel = self.control
            limit = CONTROL_BUFFER
        if (
            self._closing
            or self._disconnect_notified
            or channel is None
            or channel.readyState != "open"
            or channel.bufferedAmount > limit
        ):
            self.dropped_messages += 1
            return False
        channel.send(encode(message))
        return True

    async def _synchronize(self) -> None:
        sequence = 0
        while not self._closing:
            sent = time.monotonic()
            self._pending_pings = {s: t for s, t in self._pending_pings.items() if sent - t < 5}
            if self.send(Ping(sequence=sequence, sent=sent)):
                self._pending_pings[sequence] = sent
            sequence += 1
            await asyncio.sleep(0.5)

    def _clock_sample(self, message: Pong) -> None:
        sent = self._pending_pings.pop(message.sequence, None)
        now = time.monotonic()
        if sent is None or sent != message.sent:
            self.rejected_messages += 1
            return
        rtt = (now - sent) - (message.replied - message.received)
        if rtt < 0 or rtt > 5:
            self.rejected_messages += 1
            return
        offset = ((message.received - sent) + (message.replied - now)) / 2
        self.last_received = now
        self._samples.append((rtt, offset))
        self.clock.rtt, self.clock.offset = min(self._samples)
        self.clock_ready.set()

    async def close(self) -> None:
        if self._closing:
            await self.closed.wait()
            return
        self._closing = True
        self._notify_disconnect()
        try:
            if self._clock_task:
                self._clock_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._clock_task
            for track in self._media:
                track.stop()
            await asyncio.gather(*(track._task for track in self._media), return_exceptions=True)
            await self.pc.close()
        finally:
            self.closed.set()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()
