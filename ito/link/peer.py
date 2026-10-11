"""One peer; bounded queues keep network backlog out of the pilot's input path.

Each kind of message rides its own data channel ("lane"). aiortc sends every channel through one
association-wide FIFO and never interleaves messages, so a lane alone does not keep a large
message from delaying a small one. Frame metadata (it can carry ~1 MB of depth) is therefore only
sent once earlier frames are acknowledged: a clock ping, status or safety command waits behind at
most one frame. Frames sent without that wait piled up in flight on a CPU-starved pilot, lost
packets and stalled Status behind SCTP's one-second retransmission timeout. Clock pings and pongs
wait for nothing to be unsent, frames hold back meanwhile, and that wait is kept out of the
measured round trip.
"""

import asyncio
import logging
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

from ito import clock, diagnostics
from ito.link.media import LatestTrack
from ito.protocol import (
    Command,
    Credential,
    FrameMetadata,
    Paired,
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
# Unacknowledged SCTP chunks (1200 bytes each) past which frames hold back: room for a few
# status, clock and command messages, not for a frame with depth.
FRAME_IN_FLIGHT = 16
# aiortc's teardown normally takes ~20 ms. On Windows it has hung for good after a long
# session (#27), awaiting a transport that never answers; closing is then abandoned so the
# pilot app can still exit within its own budget.
TEARDOWN_WAIT = 0.5
log = logging.getLogger(__name__)


def awaiting(task):
    """The chain of awaits a suspended task sits in, outermost first, as one line."""
    steps = []
    coro = task.get_coro()
    while coro is not None and len(steps) < 16:
        frame = getattr(coro, "cr_frame", None) or getattr(coro, "gi_frame", None)
        if frame is None:
            break
        steps.append(f"{frame.f_code.co_qualname}:{frame.f_lineno}")
        coro = getattr(coro, "cr_await", None) or getattr(coro, "gi_yieldfrom", None)
    return " > ".join(steps)


@dataclass(frozen=True)
class Lane:
    ordered: bool
    max_retransmits: int | None
    backlog: int  # Bytes of this lane that may wait unsent before a new message is dropped.
    pilot: tuple[type, ...]  # What the pilot sends on this channel.
    driver: tuple[type, ...]  # What the driver sends on this channel.
    bulk: bool = False  # Dropped while the association is backed up or a frame is in flight.


LANES = {
    "control": Lane(
        True, None, CONTROL_BUFFER, (Command, Paired), (RobotDescription, Credential, Status)
    ),
    "frames": Lane(False, None, 0, (), (FrameMetadata,), bulk=True),
    "clock": Lane(False, 0, 0, (Ping, Pong), (Ping, Pong)),
    "pilot": Lane(False, 0, 0, (PilotState,), ()),
}
# The lane that carries each message type, by sender.
ROUTES = {
    role: {kind: label for label, lane in LANES.items() for kind in getattr(lane, role)}
    for role in ("pilot", "driver")
}


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
        credential: Credential | None = None,
        audio=None,
    ):
        self.audio = audio
        self.role = role
        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=list(ice_servers)))
        self.on_message = on_message
        self.on_disconnect = on_disconnect
        self.on_track = on_track
        self.description = description
        # The driver sends a new pilot its credential; the pilot keeps the one it received.
        self.credential = credential
        self.clock = Clock()
        self.last_received = clock.now()
        self.rejected_messages = 0
        self.dropped_messages = 0
        self.messages: asyncio.Queue[WireMessage] = asyncio.Queue(maxsize=128)
        self.tracks: asyncio.Queue = asyncio.Queue(maxsize=32)
        self.frames: dict[str, FrameMetadata] = {}
        self._media: list[LatestTrack] = []
        self.channels: dict[str, RTCDataChannel] = {}
        self.ready = asyncio.Event()
        self.closed = asyncio.Event()
        self.robot_received = asyncio.Event()
        self.clock_ready = asyncio.Event()
        self.credential_received = asyncio.Event()
        self._clock_task: asyncio.Task | None = None
        self._replies: set[asyncio.Task] = set()
        self._clock_waiting = 0
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
                if track.kind == "audio" and self.audio:
                    self.audio.receive(latest)
                elif self.on_track:
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
            diagnostics.event("peer_state", role=self.role, state=self.pc.connectionState)
            if self.pc.connectionState in {"failed", "closed"}:
                await self.close()

        if role == "pilot":
            for label, lane in LANES.items():
                self._bind(
                    self.pc.createDataChannel(
                        label,
                        ordered=lane.ordered,
                        maxRetransmits=lane.max_retransmits,
                        protocol="ito/1",
                    )
                )
        else:
            self.pc.on("datachannel", self._bind)

    @property
    def control(self) -> RTCDataChannel | None:
        return self.channels.get("control")

    @property
    def pilot(self) -> RTCDataChannel | None:
        return self.channels.get("pilot")

    def _bind(self, channel: RTCDataChannel) -> None:
        lane = LANES.get(channel.label)
        if (
            lane is None
            or channel.label in self.channels
            or channel.protocol != "ito/1"
            or channel.ordered != lane.ordered
            or channel.maxRetransmits != lane.max_retransmits
            or channel.maxPacketLifeTime is not None
        ):
            self.rejected_messages += 1
            channel.close()
            return
        self.channels[channel.label] = channel

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
            and self._open()
        )

    def _open(self) -> bool:
        return len(self.channels) == len(LANES) and all(
            channel.readyState == "open" for channel in self.channels.values()
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
        if not self._open() or self.ready.is_set():
            return
        self.ready.set()
        if self.role == "driver" and self.description:
            self.send(self.description)
        if self.role == "driver" and self.credential:
            self.send(self.credential)
        self._clock_task = asyncio.create_task(self._synchronize())

    def _receive(self, label: str, data: str | bytes) -> None:
        if self._closing or self._disconnect_notified:
            return
        try:
            message = decode(data)
            sender = "driver" if self.role == "pilot" else "pilot"
            if ROUTES[sender].get(type(message)) != label:
                raise ProtocolError("wrong message direction or channel")
            if isinstance(message, PilotState):
                if message.sequence <= self._pilot_sequence:
                    self.dropped_messages += 1
                    return
                self._pilot_sequence = message.sequence
            else:
                if isinstance(message, Credential):
                    if self.credential_received.is_set():
                        raise ProtocolError("one credential per connection")
                    self.credential = message
                    self.credential_received.set()
                    return
                if isinstance(message, Ping):
                    self.last_received = received = clock.now()
                    if len(self._replies) >= 4:
                        self.dropped_messages += 1
                        return
                    reply = asyncio.create_task(
                        self._send_clock(
                            lambda: Pong(
                                sequence=message.sequence,
                                sent=message.sent,
                                received=received,
                                replied=clock.now(),
                            )
                        )
                    )
                    self._replies.add(reply)
                    reply.add_done_callback(self._replies.discard)
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
                    # Frames and the description travel on separate channels.
                    if self.description is None:
                        self.dropped_messages += 1
                        return
                    cameras = {c.name: c for c in self.description.cameras}
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
            self.last_received = clock.now()
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
        label = ROUTES[self.role].get(type(message))
        if label is None:
            raise ValueError("wrong message direction")
        lane, channel = LANES[label], self.channels.get(label)
        if (
            not self.ready.is_set()
            or self._closing
            or self._disconnect_notified
            or channel is None
            or channel.readyState != "open"
            or channel.bufferedAmount > lane.backlog
            or (lane.bulk and (self._clock_waiting or not self._idle(FRAME_IN_FLIGHT)))
        ):
            self.dropped_messages += 1
            return False
        channel.send(encode(message))
        return True

    def _idle(self, in_flight: int | None = None) -> bool:
        """Nothing waits unsent and, if given, at most in_flight chunks wait for the receiver."""
        # aiortc exposes no association-wide queue depth; aiortc is pinned below 2.
        sctp = self.pc.sctp
        if sctp._data_channel_queue or sctp._outbound_queue:
            return False
        return in_flight is None or len(sctp._sent_queue) <= in_flight

    async def _send_clock(self, stamp: Callable[[], Ping | Pong]) -> Ping | Pong | None:
        """Stamp and send a clock message into an idle association, ahead of the next frame.

        Waiting before a ping is stamped, or between receiving a ping and stamping the pong,
        is not part of the measured round trip, so queueing never biases the offset.
        """
        self._clock_waiting += 1
        try:
            deadline = clock.now() + 1
            while not self._idle():
                if self._closing or clock.now() > deadline:
                    self.dropped_messages += 1
                    return None
                await asyncio.sleep(0.002)
        finally:
            self._clock_waiting -= 1
        message = stamp()
        return message if self.send(message) else None

    async def _synchronize(self) -> None:
        sequence = 0
        while not self._closing:
            ping = await self._send_clock(lambda s=sequence: Ping(sequence=s, sent=clock.now()))
            if ping:
                self._pending_pings[ping.sequence] = ping.sent
            now = clock.now()
            self._pending_pings = {s: t for s, t in self._pending_pings.items() if now - t < 5}
            sequence += 1
            await asyncio.sleep(0.5)

    def _clock_sample(self, message: Pong) -> None:
        sent = self._pending_pings.pop(message.sequence, None)
        now = clock.now()
        if sent is None or sent != message.sent:
            self.rejected_messages += 1
            return
        rtt = (now - sent) - (message.replied - message.received)
        # Clock granularity on either end can make a LAN round trip read slightly negative.
        if rtt < -0.001 or rtt > 5:
            self.rejected_messages += 1
            return
        offset = ((message.received - sent) + (message.replied - now)) / 2
        self.last_received = now
        self._samples.append((max(0, rtt), offset))
        self.clock.rtt, self.clock.offset = min(self._samples)
        self.clock_ready.set()

    async def close(self) -> None:
        if self._closing:
            await self.closed.wait()
            return
        self._closing = True
        self._notify_disconnect()
        try:
            tasks = [*self._replies, *([self._clock_task] if self._clock_task else [])]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for track in self._media:
                track.stop()
            await asyncio.gather(*(track._task for track in self._media), return_exceptions=True)
            with diagnostics.stage("webrtc"):
                closing = asyncio.ensure_future(self.pc.close())
                await asyncio.wait({closing}, timeout=TEARDOWN_WAIT)
                if not closing.done():
                    where = awaiting(closing)
                    log.warning(
                        "WebRTC teardown abandoned after %.1f s at %s", TEARDOWN_WAIT, where
                    )
                    diagnostics.event("webrtc_teardown_abandoned", at=where)
                    closing.cancel()
        finally:
            try:
                if self.audio:
                    with diagnostics.stage("audio"):
                        await self.audio.close()
            finally:
                self.closed.set()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()
