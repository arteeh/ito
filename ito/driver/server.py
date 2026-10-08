import asyncio
import logging
import math
from collections.abc import Sequence
from pathlib import Path

from aiohttp import web
from aiortc import RTCIceServer, RTCSessionDescription
from aiortc.sdp import SessionDescription

from ito import clock
from ito.driver.adapter import Adapter
from ito.driver.pairing import Pairing, PairingRefused, default_path
from ito.link import Peer
from ito.link.audio import Audio, opus
from ito.link.pairing import proof, token
from ito.link.signaling import parse_offer
from ito.protocol import (
    VERSION,
    Command,
    Credential,
    FrameMetadata,
    Paired,
    PilotState,
    Status,
    WireMessage,
    encode,
)

log = logging.getLogger(__name__)


class Driver:
    def __init__(
        self,
        adapter: Adapter,
        *,
        audio_source: str | None = None,
        audio_sink: str = "device",
        input_timeout: float = 0.25,
        command_rate: float = 90,
        ice_servers: Sequence[RTCIceServer] = (),
        pairing_file: Path | None = None,
    ):
        if not math.isfinite(input_timeout) or not 0.02 <= input_timeout <= 5:
            raise ValueError("input_timeout must be between 0.02 and 5 seconds")
        if not math.isfinite(command_rate) or not 1 <= command_rate <= 240:
            raise ValueError("command_rate must be between 1 and 240 Hz")
        if audio_source is not None:
            Audio(audio_source, audio_sink)  # Reject invalid CLI sources before serving.
        if (audio_source, audio_sink) == ("none", "none"):
            audio_source = None  # A robot without microphone and speaker carries no audio.
        self.audio_source, self.audio_sink = audio_source, audio_sink
        self.adapter = adapter
        adapter.frame_sink = self.publish_frame
        self.input_timeout = input_timeout
        self.command_rate = command_rate
        self.ice_servers = ice_servers
        self.pairing = Pairing(pairing_file or default_path())
        self._pairing_code: str | None = None  # The code the current pilot paired with.
        self.peer: Peer | None = None
        self.state = "neutral"
        self.reason = "waiting for pilot"
        # Every connection starts stopped: piloting begins only when the pilot resumes.
        self._stopped = True
        self._estop = False
        self._fault = False
        self._is_neutral = False
        self._last_neutral_attempt = 0.0
        self._latest: PilotState | None = None
        self._received = 0.0
        self._capture = -1.0
        self._not_before = 0.0
        self._applied_sequence = -1
        self._command_sequence = -1
        self._task: asyncio.Task | None = None
        self._runner: web.AppRunner | None = None
        self._last_status = 0.0
        self._last_apply = 0.0
        self._input_latency_ms: float | None = None
        self._connected_at = 0.0
        self._negotiating = False
        self._closing = False
        self.address: str | None = None
        self.failed = asyncio.Event()  # The watchdog died; serve() exits instead of idling.

    def _neutral(self, reason: str) -> None:
        self._latest = None
        self.reason = reason
        self.state = (
            "fault"
            if self._fault
            else ("e-stopped" if self._estop else "stopped" if self._stopped else "neutral")
        )
        if self._is_neutral:
            return
        log.info(
            "Robot neutral: %s; input_age_ms=%.1f",
            reason,
            (clock.now() - self._capture) * 1000 if self._capture >= 0 else -1,
        )
        self._last_neutral_attempt = clock.now()
        try:
            self.adapter.neutral()
            self._is_neutral = True
        except Exception:
            self._fault = True
            self.state, self.reason = "fault", "adapter neutral failed"
            log.exception("adapter neutral failed")

    def _disconnect(self, peer: Peer) -> None:
        if peer is self.peer:
            self._neutral("pilot disconnected")

    def _message(self, peer: Peer, message: WireMessage) -> None:
        if peer is not self.peer or not peer.connected or self._closing:
            return
        if isinstance(message, PilotState):
            if self.peer is None or self.peer.clock.offset is None:
                return
            now = clock.now()
            capture = self.peer.clock.remote_to_local(message.capture_time)
            if (
                capture <= self._capture
                or capture < self._not_before
                or not -0.1 <= now - capture <= self.input_timeout
            ):
                self.peer.rejected_messages += 1
                return
            self._capture = capture
            self._received = now
            if not message.deadman:
                self._neutral("deadman released")
            elif not (self._estop or self._stopped or self._fault):
                self._latest = message
        elif isinstance(message, Paired):
            credential = peer.credential
            if self._pairing_code and credential and message.pilot == credential.pilot:
                code, self._pairing_code = self._pairing_code, None
                try:
                    self.pairing.paired(code, credential.pilot, credential.secret)
                    log.info("Pilot paired; the pairing code is used up")
                except (OSError, ValueError):
                    log.exception("Cannot store the pilot's credential; the code stays valid")
            else:
                peer.rejected_messages += 1
        elif isinstance(message, Command):
            if message.sequence <= self._command_sequence:
                if self.peer:
                    self.peer.rejected_messages += 1
                return
            self._command_sequence = message.sequence
            if message.action == "e-stop":
                self._estop = True
                self._neutral("e-stop")
            elif message.action == "stop":
                self._stopped = True
                self._neutral("stop")
            elif not self._fault:
                self._stopped = self._estop = False
                self._not_before = clock.now()
                self._neutral("resumed; waiting for fresh deadman input")
            self._status()

    def _status(self) -> None:
        if not self.peer or not self.peer.ready.is_set():
            return
        try:
            self.peer.send(
                Status(
                    state=self.state,
                    reason=self.reason,
                    command_sequence=self._command_sequence
                    if self._command_sequence >= 0
                    else None,
                    rejected_messages=self.peer.rejected_messages,
                    telemetry=self.adapter.telemetry()
                    | ({"audio": self.peer.audio.status} if self.peer.audio else {})
                    | (
                        {"pilot_input_latency_ms": self._input_latency_ms}
                        if self._input_latency_ms is not None
                        else {}
                    ),
                )
            )
        except Exception:
            self._fault = True
            self._neutral("adapter telemetry failed")
            log.exception("adapter telemetry failed")
            self.peer.send(
                Status(
                    state="fault",
                    reason=self.reason,
                    command_sequence=self._command_sequence
                    if self._command_sequence >= 0
                    else None,
                    rejected_messages=self.peer.rejected_messages,
                )
            )

    def publish_frame(self, metadata: FrameMetadata) -> bool:
        camera = next(
            (c for c in self.adapter.description.cameras if c.name == metadata.camera), None
        )
        if camera is None:
            raise ValueError("unknown camera")
        if metadata.depth and (
            metadata.depth.width != camera.intrinsics.width
            or metadata.depth.height != camera.intrinsics.height
        ):
            raise ValueError("depth dimensions do not match camera")
        return self.peer.send(metadata) if self.peer else False

    async def _run(self) -> None:
        interval = min(0.01, self.input_timeout / 4, 1 / self.command_rate)
        while True:
            now = clock.now()
            if self._fault and not self._is_neutral:
                if now - self._last_neutral_attempt >= 1 / self.command_rate:
                    self._neutral(self.reason)
            # Reserve a watchdog tick so neutral is enqueued by the configured timeout.
            if (
                self._latest
                and now - min(self._received, self._capture) >= self.input_timeout - interval
            ):
                self._neutral("input timeout")
            if self._latest and self._latest.sequence != self._applied_sequence:
                if now - self._last_apply >= 1 / self.command_rate:
                    try:
                        self._is_neutral = False
                        self.adapter.apply(self._latest)
                        self._input_latency_ms = max(0, (clock.now() - self._capture) * 1000)
                        self._applied_sequence = self._latest.sequence
                        self._last_apply = now
                        self.state, self.reason = "active", "pilot input"
                    except Exception:
                        self._fault = True
                        self._neutral("adapter apply failed")
                        log.exception("adapter apply failed")
            if now - self._last_status >= 0.1:
                self._status()
                self._last_status = now
            # An abandoned offer or dead peer must not prevent the next direct connection.
            if self.peer and not self._negotiating:
                idle_since = max(self.peer.last_received, self._connected_at)
                if self.peer.closed.is_set() or now - idle_since > max(5, 4 * self.input_timeout):
                    expired = self.peer
                    await expired.close()
                    if self.peer is expired:
                        self.peer = None
            await asyncio.sleep(interval)

    def _watchdog_ended(self, task: asyncio.Task) -> None:
        """Without the watchdog nothing enforces the input timeout: stop the robot for good."""
        if task.cancelled() or self._closing:
            return
        log.critical("Driver watchdog stopped", exc_info=task.exception())
        self._fault = True
        self._is_neutral = False
        self._neutral("driver watchdog stopped")
        self.failed.set()

    async def _challenge(self, request: web.Request) -> web.Response:
        return web.json_response({"nonce": self.pairing.nonce()})

    async def _offer(self, request: web.Request) -> web.Response:
        try:
            offer = parse_offer(await request.read())
        except ValueError:
            raise web.HTTPBadRequest(text="invalid or unsupported Ito offer") from None
        if self._closing:
            raise web.HTTPServiceUnavailable()
        try:
            grant = self.pairing.check(
                offer.nonce, offer.proof, offer.sdp, offer.pilot, request.remote or ""
            )
        except PairingRefused as refused:
            log.warning("Refused pilot at %s: %s", request.remote, refused)
            return web.Response(status=refused.status, text=str(refused))
        if self.peer and not self.peer.closed.is_set():
            raise web.HTTPConflict(text="this robot already has a pilot")
        description = self.adapter.description
        encode(description)
        self._stopped = True
        self._neutral("connecting; the pilot resumes to begin")
        self._capture = -1.0
        self._not_before = clock.now()
        self._received = 0.0
        self._command_sequence = self._applied_sequence = -1
        peer = Peer(
            "driver",
            description=description,
            credential=Credential(pilot=token(), secret=token()) if grant.code else None,
            audio=Audio(self.audio_source, self.audio_sink) if self.audio_source else None,
            ice_servers=self.ice_servers,
            on_message=lambda message: self._message(peer, message),
            on_disconnect=lambda: self._disconnect(peer),
            on_track=lambda track: audio_received(track),
        )
        self.peer = peer
        self._pairing_code = grant.code
        self._connected_at = clock.now()
        self._negotiating = True

        def audio_received(track):
            if track.kind == "audio":
                try:
                    self.adapter.incoming_audio(track)
                except Exception:
                    track.stop()
                    self._fault = True
                    self._neutral("adapter audio failed")
                    log.exception("adapter audio failed")
            else:
                track.stop()

        tracks = []
        try:
            async with asyncio.timeout(15):
                remote_media = SessionDescription.parse(offer.sdp).media
                if len(remote_media) > 18:
                    raise ValueError("too many media tracks")
                tracks = list(self.adapter.media_tracks())
                # Adapters with their own audio keep ownership of both directions.
                if any(track.kind == "audio" for track in tracks):
                    peer.audio = None
                await peer.pc.setRemoteDescription(RTCSessionDescription(offer.sdp, "offer"))
                if peer.audio:
                    await peer.audio.start()
                    if peer.audio.source != "none" and any(
                        m.kind == "audio" and m.direction in {"recvonly", "sendrecv"}
                        for m in remote_media
                    ):
                        tracks.append(peer.audio.track)
                    # The pilot's microphone needs the robot's speaker and its speakers the
                    # robot's microphone; it opens only the devices that have a counterpart.
                    audio = {"microphone": peer.audio.source, "speaker": peer.audio.sink}
                    capabilities = description.capabilities + tuple(
                        name
                        for name, choice in audio.items()
                        if choice != "none" and name not in description.capabilities
                    )
                    peer.description = description.model_copy(update={"capabilities": capabilities})
                video = [track for track in tracks if track.kind == "video"]
                if len(video) != len(description.cameras) or {track.id for track in video} != {
                    c.track_id for c in description.cameras
                }:
                    raise ValueError("camera descriptions do not match media tracks")
                for kind in ("video", "audio"):
                    offered = sum(
                        media.kind == kind and media.direction in {"recvonly", "sendrecv"}
                        for media in remote_media
                    )
                    outgoing = sum(track.kind == kind for track in tracks)
                    if outgoing > offered:
                        raise ValueError(f"pilot must offer {outgoing} receiving {kind} tracks")
                for track in tracks:
                    peer.pc.addTrack(track)
                opus(peer.pc)
                await peer.pc.setLocalDescription(await peer.pc.createAnswer())
            sdp = peer.pc.localDescription.sdp
            return web.json_response(
                {
                    "version": VERSION,
                    "type": "answer",
                    "sdp": sdp,
                    "proof": proof(grant.key, offer.nonce, "answer", sdp),
                }
            )
        except asyncio.CancelledError:
            for track in tracks:
                track.stop()
            await peer.close()
            raise
        except Exception:
            for track in tracks:
                track.stop()
            await peer.close()
            log.exception("connection negotiation failed")
            raise web.HTTPBadRequest(text="unable to negotiate media; check driver logs") from None
        finally:
            self._negotiating = False

    async def start(self, host: str = "0.0.0.0", port: int = 8080) -> str:
        if self._closing or self._runner is not None:
            raise RuntimeError("driver already started or closed")
        app = web.Application(client_max_size=220_000)
        app.router.add_post("/pairing", self._challenge)
        app.router.add_post("/offer", self._offer)
        self._runner = web.AppRunner(app, shutdown_timeout=2)
        try:
            await self.adapter.start()
            encode(self.adapter.description)
            self._neutral("driver started")
            await self._runner.setup()
            site = web.TCPSite(self._runner, host, port)
            await site.start()
            socket = site._server.sockets[0]
            actual_port = socket.getsockname()[1]
            url_host = f"[{host}]" if ":" in host else host
            self.address = f"http://{url_host}:{actual_port}"
            self._task = asyncio.create_task(self._run())
            self._task.add_done_callback(self._watchdog_ended)
            return self.address
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._neutral("driver shutting down")
        self.adapter.frame_sink = None
        try:
            if self._task:
                self._task.cancel()
                # A watchdog that died has been reported by _watchdog_ended already.
                await asyncio.gather(self._task, return_exceptions=True)
            if self._runner:
                await self._runner.cleanup()
            if self.peer:
                await self.peer.close()
        finally:
            await self.adapter.close()
