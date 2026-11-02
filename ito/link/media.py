import asyncio
import fractions
import logging
import sys

import av
from aiortc import MediaStreamTrack, codecs, jitterbuffer, rtcrtpreceiver
from aiortc.mediastreams import MediaStreamError

log = logging.getLogger(__name__)

RTP_TIME_BASE = fractions.Fraction(1, 90000)
# The GPU video decoders FFmpeg can open, by platform, best first; each is every vendor's.
HARDWARE = {
    "win32": ("d3d12va", "d3d11va", "dxva2"),
    "linux": ("vaapi", "vulkan"),
    "darwin": ("videotoolbox",),
}


class MarkerJitterBuffer(jitterbuffer.JitterBuffer):
    """aiortc's receive buffer, but a video frame leaves it with its last packet (the one with
    the RTP marker bit). aiortc's own waits for the next frame's first packet, which holds every
    frame back one frame interval (33 ms at 30 fps) at every hop. A frame whose marker never
    comes still leaves when the next one starts, as before."""

    def _remove_frame(self, sequence_number):
        if not self._is_video:
            return super()._remove_frame(sequence_number)
        packets = []
        for count in range(self.capacity):
            packet = self._packets[(self._origin + count) % self._capacity]
            if packet is None:
                return None
            if packets and packet.timestamp != packets[0].timestamp:
                break
            packets.append(packet)
            if packet.marker:
                break
        else:
            return None
        self.remove(len(packets))
        return jitterbuffer.JitterFrame(
            data=b"".join(p._data for p in packets), timestamp=packets[0].timestamp
        )


rtcrtpreceiver.JitterBuffer = MarkerJitterBuffer


def receive_h264(decoder):
    """Every H.264 stream this process receives goes through a `decoder()` of its own instead of
    aiortc's software one: its decode(JitterFrame) returns what the receiving track hands out."""
    rtcrtpreceiver.get_decoder = lambda codec: (
        decoder() if codec.mimeType.lower() == "video/h264" else codecs.get_decoder(codec)
    )


def _packet(encoded_frame):
    packet = av.Packet(encoded_frame.data)
    packet.pts = encoded_frame.timestamp
    packet.time_base = RTP_TIME_BASE
    return packet


class Passthrough:
    """Each received H.264 access unit as it came (an av.Packet, Annex B, pts its RTP time):
    for a driver that forwards a robot's own encoding rather than decode and encode it again."""

    def decode(self, encoded_frame):
        return [_packet(encoded_frame)] if encoded_frame.data else []


def _download(frame):
    """A GPU frame in memory, NV12. Downloading it drops its timestamp, which frames.FrameJoin
    needs to find its metadata."""
    out = frame.reformat(format="nv12")
    out.pts, out.time_base = frame.pts, frame.time_base
    return out


class VideoDecoder:
    """H.264 on the GPU's video decoder, whatever its vendor (Direct3D 12 or 11 video on Windows,
    VA-API or Vulkan on Linux), into frames a shader turns into RGB. Direct3D 12's stay in the
    GPU's memory, for render/camera.py to take straight to Vulkan; every other decoder's are
    downloaded here, on the decoder's thread (NV12). The CPU decodes only when no GPU decoder
    opens, or one cannot take a stream."""

    # False once render/camera.py finds Direct3D 12's frames cannot be shared with Vulkan: they
    # are downloaded here then too.
    share = True

    def __init__(self):
        from av.codec.hwaccel import HWAccel

        self.codec = None
        for kind in HARDWARE.get(sys.platform, ()):
            try:
                codec = av.CodecContext.create(
                    "h264",
                    "r",
                    hwaccel=HWAccel(
                        kind, allow_software_fallback=True, is_hw_owned=kind == "d3d12va"
                    ),
                )
            except (av.FFmpegError, ValueError, NotImplementedError):
                continue
            if codec.is_hwaccel:
                self.codec, self.kind = codec, kind
                break
        if self.codec is None:
            self.codec, self.kind = av.CodecContext.create("h264", "r"), "software"
        self.codec.flags |= av.codec.context.Flags.low_delay
        self.reported = False

    def decode(self, encoded_frame):
        if not encoded_frame.data:
            return []
        try:
            frames = self.codec.decode(_packet(encoded_frame))
        except av.FFmpegError as exc:
            log.warning("H.264 decode (%s) failed, frame skipped: %s", self.kind, exc)
            return []
        if not VideoDecoder.share:
            frames = [_download(f) if f.format.name == "d3d12" else f for f in frames]
        if frames and not self.reported:
            self.reported = True
            log.info(
                "Camera video: H.264 decoded by %s into %s",
                self.kind,
                frames[0].format.name,
            )
        return frames


class EncodedVideoTrack(MediaStreamTrack):
    """A video track whose recv() returns encoded av.Packets, which aiortc sends as they are.
    keyframe() asks its source for a keyframe (a receiver joining, or one that lost a frame)."""

    kind = "video"

    def keyframe(self):
        raise NotImplementedError


def forward_keyframes(sender, track):
    """A receiver's picture loss report (RTCP PLI) on `sender` asks `track`'s source for a
    keyframe: aiortc's own answer, a keyframe from its encoder, has no encoder to go to."""
    own = sender._send_keyframe

    def keyframe():
        own()
        track.keyframe()

    sender._send_keyframe = keyframe


class LatestTrack(MediaStreamTrack):
    def __init__(self, source: MediaStreamTrack):
        super().__init__()
        self.kind = source.kind
        self.source = source
        self._frames: asyncio.Queue = asyncio.Queue(maxsize=2)
        self._task = asyncio.create_task(self._drain())

    @property
    def id(self) -> str:
        return self.source.id

    def _put(self, frame) -> None:
        if self._frames.full():
            self._frames.get_nowait()
        self._frames.put_nowait(frame)

    async def _drain(self) -> None:
        try:
            while self.readyState == "live":
                self._put(await self.source.recv())
        except (MediaStreamError, asyncio.CancelledError):
            pass
        except Exception:
            log.exception("media track failed")
        finally:
            self._put(None)

    async def recv(self):
        if self.readyState != "live":
            raise MediaStreamError
        frame = await self._frames.get()
        if frame is None:
            self.stop()
            raise MediaStreamError
        return frame

    def stop(self) -> None:
        super().stop()
        self.source.stop()
        if self._task is not asyncio.current_task():
            self._task.cancel()
        self._put(None)
