"""20 ms mono audio; bounded device queues discard backlog instead of adding latency."""

import asyncio
import logging
import time
import wave
from collections import deque
from fractions import Fraction

import numpy as np
from aiortc import MediaStreamTrack, RTCRtpSender
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame, AudioResampler

RATE, SAMPLES = 48000, 960
log = logging.getLogger(__name__)


def arguments(parser, *, source="device"):
    parser.add_argument("--audio-source", default=source, help="device, none, or tone:Hz")
    parser.add_argument("--audio-sink", default="device", help="device, none, or WAV output path")


def opus(pc):
    # aiortc's audio jitter buffer prefetches four packets (80 ms at our packet size).
    codecs = [c for c in RTCRtpSender.getCapabilities("audio").codecs if c.mimeType == "audio/opus"]
    for transceiver in pc.getTransceivers():
        if transceiver.kind == "audio":
            transceiver.setCodecPreferences(codecs)


class Microphone(MediaStreamTrack):
    kind = "audio"

    def __init__(self, audio):
        super().__init__()
        self.audio = audio
        self.pts = 0
        self.deadline = None

    async def recv(self):
        if self.readyState != "live":
            raise MediaStreamError
        now = time.monotonic()
        self.deadline = max(self.deadline or now, now - 0.02)
        await asyncio.sleep(max(0, self.deadline - now))
        self.deadline += 0.02
        audio = self.audio
        samples = np.zeros(SAMPLES, dtype=np.int16)
        if audio.frequency:
            samples = (
                4000 * np.sin(2 * np.pi * audio.frequency * (np.arange(SAMPLES) + self.pts) / RATE)
            ).astype(np.int16)
        elif audio.capture:
            samples = audio.capture.popleft()
        if audio.mic_muted:
            samples.fill(0)
        frame = AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate, frame.pts, frame.time_base = RATE, self.pts, Fraction(1, RATE)
        self.pts += SAMPLES
        return frame


class Audio:
    def __init__(self, source="device", sink="device"):
        self.source, self.sink = source, sink
        self.frequency = 0.0
        if source.startswith("tone:"):
            self.frequency = float(source[5:])
            if not 20 <= self.frequency <= 20000:
                raise ValueError("audio tone must be between 20 and 20000 Hz")
        elif source not in {"device", "none"}:
            raise ValueError("audio source must be device, none, or tone:Hz")
        self.capture, self.playback = deque(maxlen=2), deque(maxlen=2)
        self.mic_muted = self.speaker_muted = False
        self.input_status = self.output_status = "starting"
        self.streams = []
        self.writer = None
        self.task = None
        self.track = Microphone(self)

    @property
    def status(self):
        return f"Audio: mic {self.input_status} | speaker {self.output_status}"

    def _open(self):
        sd = None
        if "device" in {self.source, self.sink}:
            try:
                import sounddevice as sd
            except (ImportError, OSError):
                pass
        for incoming, choice in ((True, self.source), (False, self.sink)):
            status = "off" if choice == "none" else "ready"
            stream = None
            try:
                if choice == "device":
                    if sd is None:
                        raise RuntimeError("PortAudio unavailable")
                    if incoming:

                        def capture(data, frames, timing, flags):
                            self.capture.append(data[:, 0].copy())

                        stream = sd.InputStream(
                            callback=capture,
                            samplerate=RATE,
                            blocksize=SAMPLES,
                            channels=1,
                            dtype="int16",
                            latency="low",
                        )
                    else:

                        def playback(data, frames, timing, flags):
                            data.fill(0)
                            if self.playback and not self.speaker_muted:
                                data[:, 0] = self.playback.popleft()
                            else:
                                self.playback.clear()

                        stream = sd.OutputStream(
                            callback=playback,
                            samplerate=RATE,
                            blocksize=SAMPLES,
                            channels=1,
                            dtype="int16",
                            latency="low",
                        )
                    stream.start()
                    self.streams.append((incoming, stream))
                elif not incoming and choice != "none":
                    self.writer = wave.open(choice, "wb")
                    self.writer.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
            except Exception as exc:
                if stream:
                    stream.close()
                status = "unavailable"
                log.info("Audio %s unavailable: %s", "mic" if incoming else "speaker", exc)
            if incoming:
                self.input_status = status
            else:
                self.output_status = status
        log.info(self.status)

    async def start(self):
        await self._io(self._open)

    def receive(self, track):
        if self.task is not None:
            track.stop()
            return
        self.task = asyncio.create_task(self._receive(track))

    async def _receive(self, track):
        resampler = AudioResampler(format="s16", layout="mono", rate=RATE, frame_size=SAMPLES)
        try:
            while True:
                received = await track.recv()
                for frame in resampler.resample(received):
                    samples = frame.to_ndarray().reshape(-1).copy()
                    if self.speaker_muted:
                        samples.fill(0)
                    if self.writer:
                        await self._io(self.writer.writeframesraw, samples.tobytes())
                    self.playback.append(samples)
                for incoming, stream in self.streams:
                    if not stream.active:
                        if incoming:
                            self.input_status = "unavailable"
                        else:
                            self.output_status = "unavailable"
        except (MediaStreamError, asyncio.CancelledError):
            pass
        except Exception as exc:
            self.output_status = "unavailable"
            log.info("Audio playback unavailable: %s", exc)
        finally:
            track.stop()
            self.playback.clear()

    @staticmethod
    async def _io(function, *args):
        # Finish device/file I/O before cancellation can close its resources.
        task = asyncio.create_task(asyncio.to_thread(function, *args))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    async def close(self):
        self.track.stop()
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None
        await asyncio.to_thread(self._close)

    def _close(self):
        for _, stream in self.streams:
            try:
                stream.close()
            except Exception as exc:
                log.info("Audio device closed: %s", exc)
        self.streams.clear()
        if self.writer:
            try:
                self.writer.close()
            except OSError as exc:
                log.info("Audio recording closed: %s", exc)
            self.writer = None
        self.capture.clear()
        self.playback.clear()
