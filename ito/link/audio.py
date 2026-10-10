"""20 ms mono audio; bounded device queues discard backlog instead of adding latency."""

import asyncio
import atexit
import contextlib
import logging
import queue
import sys
import threading
import wave
from collections import deque
from fractions import Fraction

import numpy as np
from aiortc import MediaStreamTrack, RTCRtpSender
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame, AudioResampler

from ito import clock, diagnostics

RATE, SAMPLES = 48000, 960
CLOSE_TIMEOUT = 2.0  # A device that takes longer is left to close on its own thread.
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
        now = clock.now()
        self.deadline = max(self.deadline or now, now - 0.02)
        await clock.sleep_until(self.deadline)
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
        self.capture, self.playback = deque(maxlen=2), deque(maxlen=5)
        self.buffering = True
        self.counters = dict(
            received=0,
            played=0,
            underruns=0,
            device_underflows=0,
            capture_overflows=0,
            dropped=0,
            rebuffers=0,
        )
        self.mic_muted = self.speaker_muted = False
        self.input_status = self.output_status = "starting"
        self.streams = []
        self.writer = None
        self.task = None
        self.jobs = None
        self.close_timeout = CLOSE_TIMEOUT
        self.track = Microphone(self)

    @property
    def status(self):
        return f"Audio: mic {self.input_status} | speaker {self.output_status}"

    def _open(self, capture, playback):
        choices = (self.source if capture else "none", self.sink if playback else "none")
        sd = None
        if "device" in choices:
            try:
                import sounddevice as sd
            except (ImportError, OSError):
                pass
        for incoming, choice in zip((True, False), choices, strict=True):
            status = "off" if choice == "none" else "ready"
            stream = None
            try:
                if choice == "device":
                    if sd is None:
                        raise RuntimeError("PortAudio unavailable")
                    if incoming:

                        def capture(data, frames, timing, flags):
                            self.counters["capture_overflows"] += int(flags.input_overflow)
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
                            self._play(data, flags)

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

    def _play(self, data, flags):
        # Network delivery is bursty even after RTP reordering. Keep 60 ms ahead of
        # the device clock, with a hard 100 ms ceiling and re-prime after a gap.
        self.counters["device_underflows"] += int(flags.output_underflow)
        data.fill(0)
        if self.speaker_muted:
            self.playback.clear()
            self.buffering = True
            return
        if self.buffering:
            if len(self.playback) < 3:
                return
            self.buffering = False
        try:
            data[:, 0] = self.playback.popleft()
            self.counters["played"] += 1
        except IndexError:
            self.counters["underruns"] += 1
            self.counters["rebuffers"] += 1
            self.buffering = True

    def start(self, *, capture=True, playback=True):
        """Queue opening the source and sink; await the result to wait for the devices."""
        return self._io(self._open, capture, playback)

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
                    self.counters["received"] += 1
                    if self.sink == "device":
                        self.counters["dropped"] += int(len(self.playback) == self.playback.maxlen)
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

    def _io(self, function, *args):
        """Device and file I/O run in order on this Audio's own thread.

        PortAudio opens and closes a stream on the same thread, a close always follows the
        open it undoes, and a device that blocks never stalls the event loop or the
        executor asyncio.run joins on shutdown.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        if self.jobs is None:
            self.jobs = queue.SimpleQueue()
            threading.Thread(target=_work, args=(self.jobs,), name="ito-audio", daemon=True).start()
        self.jobs.put((function, args, loop, future))
        return future

    async def close(self):
        self.track.stop()
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None
        if self.jobs is None:
            return
        closed = self._io(self._close)
        self.jobs.put(None)
        self.jobs = None
        try:
            async with asyncio.timeout(self.close_timeout):
                with diagnostics.stage("audio_devices"):
                    await asyncio.shield(closed)
        except TimeoutError:
            diagnostics.event("audio_close_timeout", timeout_s=self.close_timeout)
            log.warning("Audio devices still closing after %.1f s; continuing", self.close_timeout)
            # sounddevice's exit handler would wait on the same device and keep a closed
            # Ito running; the operating system releases the device with the process.
            if sd := sys.modules.get("sounddevice"):
                atexit.unregister(sd._exit_handler)

    def _close(self):
        for _, stream in self.streams:
            try:
                # PortAudio can hang closing a running stream; stop it first.
                stream.stop()
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


def _work(jobs):
    while (job := jobs.get()) is not None:
        function, args, loop, future = job
        try:
            result = function(*args)
        except Exception as exc:
            settle = future.set_exception
            result = exc
        else:
            settle = future.set_result

        def done(settle=settle, result=result, future=future):
            if not future.done():
                settle(result)

        with contextlib.suppress(RuntimeError):  # The loop may be gone after a timed-out close.
            loop.call_soon_threadsafe(done)
