import asyncio

import av
import numpy as np
from aiortc.mediastreams import MediaStreamError

from ito import clock
from ito.link.media import RTP_TIME_BASE, EncodedVideoTrack
from ito.protocol import FrameMetadata, Intrinsics, Lens

# RTP's video clock (RFC 6184) and how long its 32-bit timestamps take to wrap.
RTP_RATE = 90_000
RTP_WRAP = 2**32 / RTP_RATE
# A frame's capture time is believed only if it puts the frame at most this long before it arrived
# (and not after): anything else is a pipeline that restarted its clock, and the frame falls back
# to its arrival.
LONGEST_DELAY = 1.0
# Encoded frames waiting for one pilot's link: past this many it has fallen behind, and starts
# again from the next keyframe.
BACKLOG = 8

# The widest pinhole Ito is given, degrees across: a pinhole's edges stretch as 1 / cos^2 of their
# angle, so a wider one spends its pixels on smeared edges. The beta's lens (about 114
# across, 60 down) holds a 16:9 pinhole only 91.6 wide anyway (its 60 down binds); whichever is
# narrower wins.
MAX_RECTIFIED_HFOV_DEG = 100.0


def keyframe(data):
    """Whether an H.264 access unit (Annex B) holds an IDR slice."""
    start = data.find(b"\x00\x00\x01")
    while 0 <= start < len(data) - 3:
        if data[start + 3] & 0x1F == 5:
            return True
        start = data.find(b"\x00\x00\x01", start + 3)
    return False


def rectified(matrix, distortion, width, height):
    """The camera matrix of the widest pinhole of the frame's size whose every pixel the distorted
    frame covers (or the frame's own matrix when it has no distortion), which Ito straightens the
    frames to. It keeps the calibration's principal point and pixel aspect and scales only the
    focal length: centring it would throw away twice the offset on one side (an alpha's cy 474 of
    720 would lose a third of the view down). Each pixel's centre must map within the sent frame's
    outer pixel centres, so bilinear sampling never mixes in its black border."""
    import cv2

    if not np.any(distortion):
        return matrix
    cx, cy = matrix[0, 2], matrix[1, 2]
    if not (0 < cx < width - 1 and 0 < cy < height - 1):
        raise ValueError("the Microduck camera's principal point is outside its frame")
    columns, rows = np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)
    edge = np.concatenate(
        [
            np.stack([columns, np.zeros(width)], -1),
            np.stack([columns, np.full(width, height - 1.0)], -1),
            np.stack([np.zeros(height), rows], -1),
            np.stack([np.full(height, width - 1.0), rows], -1),
        ]
    )
    # CV_16SC2 maps round to 1/32 pixel: keep that much inside the outer pixel centres.
    margin = 1 / 32

    def matrix_at(scale):
        return np.array(
            [[matrix[0, 0] * scale, 0, cx], [0, matrix[1, 1] * scale, cy], [0, 0, 1]], np.float64
        )

    def inside(scale):
        k = matrix_at(scale)
        rays = np.concatenate(
            [(edge - (cx, cy)) / (k[0, 0], k[1, 1]), np.ones((len(edge), 1))], -1
        ).reshape(-1, 1, 3)
        sent, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), matrix, distortion)
        sent = sent.reshape(-1, 2)
        return bool(
            np.all((sent[:, 0] >= margin) & (sent[:, 0] <= width - 1 - margin))
            and np.all((sent[:, 1] >= margin) & (sent[:, 1] <= height - 1 - margin))
        )

    # The cap is on the whole width, however the principal point splits it.
    def across(scale):
        fx = matrix[0, 0] * scale
        return np.degrees(np.arctan((cx + 0.5) / fx) + np.arctan((width - 0.5 - cx) / fx))

    low, high = 1e-3, 1.0
    while across(low) < MAX_RECTIFIED_HFOV_DEG:
        low /= 2
    while not inside(high):
        high *= 2
        if high > 64:
            raise ValueError("the Microduck camera's distortion does not cover its own frame")
    for _ in range(60):
        middle = (low + high) / 2
        if inside(middle) and across(middle) <= MAX_RECTIFIED_HFOV_DEG:
            high = middle
        else:
            low = middle
    return matrix_at(high)


class Camera:
    """The head camera's H.264, as mediad encoded it, each frame with when it was captured.

    The frames go on to Ito as they came, with the lens they were taken through, and Ito
    straightens them on its GPU: decoding and encoding them again here would cost the robot's
    CPU and a frame of delay. mediad starts every RTP timestamp at zero on CLOCK_MONOTONIC's
    `rtp.zero_ns` (its pipeline's base time), and a camera stack that stamps its own frames
    (V4L2's buffer timestamps on a robot; the simulator's exposures) makes that the capture: so
    a frame's RTP timestamp is when it was captured, on the robot's clock. aiortc hands out
    `frame.pts` as the RTP timestamp minus its first frame's (`rtp_origin()`), and
    `to_local(t_ns)` puts a robot time on this driver's clock (the adapter's SourceClock). A
    mediad that says nothing about RTP time, or whose frames are stamped on arrival, leaves only
    the receive time and the driver's FRAME_DELAY guess.
    """

    def __init__(
        self,
        info,
        measured=dict,
        to_local=None,
        asked_at=None,
        fallback_delay=0.0,
        rtp_origin=lambda: None,
        keyframe=lambda: None,
    ):
        geometry = info.get("intrinsics")
        if not geometry or geometry.get("source") not in {"robot", "family", "sim"}:
            # mediad publishes no geometry it cannot vouch for (a beta has had a family solve
            # since microduck 0.18), and every direction Ito drew from a guessed lens would be
            # quietly wrong.
            raise RuntimeError(
                "the Microduck publishes no camera geometry (media.video has no intrinsics): "
                "calibrate its camera (duckslam calib intrinsics) and write the solve to "
                "[media.intrinsics] in its robotd.toml, then restart mediad"
            )
        self.width, self.height = info["width"], info["height"]
        if info.get("rotate", 0) != 0:
            raise RuntimeError("Ito supports the Microduck beta, whose camera is upright")
        matrix = np.array(
            [[geometry["fx"], 0, geometry["cx"]], [0, geometry["fy"], geometry["cy"]], [0, 0, 1]],
            np.float64,
        )
        # OpenCV's order, k1 k2 p1 p2 k3 and on, as microduck's [media.intrinsics] carries it.
        distortion = np.array(geometry.get("distortion") or [0.0] * 5, np.float64)
        if distortion.shape not in {(4,), (5,), (8,), (12,), (14,)}:
            raise ValueError("unsupported Microduck camera distortion")
        if not np.isfinite(distortion).all():
            raise ValueError("non-finite Microduck camera distortion")
        if len(distortion) == 14:
            if np.any(distortion[12:]):
                raise ValueError("Ito does not straighten a tilted Microduck camera sensor")
            distortion = distortion[:12]
        self.rectified = rectified(matrix, distortion, self.width, self.height)
        self.lens = (
            Lens(
                fx=float(matrix[0, 0]),
                fy=float(matrix[1, 1]),
                cx=float(matrix[0, 2]),
                cy=float(matrix[1, 2]),
                distortion=tuple(float(k) for k in distortion),
            )
            if np.any(distortion)
            else None
        )
        self.intrinsics = Intrinsics(
            width=self.width,
            height=self.height,
            fx=float(self.rectified[0, 0]),
            fy=float(self.rectified[1, 1]),
            cx=float(self.rectified[0, 2]),
            cy=float(self.rectified[1, 2]),
        )
        self.source = geometry["source"]
        rtp = info.get("rtp") or {}
        self.rtp_zero = (
            rtp.get("zero_ns")
            if to_local is not None
            and rtp.get("stamps") == "capture"
            and rtp.get("clock_rate") == RTP_RATE
            and isinstance(rtp.get("zero_ns"), int)
            else None
        )
        # The robot's CLOCK_MONOTONIC minus ours, roughly (the answer's travel time): enough to
        # pick which wrap of the RTP clock a frame is in, never used for the time itself.
        mono = info.get("mono_ns")
        self.robot_ahead = (
            mono / 1e9 - asked_at if isinstance(mono, int) and asked_at is not None else 0.0
        )
        self.to_local = to_local
        self.rtp_origin = rtp_origin  # aiortc's frame.pts is the RTP timestamp minus this
        # With capture times promised, a frame without a believable one is put this long before
        # its arrival, so it still means "captured" like the others.
        self.fallback_delay = fallback_delay
        self.stamped = 0  # frames with a believable capture time
        self.unstamped = 0  # frames whose stamp was missing or not believable
        self.measured = measured
        self.keyframe = keyframe  # asks mediad for a keyframe
        self.frames = 0
        self.tracks = set()
        self.closed = False

    async def receive(self, track):
        """mediad's encoded frames (Passthrough's av.Packets), on to every pilot's track."""
        while True:
            try:
                async with asyncio.timeout(2):
                    packet = await track.recv()
            except TimeoutError as exc:
                raise RuntimeError("Microduck camera stalled for two seconds") from exc
            received = clock.now()
            captured = self.captured(packet, received)
            if captured is None and self.exact:
                captured = received - self.fallback_delay
            data = bytes(packet)
            frame = (
                data,
                received if captured is None else captured,
                self.measured(captured),
                keyframe(data),
            )
            self.frames += 1
            for pilot in list(self.tracks):
                pilot.put(frame)

    @property
    def exact(self):
        """Whether frames carry their capture time (until one turns out not to)."""
        return self.rtp_zero is not None

    def captured(self, frame, received):
        """When `frame` was captured, on this driver's clock, from its RTP timestamp; None
        without a believable one."""
        origin = self.rtp_origin() if self.rtp_zero is not None else None
        if origin is None or frame.pts is None:
            self.unstamped += self.rtp_zero is not None
            return None
        ticks = (int(frame.pts) + origin) % 2**32
        since_zero = ticks / RTP_RATE
        robot_now = received + self.robot_ahead
        wraps = round((robot_now - self.rtp_zero / 1e9 - since_zero) / RTP_WRAP)
        t_ns = self.rtp_zero + round((since_zero + wraps * RTP_WRAP) * 1e9)
        local = self.to_local(t_ns)
        if local is None or not 0.0 <= received - local <= LONGEST_DELAY:
            self.unstamped += 1
            return None
        self.stamped += 1
        return local

    def close(self):
        self.closed = True
        for pilot in list(self.tracks):
            pilot.put(None)


class Track(EncodedVideoTrack):
    """One pilot's camera: mediad's frames as they are, from a keyframe on, each published
    with its capture time and pose."""

    def __init__(self, camera, publish):
        super().__init__()
        self._id = "microduck-rgb"
        self.camera = camera
        self.publish = publish
        self.frames = asyncio.Queue()
        self.started = False  # past the first keyframe: frames before it cannot be decoded
        self.sequence = 0
        self.origin = None
        camera.tracks.add(self)
        camera.keyframe()

    def put(self, frame):
        if frame is not None and self.frames.qsize() >= BACKLOG:
            while not self.frames.empty():
                self.frames.get_nowait()
            self.started = False
            self.camera.keyframe()
        if frame is not None and not self.started:
            if not frame[3]:
                return
            self.started = True
        self.frames.put_nowait(frame)

    def keyframe(self):
        self.camera.keyframe()

    async def recv(self):
        if self.camera.closed or self.readyState != "live":
            raise MediaStreamError
        frame = await self.frames.get()
        if frame is None:
            raise MediaStreamError
        data, captured, measured, _ = frame
        if self.origin is None:
            self.origin = captured
        packet = av.Packet(data)
        packet.pts = round((captured - self.origin) * RTP_RATE)
        packet.time_base = RTP_TIME_BASE
        self.publish(
            FrameMetadata(
                camera="head",
                sequence=self.sequence,
                capture_time=captured,
                video_pts=packet.pts,
                **measured,
            )
        )
        self.sequence += 1
        return packet

    def stop(self):
        super().stop()
        self.camera.tracks.discard(self)
        self.frames.put_nowait(None)
