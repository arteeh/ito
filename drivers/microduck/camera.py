import asyncio
from fractions import Fraction

import av
import numpy as np
from aiortc import VideoStreamTrack
from aiortc.mediastreams import MediaStreamError

from ito import clock
from ito.protocol import FrameMetadata, Intrinsics

# RTP's video clock (RFC 6184) and how long its 32-bit timestamps take to wrap.
RTP_RATE = 90_000
RTP_WRAP = 2**32 / RTP_RATE
# A frame's capture time is believed only if it puts the frame at most this long before it arrived
# (and not after): anything else is a pipeline that restarted its clock, and the frame falls back
# to its arrival.
LONGEST_DELAY = 1.0

# The widest pinhole Ito is given, degrees across: a pinhole's edges stretch as 1 / cos^2 of their
# angle, so a wider one spends its pixels on smeared edges. The beta's lens (about 114
# across, 60 down) holds a 16:9 pinhole only 91.6 wide anyway (its 60 down binds); whichever is
# narrower wins.
MAX_RECTIFIED_HFOV_DEG = 100.0


def rectified(matrix, distortion, width, height):
    """The camera matrix of the widest pinhole of the frame's size whose every pixel the distorted
    frame covers (or the frame's own matrix when it has no distortion). It keeps the calibration's
    principal point and pixel aspect and scales only the focal length: centring it would throw
    away twice the offset on one side (an alpha's cy 474 of 720 would lose a third of the view
    down). Each pixel's centre must map within the sent frame's outer pixel centres, so bilinear
    remap never mixes in its black border."""
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
    """The head camera's frames, rectified, each with when it was captured.

    mediad starts every RTP timestamp at zero on CLOCK_MONOTONIC's `rtp.zero_ns` (its pipeline's
    base time), and a camera stack that stamps its own frames (V4L2's buffer timestamps on a
    robot; the simulator's exposures) makes that the capture: so a frame's RTP timestamp is when
    it was captured, on the robot's clock. aiortc hands out `frame.pts` as the RTP timestamp minus
    its first frame's (`rtp_origin()`), and `to_local(t_ns)` puts a robot time on this driver's
    clock (the adapter's SourceClock). A mediad that says nothing about RTP time, or whose
    frames are stamped on arrival, leaves only the receive time and the driver's FRAME_DELAY guess.
    """

    def __init__(
        self,
        info,
        measured=dict,
        to_local=None,
        asked_at=None,
        fallback_delay=0.0,
        rtp_origin=lambda: None,
    ):
        import cv2

        self.cv = cv2
        cv2.setNumThreads(1)
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
        self.rectified = rectified(matrix, distortion, self.width, self.height)
        self.maps = (
            cv2.initUndistortRectifyMap(
                matrix,
                distortion,
                None,
                self.rectified,
                (self.width, self.height),
                cv2.CV_16SC2,
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
        self.latest = None
        self.revision = 0
        self.changed = asyncio.Event()
        self.closed = False

    def rectify(self, frame):
        """The frame as the pinhole Ito is told about (self.intrinsics)."""
        if (frame.width, frame.height) != (self.width, self.height):
            raise RuntimeError("Microduck camera resolution changed; restart the driver")
        rgb = frame.to_ndarray(format="rgb24")
        if self.maps is not None:
            rgb = self.cv.remap(rgb, *self.maps, self.cv.INTER_LINEAR)
        return np.ascontiguousarray(rgb)

    async def receive(self, track):
        while True:
            try:
                async with asyncio.timeout(2):
                    frame = await track.recv()
            except TimeoutError as exc:
                raise RuntimeError("Microduck camera stalled for two seconds") from exc
            # mediad's WebRTC API exposes no per-frame capture timestamp. This is receive time,
            # not sensor exposure time; keep that distinction visible in telemetry.
            received = clock.now()
            captured = self.captured(frame, received)
            if captured is None and self.exact:
                captured = received - self.fallback_delay
            measured = self.measured(captured)
            rgb = await asyncio.to_thread(self.rectify, frame)
            self.latest = (rgb, received if captured is None else captured, measured)
            self.revision += 1
            self.changed.set()

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
        self.changed.set()


class Track(VideoStreamTrack):
    def __init__(self, camera, publish):
        super().__init__()
        self._id = "microduck-rgb"
        self.camera = camera
        self.publish = publish
        self.revision = 0
        self.sequence = 0
        self.origin = None

    async def recv(self):
        camera = self.camera
        while camera.revision == self.revision:
            if camera.closed or self.readyState != "live":
                raise MediaStreamError
            camera.changed.clear()
            await camera.changed.wait()
        if camera.closed:
            raise MediaStreamError
        self.revision = camera.revision
        rgb, captured, measured = camera.latest
        if self.origin is None:
            self.origin = captured
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = round((captured - self.origin) * 90000)
        frame.time_base = Fraction(1, 90000)
        self.publish(
            FrameMetadata(
                camera="head",
                sequence=self.sequence,
                capture_time=captured,
                video_pts=frame.pts,
                **measured,
            )
        )
        self.sequence += 1
        return frame

    def stop(self):
        super().stop()
        self.camera.changed.set()
