import asyncio
from fractions import Fraction

import av
import numpy as np
from aiortc import VideoStreamTrack
from aiortc.mediastreams import MediaStreamError

from ito import clock
from ito.protocol import FrameMetadata, Intrinsics

# The widest pinhole Ito is given, degrees across: a pinhole's edges stretch as 1 / cos^2 of their
# angle, so a wider one spends MASt3R's 512 pixels on smeared edges. The beta's lens (116 across,
# 60 down) holds a 16:9 pinhole only 91.4 wide anyway (its 60 down binds), already stretching
# the lens 2.2x at the sides and 2.7x in the corners; whichever is narrower wins.
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
    def __init__(self, info, measured=dict):
        import cv2

        self.cv = cv2
        cv2.setNumThreads(1)
        geometry = info.get("intrinsics")
        if not geometry or geometry.get("source") not in {"robot", "family", "sim"}:
            # mediad publishes no geometry it cannot vouch for. A beta's GC2093 has no family
            # solve (microduck 0.17.4), so a beta that has not been calibrated sends none, and
            # every direction Ito drew from a guessed lens would be quietly wrong.
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
        # OpenCV's order, k1 k2 p1 p2 k3 and on: a beta's lens needs the rational model's 8
        # (lens.py), which microduck's [media.intrinsics] carries as it is.
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
            measured = self.measured()
            rgb = await asyncio.to_thread(self.rectify, frame)
            self.latest = (rgb, received, measured)
            self.revision += 1
            self.changed.set()

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
