"""Rectify mediad's calibrated sensor image before exposing a pinhole camera to Ito."""

import asyncio
from fractions import Fraction

import av
import numpy as np
from aiortc import VideoStreamTrack
from aiortc.mediastreams import MediaStreamError

from drivers.microduck import lens
from ito import clock
from ito.protocol import FrameMetadata, Intrinsics


class Camera:
    def __init__(self, info, measured=dict):
        """Measured returns the robot's FrameMetadata fields for a frame arriving now."""
        import cv2

        self.cv = cv2
        cv2.setNumThreads(1)
        geometry = info.get("intrinsics")
        if not geometry or geometry.get("source") not in {"robot", "family", "sim"}:
            raise RuntimeError("Microduck camera needs calibrated media.video intrinsics")
        self.width, self.height = info["width"], info["height"]
        k = Intrinsics(
            width=self.width,
            height=self.height,
            **{key: geometry[key] for key in ("fx", "fy", "cx", "cy")},
        )
        if geometry["source"] == "sim":
            # mediad describes the twin as MuJoCo's default 45 degrees; sim_body renders the lens.
            k = k.model_copy(
                update={"fx": lens.focal_px(self.width), "fy": lens.focal_px(self.width)}
            )
        self.rotation = info["rotate"]
        if self.rotation not in {0, 90, 180, 270}:
            raise ValueError("Microduck camera mount must be a quarter turn")
        matrix = np.array([[k.fx, 0, k.cx], [0, k.fy, k.cy], [0, 0, 1]])
        distortion = np.array(geometry.get("distortion") or [0.0] * 5)
        if distortion.shape not in {(4,), (5,), (8,), (12,), (14,)}:
            raise ValueError("unsupported Microduck camera distortion")
        if not np.isfinite(distortion).all():
            raise ValueError("non-finite Microduck camera distortion")
        self.maps = (
            cv2.initUndistortRectifyMap(
                matrix,
                distortion,
                None,
                matrix,
                (k.width, k.height),
                cv2.CV_32FC1,
            )
            if np.any(distortion)
            else None
        )
        for _ in range(self.rotation // 90):
            k = Intrinsics(
                width=k.height, height=k.width, fx=k.fy, fy=k.fx, cx=k.height - 1 - k.cy, cy=k.cx
            )
        self.intrinsics = k
        self.source = geometry["source"]
        self.measured = measured
        self.latest = None
        self.revision = 0
        self.changed = asyncio.Event()
        self.closed = False

    def rectify(self, frame):
        if (frame.width, frame.height) != (self.width, self.height):
            raise RuntimeError("Microduck camera resolution changed; restart the driver")
        rgb = frame.to_ndarray(format="rgb24")
        if self.maps:
            rgb = self.cv.remap(rgb, *self.maps, self.cv.INTER_LINEAR)
        return np.ascontiguousarray(np.rot90(rgb, -(self.rotation // 90)))

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
