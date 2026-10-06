"""RGB, optical-axis depth and camera pose come from the same immutable physics snapshot."""

import asyncio
import time
from fractions import Fraction

import numpy as np
from aiortc import VideoStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame

from ito.protocol import Depth, FrameMetadata


class CameraTrack(VideoStreamTrack):
    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter
        self._id = adapter.description.cameras[0].track_id
        self._origin: float | None = None
        self._last_pts = -90
        self._deadline = 0.0
        self._sequence = 0

    def _capture(self):
        robot = self.adapter
        robot._check_fault()
        with robot._lock:
            robot.mj.mj_copyData(robot._render_data, robot.model, robot.data)
            captured = robot._captured
        data, renderer = robot._render_data, robot._renderer
        # mj_step integrates qpos after computing camera transforms; refresh the snapshot.
        robot.mj.mj_forward(robot.model, data)
        pose = robot.camera_pose(data)
        renderer.update_scene(data, camera=robot.camera_id)
        renderer.disable_depth_rendering()
        rgb = renderer.render()
        if robot.rgb_only:
            return VideoFrame.from_ndarray(rgb, format="rgb24"), captured, None, None
        renderer.enable_depth_rendering()
        metres = renderer.render()
        far = robot.model.vis.map.zfar * robot.model.stat.extent
        valid = np.isfinite(metres) & (metres > 0) & (metres < min(far * 0.999, 65.535))
        mm = np.where(valid, np.clip(np.rint(metres * 1000), 1, 65535), 0).astype("<u2")
        depth = Depth.from_bytes(robot.width, robot.height, mm.tobytes())
        return VideoFrame.from_ndarray(rgb, format="rgb24"), captured, pose, depth

    async def recv(self):
        if self.readyState != "live" or self.adapter._stop.is_set():
            raise MediaStreamError
        await asyncio.sleep(max(0, self._deadline - time.monotonic()))
        self._deadline = time.monotonic() + 1 / self.adapter.fps
        try:
            frame, captured, pose, depth = await asyncio.get_running_loop().run_in_executor(
                self.adapter._executor, self._capture
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.adapter._fault = exc
            self.adapter.neutral()
            self.stop()
            raise MediaStreamError from exc
        if self.readyState != "live" or self.adapter._stop.is_set():
            raise MediaStreamError
        if self._origin is None:
            self._origin = captured
        # Millisecond ticks survive FFmpeg's internal time-base rescaling exactly.
        pts = max(self._last_pts + 90, round((captured - self._origin) * 1000) * 90)
        self._last_pts = pts
        frame.pts, frame.time_base = pts, Fraction(1, 90000)
        metadata = FrameMetadata(
            camera=self.adapter.camera_name,
            sequence=self._sequence,
            capture_time=captured,
            video_pts=pts,
            camera_pose=pose,
            depth=depth,
        )
        self.adapter.publish_frame(metadata)
        self._sequence += 1
        return frame
