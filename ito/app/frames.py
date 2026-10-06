"""Bounded exact joins: never paint one exposure with another exposure's depth."""

import time

import numpy as np

from ito.render import pose


def camera_matrix(value):
    x, y, z, w = value.orientation
    result = pose(value.position)
    result[:3, :3] = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    return result


class FrameJoin:
    def __init__(self):
        self.video = {}
        self.metadata = {}
        self.last_capture = -1.0

    def add(self, table, key, value):
        now = time.monotonic()
        table[key] = (now, value)
        for pending in (self.video, self.metadata):
            for stamp, (received, _) in list(pending.items()):
                if now - received > 2 or len(pending) > 64:
                    del pending[stamp]
        matches = self.video.keys() & self.metadata.keys()
        if not matches:
            return None
        stamp = max(matches)
        _, frame = self.video.pop(stamp)
        _, metadata = self.metadata.pop(stamp)
        if metadata.capture_time <= self.last_capture:
            return None
        self.last_capture = metadata.capture_time
        for pending in (self.video, self.metadata):
            for old in list(pending):
                if old <= stamp:
                    del pending[old]
        return frame, metadata

    def decoded(self, frame):
        if frame.pts is None or frame.time_base is None:
            return None
        return self.add(self.video, round(frame.pts * frame.time_base * 90000), frame)

    def described(self, metadata):
        if metadata.video_pts is None:
            raise ValueError("Camera metadata needs video_pts to match RGB and depth")
        if metadata.depth is None or metadata.camera_pose is None:
            raise ValueError("Live reconstruction requires posed RGB-D from this driver")
        return self.add(self.metadata, metadata.video_pts, metadata)

    @staticmethod
    def arrays(pair, clock):
        frame, metadata = pair
        depth = metadata.depth
        return (
            frame.to_ndarray(format="rgb24"),
            np.frombuffer(depth.to_bytes(), dtype="<u2")
            .reshape(depth.height, depth.width)
            .astype(np.float32)
            * np.float32(0.001),
            camera_matrix(metadata.camera_pose),
            clock.remote_to_local(metadata.capture_time),
        )
