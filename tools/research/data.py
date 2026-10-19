"""A driver recording (drivers/microduck/record.py, ITO_MICRODUCK_RECORD) as a research dataset.

    rec = Recording(path)
    rec.frames            list of dicts: seq, received, exposed, pose (4x4 world-from-camera, Ito
                          axes: +X right, +Y up, looking down -Z), file, depth (file or None)
    rec.rgb(frame, size)  the frame rectified to the pinhole Ito is told (rec.K at 1280x720),
                          optionally resized to size (w, h)
    rec.depth(frame, size) exact axial depth (metres, 0 = none) rectified the same way
    rec.K_at(size)        the pinhole's 3x3 matrix at that size
    rec.motion            (times, 4x4 poses, gyro, states) of the robot's own camera estimate
    rec.sensors           list of Sensors dicts (imu, ranges) as published

World: Ito's, the robot's odometry where it started (x right, y up, z back). OpenCV camera axes
are pose @ AXES."""

import json
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np

AXES = np.diag([1.0, -1.0, -1.0, 1.0])


def quat_matrix(xyzw):
    x, y, z, w = xyzw
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def pose_matrix(position, xyzw):
    m = np.eye(4)
    m[:3, :3] = quat_matrix(xyzw)
    m[:3, 3] = position
    return m


class Recording:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.width, self.height = self.meta["width"], self.meta["height"]
        self.matrix = np.array(self.meta["matrix"], np.float64)
        self.distortion = np.array(self.meta["distortion"], np.float64)
        self.K = np.array(self.meta["rectified"], np.float64)
        self.maps = cv2.initUndistortRectifyMap(
            self.matrix, self.distortion, None, self.K, (self.width, self.height), cv2.CV_32FC1
        )
        frames = []
        for line in (self.path / "frames.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row.get("file") is None:
                continue
            if "position" in row:
                row["pose"] = pose_matrix(row["position"], row["orientation"])
            else:
                row["pose"] = None
            frames.append(row)
        self.frames = frames
        self._depth_maps = {}

    def K_at(self, size):
        w, h = size
        sx, sy = w / self.width, h / self.height
        K = self.K.copy()
        K[0] *= sx
        K[1] *= sy
        # Pixel centres: (c + 0.5) * s - 0.5
        K[0, 2] = (self.K[0, 2] + 0.5) * sx - 0.5
        K[1, 2] = (self.K[1, 2] + 0.5) * sy - 0.5
        return K

    def raw(self, frame):
        bgr = cv2.imread(str(self.path / "frames" / frame["file"]), cv2.IMREAD_COLOR)
        return bgr[..., ::-1]

    def rgb(self, frame, size=None):
        rect = cv2.remap(self.raw(frame), *self.maps, cv2.INTER_LINEAR)
        if size is not None and tuple(size) != (self.width, self.height):
            rect = cv2.resize(rect, tuple(size), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(rect)

    def depth(self, frame, size=None):
        """Exact axial depth (m) rectified to the pinhole at size, nearest sample; None if the
        frame has none."""
        if not frame.get("depth"):
            return None
        mm = cv2.imread(str(self.path / "depth" / frame["depth"]), cv2.IMREAD_UNCHANGED)
        rows, columns = mm.shape
        w, h = size if size is not None else (self.width, self.height)
        key = (rows, columns, w, h)
        if key not in self._depth_maps:
            K = self.K_at((w, h))
            u, v = np.meshgrid(np.arange(w) + 0.0, np.arange(h) + 0.0)
            rays = np.stack(
                [(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1
            ).reshape(-1, 1, 3)
            zero = np.zeros(3)
            sent, _ = cv2.projectPoints(rays, zero, zero, self.matrix, self.distortion)
            sent = sent.reshape(h, w, 2)
            row = np.rint((sent[..., 1] + 0.5) * rows / self.height - 0.5)
            column = np.rint((sent[..., 0] + 0.5) * columns / self.width - 0.5)
            self._depth_maps[key] = (
                np.clip(row, 0, rows - 1).astype(np.intp),
                np.clip(column, 0, columns - 1).astype(np.intp),
            )
        return mm[self._depth_maps[key]].astype(np.float32) / 1000

    @cached_property
    def motion(self):
        times, poses, gyro, states = [], [], [], []
        for line in (self.path / "motion.jsonl").read_text().splitlines():
            row = json.loads(line)
            times.append(row["time"])
            p = row["camera_pose"]
            poses.append(pose_matrix(p["position"], p["orientation"]))
            gyro.append(row["angular_velocity"])
            states.append(row["state"])
        return np.array(times), np.array(poses), np.array(gyro), states

    @cached_property
    def sensors(self):
        return [json.loads(line) for line in (self.path / "sensors.jsonl").read_text().splitlines()]

    def truth_frames(self):
        return [f for f in self.frames if f["pose"] is not None]

    def between(self, start=None, end=None, every=1):
        """Frames with truth received between start and end seconds after the first frame."""
        t0 = self.frames[0]["received"]
        out = [
            f
            for f in self.truth_frames()
            if (start is None or f["received"] - t0 >= start)
            and (end is None or f["received"] - t0 <= end)
        ]
        return out[::every]
