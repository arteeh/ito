"""Poses for the ToF mesh: interpolation of the robot's own camera pose stream.

All poses are 4x4 world-from-camera in Ito's axes (+X right, +Y up, looking down -Z), the world
anchored where the robot started (the driver's odometry frame, y up from the floor).
"""

import math
from collections import deque

import numpy as np


def quaternion(rotation):
    """xyzw unit quaternion of a 3x3 rotation."""
    m = np.asarray(rotation, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = (s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s)
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = ((m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s)
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = ((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4, (m[1, 0] - m[0, 1]) / s)
    q = np.array(q)
    return q / np.linalg.norm(q)


def rotation(q):
    """3x3 rotation of an xyzw quaternion."""
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def slerp(a, b, t):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    dot = float(a @ b)
    if dot < 0:
        b, dot = -b, -dot
    if dot > 0.9995:
        q = a + t * (b - a)
        return q / np.linalg.norm(q)
    angle = math.acos(dot)
    return (math.sin((1 - t) * angle) * a + math.sin(t * angle) * b) / math.sin(angle)


def angle_between(a, b):
    """Rotation angle (radians) between two 3x3 rotations."""
    c = (np.trace(a.T @ b) - 1) / 2
    return math.acos(max(-1.0, min(1.0, c)))


class PoseTrack:
    """The robot's camera pose stream (driver Motion samples on the pilot clock), interpolated
    to any time inside it. Bounded: only the last `seconds` are kept."""

    def __init__(self, seconds=5.0):
        self.seconds = seconds
        self.samples = deque()

    def add(self, when, world_from_camera):
        m = np.asarray(world_from_camera, dtype=np.float64)
        if self.samples and when <= self.samples[-1][0]:
            return
        self.samples.append((when, m[:3, 3].copy(), quaternion(m[:3, :3])))
        while self.samples and when - self.samples[0][0] > self.seconds:
            self.samples.popleft()

    @property
    def latest(self):
        return self.samples[-1][0] if self.samples else None

    def at(self, when, slack=0.1):
        """Interpolated pose at `when`, or None outside the stream (beyond `slack` seconds of
        its ends, where it holds the end pose)."""
        if not self.samples:
            return None
        first, last = self.samples[0], self.samples[-1]
        if when < first[0] - slack or when > last[0] + slack:
            return None
        if when <= first[0]:
            return self._matrix(first[1], first[2])
        if when >= last[0]:
            return self._matrix(last[1], last[2])
        lo, hi = 0, len(self.samples) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.samples[mid][0] <= when:
                lo = mid
            else:
                hi = mid
        (ta, pa, qa), (tb, pb, qb) = self.samples[lo], self.samples[hi]
        t = (when - ta) / (tb - ta)
        return self._matrix(pa + t * (pb - pa), slerp(qa, qb, t))

    @staticmethod
    def _matrix(position, q):
        m = np.eye(4)
        m[:3, :3] = rotation(q)
        m[:3, 3] = position
        return m
