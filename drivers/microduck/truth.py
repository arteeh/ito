"""The virtual Microduck's exact head-camera pose for each frame it sends, a diagnostic.

The sim body sends one JSON line per frame it hands mediad (sim_body.TruthServer): when the
frame was exposed (CLOCK_MONOTONIC seconds, the driver's clock), the camera's pose in
MuJoCo's world and a thumbnail. mediad's H.264 carries no frame id, so the driver finds each
received frame among those sent: the one sent a steady link delay before it (mediad stamps
a frame when it reads it off the sim's socket, so the RTP clock differs from the send clock
by a near-constant), confirmed by its thumbnail.
"""

import base64
import collections
import json
import math
import socket
import threading

import numpy as np

from drivers.microduck import frames

THUMB = (18, 32)  # rows, columns
SLOT = 0.006  # seconds: frames are 33 ms apart; mediad's read jitter is far below this
CONFIRM = 10  # consecutive thumbnail-confirmed matches before the delay is trusted


def thumbnail(rgb):
    """Grey block means of an RGB frame, THUMB in size, uint8."""
    rows, columns = THUMB
    height, width = rgb.shape[0] // rows * rows, rgb.shape[1] // columns * columns
    grey = rgb[:height, :width].astype(np.float32).mean(-1)
    blocks = grey.reshape(rows, height // rows, columns, width // columns).mean((1, 3))
    return np.clip(np.rint(blocks), 0, 255).astype(np.uint8)


def encode(rgb):
    return base64.b64encode(thumbnail(rgb).tobytes()).decode()


def quaternion(matrix):
    """xyzw of a rotation matrix."""
    m = np.asarray(matrix, np.float64)
    w = math.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (x / n, y / n, z / n, w / n)


def in_world(row, origin):
    """Ito world-from-camera (position, xyzw) of a truth row. MuJoCo's world is robotd's (the
    sim's IMU is the trunk's orientation in it), so the origin is the measured pose's."""
    x0, y0, yaw0 = origin
    start = frames.heading(-yaw0)
    p = row["position"]
    world = frames.rotate(start, (p[0] - x0, p[1] - y0, p[2]))
    x, y, z, w = row["orientation"]
    w, x, y, z = frames.normalized(
        frames.multiply(frames.TO_ITO, frames.multiply(start, (w, x, y, z)))
    )
    return frames.rotate(frames.TO_ITO, world), (x, y, z, w)


class SimTruth:
    """Reads the sim's truth lines; match() names the one a received frame is."""

    def __init__(self, address):
        host, port = address.rsplit(":", 1)
        self.sock = socket.create_connection((host, int(port)), timeout=5)
        self.sock.settimeout(None)
        self.sent = collections.deque(maxlen=90)
        self.lock = threading.Lock()
        self.delay = None  # RTP seconds minus send seconds
        self.streak = 0
        self.candidate = None
        threading.Thread(target=self._read, name="sim-truth", daemon=True).start()

    def _read(self):
        with self.sock.makefile("r") as lines:
            for line in lines:
                row = json.loads(line)
                row["thumb"] = np.frombuffer(base64.b64decode(row["thumb"]), np.uint8)
                with self.lock:
                    self.sent.append(row)

    def match(self, rgb, rtp):
        """The truth row of the frame received with this RTP time (seconds), or None."""
        with self.lock:
            rows = list(self.sent)
        if not rows:
            return None
        image = thumbnail(rgb).ravel().astype(np.float32)
        errors = np.mean((np.stack([r["thumb"] for r in rows]) - image) ** 2, axis=1)
        offsets = np.array([rtp - r["sent"] for r in rows])
        if self.delay is not None:
            near = np.flatnonzero(np.abs(offsets - self.delay) < SLOT)
            # Timed, unless the thumbnail clearly prefers another frame.
            if len(near) == 1 and errors[near[0]] <= errors.min() + 2.0:
                i = int(near[0])
                self.delay += 0.02 * (offsets[i] - self.delay)
                return rows[i]
            self.streak = 0
            self.delay = None
        # Unlocked: the thumbnail decides, and only if it is unambiguous.
        order = np.argsort(errors)
        best = int(order[0])
        if len(order) > 1 and errors[order[1]] < errors[best] + 4.0:
            self.streak = 0
            return None
        if self.candidate is not None and abs(offsets[best] - self.candidate) < SLOT:
            self.streak += 1
        else:
            self.streak = 1
        self.candidate = offsets[best]
        if self.streak >= CONFIRM:
            self.delay = offsets[best]
        return rows[best]
