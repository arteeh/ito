"""Video-to-metadata joins: every frame gets its exposure's pose; only its own depth."""

import bisect
import math

import numpy as np

from ito import clock, diagnostics
from ito.protocol import Pose
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


TICKS = 90000  # FrameMetadata.video_pts runs at 90 kHz
SLACK = 1  # RTP's pts round trip can land a decoded frame one tick off its metadata
NEAREST = round(0.034 * TICKS)  # one 30 fps frame: a pose this close is the exposure's pose
BRIDGE = round(0.25 * TICKS)  # interpolate across lost metadata up to this span
KEEP = 64


def slerp(q0, q1, f):
    dot = sum(a * b for a, b in zip(q0, q1, strict=True))
    if dot < 0:
        q1, dot = tuple(-v for v in q1), -dot
    if dot > 0.9995:
        q = tuple(a + (b - a) * f for a, b in zip(q0, q1, strict=True))
    else:
        theta = math.acos(dot)
        s0, s1 = math.sin((1 - f) * theta), math.sin(f * theta)
        q = tuple(a * s0 + b * s1 for a, b in zip(q0, q1, strict=True))
    n = math.sqrt(sum(v * v for v in q))
    return tuple(v / n for v in q)


def _lerp(a, b, f):
    return tuple(x + (y - x) * f for x, y in zip(a, b, strict=True))


def between(m0, m1, k0, k1, key):
    """Metadata for the exposure at `key`, between the exposures at k0 and k1.

    Pose and gaze are interpolated; depth belongs to one exposure only, so there is none.
    """
    f = (key - k0) / (k1 - k0)
    near = m0 if f < 0.5 else m1
    pose = None
    if m0.camera_pose is not None and m1.camera_pose is not None:
        pose = Pose(
            position=_lerp(m0.camera_pose.position, m1.camera_pose.position, f),
            orientation=slerp(m0.camera_pose.orientation, m1.camera_pose.orientation, f),
        )
    elif near.camera_pose is not None:
        pose = near.camera_pose
    head = None
    if m0.head_angles is not None and m1.head_angles is not None:
        head = _lerp(m0.head_angles, m1.head_angles, f)
    yaw = m0.body_yaw + math.remainder(m1.body_yaw - m0.body_yaw, math.tau) * f
    return near.model_copy(
        update=dict(
            capture_time=m0.capture_time + (m1.capture_time - m0.capture_time) * f,
            video_pts=key,
            camera_pose=pose,
            head_angles=head,
            body_yaw=math.remainder(yaw, math.tau),
            depth=None,
        )
    )


class FrameJoin:
    """Pairs each decoded video frame with what the robot measured at its exposure.

    Metadata sits in a small pts-sorted ring; a frame looks itself up by bisection. Its own
    metadata (within a tick) joins whole, depth included. Otherwise its pose comes from the
    metadata either side, interpolated, or the nearest within one frame; failing that it
    still reaches the pilot, with its capture time and heading but no measured pose. A frame
    that arrives before any later metadata waits for one, at most until the next frame.
    """

    def __init__(self):
        self.keys = []  # sorted metadata pts
        self.metadata = {}  # pts -> (received, FrameMetadata)
        self.waiting = None  # (key, frame) for a frame with no later metadata yet
        self.last_capture = -1.0
        self.joins = {"exact": 0, "between": 0, "nearest": 0, "bare": 0, "dropped": 0}

    def decoded(self, frame):
        """Pairs (frame, metadata) now ready, oldest first."""
        if frame.pts is None or frame.time_base is None:
            return []
        key = round(frame.pts * frame.time_base * TICKS)
        out = []
        if self.waiting is not None:
            self._emit(out, self._resolve(*self.waiting, force=True))
            self.waiting = None
        result = self._resolve(key, frame, force=False)
        if result is None:
            self.waiting = (key, frame)
        else:
            self._emit(out, result)
        return out

    def described(self, metadata):
        if metadata.video_pts is None:
            raise ValueError("Camera metadata needs video_pts to match RGB and depth")
        key = metadata.video_pts
        now = clock.now()
        if key not in self.metadata:
            if not self.keys or key > self.keys[-1]:
                self.keys.append(key)
            else:
                bisect.insort(self.keys, key)
        self.metadata[key] = (now, metadata)
        while len(self.keys) > KEEP or now - self.metadata[self.keys[0]][0] > 2:
            del self.metadata[self.keys.pop(0)]
        out = []
        if self.waiting is not None and key >= self.waiting[0] - SLACK:
            result = self._resolve(*self.waiting, force=False)
            if result is not None:
                self.waiting = None
                self._emit(out, result)
        return out

    def _resolve(self, key, frame, force):
        keys = self.keys
        i = bisect.bisect_left(keys, key - SLACK)
        if i < len(keys) and keys[i] <= key + SLACK:
            return "exact", frame, self.metadata[keys[i]][1], keys[i]
        after = keys[i] if i < len(keys) else None
        before = keys[i - 1] if i else None
        if after is None and not force:
            return None
        if before is not None and after is not None and after - before <= BRIDGE:
            m0, m1 = self.metadata[before][1], self.metadata[after][1]
            return "between", frame, between(m0, m1, before, after, key), before
        near = min(
            (k for k in (before, after) if k is not None), key=lambda k: abs(k - key), default=None
        )
        if near is None:
            return "dropped", frame, None, None
        metadata = self.metadata[near][1]
        update = dict(
            capture_time=metadata.capture_time + (key - near) / TICKS, video_pts=key, depth=None
        )
        if abs(near - key) > NEAREST:
            # Too far from any measurement to claim its pose; keep heading as SLAM's hint.
            update.update(camera_pose=None, head_angles=None)
            return "bare", frame, metadata.model_copy(update=update), before
        return "nearest", frame, metadata.model_copy(update=update), before

    def _emit(self, out, result):
        kind, frame, metadata, used = result
        self.joins[kind] += 1
        if metadata is None:
            diagnostics.event("frame_unjoined", interval=1, dropped=self.joins["dropped"])
            return
        if metadata.capture_time <= self.last_capture:
            diagnostics.event("frame_out_of_order", interval=1, sequence=metadata.sequence)
            return
        diagnostics.event(
            "frame_join",
            interval=1,
            kind=kind,
            sequence=metadata.sequence,
            capture_time=metadata.capture_time,
            video_pts=metadata.video_pts,
            **self.joins,
        )
        self.last_capture = metadata.capture_time
        # Keep the newest metadata at or before this frame: the next frame's "before".
        if used is not None:
            drop = bisect.bisect_left(self.keys, used)
            for old in self.keys[:drop]:
                del self.metadata[old]
            del self.keys[:drop]
        out.append((frame, metadata))

    @staticmethod
    def arrays(pair, clock):
        frame, metadata = pair
        depth = metadata.depth
        return (
            frame.to_ndarray(format="rgb24"),
            np.frombuffer(depth.to_bytes(), dtype="<u2")
            .reshape(depth.height, depth.width)
            .astype(np.float32)
            * np.float32(0.001)
            if depth is not None
            else None,
            camera_matrix(metadata.camera_pose) if metadata.camera_pose is not None else None,
            clock.remote_to_local(metadata.capture_time),
        )
