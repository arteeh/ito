"""Point the head camera where the pilot looks: the robot.look target, and the pitch trim.

The walking policy tracks head commands with a droop that depends on the gait and the tilt
(about 15 degrees standing, 5 walking forward, -8 backward), so the trim is learned per gait
and per tilt from the camera pitch robotd measures."""

import math

from .frames import normalized, rotate

TRIM_STEP = math.radians(15)
TRIM_BINS = 13  # -90 to +90 degrees.
TRIM_TIME = 0.3
TRIM_LIMIT = math.radians(35)
STEADY = math.radians(2)
SETTLE = 0.3
GAIT_BLEND = 0.3
AT_ONCE = {"backward"}


def gait(forward, left, turn):
    if forward or left:
        return "backward" if forward < 0 else "walking"
    return "turning" if turn else "standing"


def up_in_trunk(trunk):
    w, x, y, z = normalized(trunk)
    return rotate((w, -x, -y, -z), (0.0, 0.0, 1.0))


def target(pan, tilt, up, camera, distance):
    norm = math.sqrt(sum(v * v for v in up))
    if not math.isfinite(norm) or norm < 1e-6:
        raise ValueError("Microduck sent a degenerate up direction")
    up = tuple(v / norm for v in up)
    forward = tuple(a - up[0] * b for a, b in zip((1.0, 0.0, 0.0), up, strict=True))
    norm = math.sqrt(sum(v * v for v in forward))
    forward = tuple(v / norm for v in forward)
    left = (
        up[1] * forward[2] - up[2] * forward[1],
        up[2] * forward[0] - up[0] * forward[2],
        up[0] * forward[1] - up[1] * forward[0],
    )
    along = (math.cos(tilt) * math.cos(pan), math.cos(tilt) * math.sin(pan), math.sin(tilt))
    return tuple(
        c + distance * (along[0] * f + along[1] * s + along[2] * u)
        for c, f, s, u in zip(camera, forward, left, up, strict=True)
    )


class TiltTrim:
    def __init__(self):
        self.tables = {}
        self.bins = None
        self.gait = None
        self.blend = None
        self.tilt = None
        self.anchor = None

    def _place(self, tilt):
        position = min(TRIM_BINS - 1.0, max(0.0, tilt / TRIM_STEP + (TRIM_BINS - 1) / 2))
        low = min(TRIM_BINS - 2, int(position))
        return low, position - low

    def _weight(self, now):
        return 1.0 if self.blend is None else min(1.0, (now - self.blend[1]) / GAIT_BLEND)

    def __call__(self, tilt, now, gait):
        if gait != self.gait:
            if self.bins is not None:
                w = self._weight(now)
                start = self.blend[0] if self.blend else self.bins
                self.blend = (
                    [a * (1 - w) + b * w for a, b in zip(start, self.bins, strict=True)],
                    now,
                )
            if gait in AT_ONCE:
                self.blend = None
            self.gait, self.bins = gait, self.tables.setdefault(gait, [0.0] * TRIM_BINS)
            self.anchor = (tilt, now)
        elif self.anchor is None or abs(tilt - self.anchor[0]) > STEADY:
            self.anchor = (tilt, now)
        self.tilt = tilt
        low, f = self._place(tilt)
        trim = self.bins[low] * (1 - f) + self.bins[low + 1] * f
        w = self._weight(now)
        if w < 1:
            start = self.blend[0]
            trim = (start[low] * (1 - f) + start[low + 1] * f) * (1 - w) + trim * w
        return tilt + trim

    def release(self):
        self.tilt = self.anchor = None

    def measured(self, pitch, now, dt):
        if self.anchor is None or now - self.anchor[1] < SETTLE or self._weight(now) < 1:
            return
        error = self.tilt - pitch
        low, f = self._place(self.tilt)
        step = error * min(dt, 0.1) / TRIM_TIME / ((1 - f) ** 2 + f**2)
        for index, weight in ((low, 1 - f), (low + 1, f)):
            value = self.bins[index] + weight * step
            self.bins[index] = min(TRIM_LIMIT, max(-TRIM_LIMIT, value))
