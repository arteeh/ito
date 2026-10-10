"""Point the head camera where the pilot looks: the robot.look target, and the pitch trim.

robot.look aims the camera at a trunk-frame point measured from the trunk origin. Placing that
point along the trunk's own axes from the origin points the camera below a level gaze: the
camera sits centimetres above the origin and the trunk leans. Even aimed right, the walking
policy tracks head commands with a droop (it moves the neck itself, and its head command is
only a reward to track), measured at 9 degrees down for a level gaze and up to 15 looking up
in the simulated twin. The trim closes that loop on the camera pitch robotd measures.
"""

import math

from .frames import normalized, rotate

# The trim is learned per pilot tilt, since the droop depends on it: one value every 15 degrees,
# interpolated, so returning to a tilt does not first swing the camera to the last one's trim.
TRIM_STEP = math.radians(15)
TRIM_BINS = 13  # -90 to +90 degrees.
# How fast the trim follows the measured error, and how far it may go: past this the head is
# at its travel and more trim only winds up.
TRIM_TIME = 0.3
TRIM_LIMIT = math.radians(25)
# The trim learns only once the pilot's tilt has held within STEADY for SETTLE: while the
# head is still getting there, the error is lag, not droop.
STEADY = math.radians(2)
SETTLE = 0.3


def up_in_trunk(trunk):
    """The world's up direction in the trunk frame, from the IMU trunk-to-world quaternion."""
    w, x, y, z = normalized(trunk)
    return rotate((w, -x, -y, -z), (0.0, 0.0, 1.0))


def target(pan, tilt, up, camera, distance):
    """The trunk-frame point robot.look aims at for a gaze `pan` left of the body's heading
    and `tilt` above the horizon, seen from the camera at `camera`, `up` being the world's up
    in the trunk frame."""
    norm = math.sqrt(sum(v * v for v in up))
    if not math.isfinite(norm) or norm < 1e-6:
        raise ValueError("Microduck sent a degenerate up direction")
    up = tuple(v / norm for v in up)
    # The trunk's forward axis with its lean removed, and left completing the levelled frame.
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
    """How much higher to aim than the pilot's tilt for the camera to end up at it."""

    def __init__(self):
        self.bins = [0.0] * TRIM_BINS
        self.tilt = None  # The pilot's tilt now, and the one its steady stretch began at.
        self.anchor = None

    def _place(self, tilt):
        position = min(TRIM_BINS - 1.0, max(0.0, tilt / TRIM_STEP + (TRIM_BINS - 1) / 2))
        low = min(TRIM_BINS - 2, int(position))
        return low, position - low

    def __call__(self, tilt, now):
        """The tilt to aim for, given the pilot's tilt now."""
        if self.anchor is None or abs(tilt - self.anchor[0]) > STEADY:
            self.anchor = (tilt, now)
        self.tilt = tilt
        low, f = self._place(tilt)
        return tilt + self.bins[low] * (1 - f) + self.bins[low + 1] * f

    def release(self):
        """The head is no longer aimed: nothing to learn from until it is again."""
        self.tilt = self.anchor = None

    def measured(self, pitch, now, dt):
        """Learn from the camera pitch robotd measured `dt` seconds after the last sample."""
        if self.anchor is None or now - self.anchor[1] < SETTLE:
            return
        error = self.tilt - pitch
        low, f = self._place(self.tilt)
        # Shared between the two bins by their weight in the reading, scaled so the reading
        # moves by the full step wherever the tilt falls between them.
        step = error * min(dt, 0.1) / TRIM_TIME / ((1 - f) ** 2 + f**2)
        for index, weight in ((low, 1 - f), (low + 1, f)):
            value = self.bins[index] + weight * step
            self.bins[index] = min(TRIM_LIMIT, max(-TRIM_LIMIT, value))
