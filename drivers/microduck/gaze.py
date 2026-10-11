"""Point the head camera where the pilot looks: the robot.look target, and the pitch trim.

robot.look aims the camera at a trunk-frame point measured from the trunk origin. Placing that
point along the trunk's own axes from the origin points the camera below a level gaze: the
camera sits centimetres above the origin and the trunk leans. Even aimed right, the walking
policy tracks head commands with a droop (it moves the neck itself, and its head command is
only a reward to track), measured at 9 degrees down for a level gaze and up to 15 looking up
in the simulated twin. The trim closes that loop on the camera pitch robotd measures.
The droop is the gait's: about 15 degrees standing, 5 walking forward and -8 walking
backward for a level gaze, so each gait keeps its own trim. One trim relearning after every
start and stop left the camera 14 degrees low for a second after a backward walk stopped.
A walking gait's trim starts from the standing one, so the first walk of a session is close,
and learns only the gait's steady droop, never the neck's swing as the gait begins.
"""

import math

from .frames import normalized, rotate

# The trim is learned per pilot tilt, since the droop depends on it: one value every 15 degrees,
# interpolated, so returning to a tilt does not first swing the camera to the last one's trim.
TRIM_STEP = math.radians(15)
TRIM_BINS = 13  # -90 to +90 degrees.
# How fast the trim follows the measured error, and how far it may go: past this the head is
# at its travel and more trim only winds up. Walking gaits learn slower: their sway swings the
# camera +-3 degrees every half second, and a fast trim chasing it swung the camera further.
TRIM_TIME = 0.3
WALK_TRIM_TIME = 1.5
TRIM_LIMIT = math.radians(25)
# Until a walking gait has learned its own trim at a tilt, it uses the standing trim there plus
# the gait's usual difference from it: in the simulated twin, for a level gaze, walking forward
# settles 9 degrees below standing and backward 21 (live backward 20-24). A backward trim that
# started at zero left the camera 10-14 degrees high through the first walk-back (Ceres, #28).
GAIT_OFFSET = {"walking": math.radians(-9), "backward": math.radians(-20)}
# After a gait change the policy swings the neck into the new posture and past it for about
# a second (walking back: up 17 degrees, then the head overshoots its own target). Learning
# that swing taught the backward trim a dip of 8-10 degrees a second into every later walk;
# walking learns nothing until GAIT_SETTLE after a gait change. Standing keeps its quick
# correction so a stop does not leave the camera dipping.
GAIT_SETTLE = 1.2
# The trim learns only once the pilot's tilt has held within STEADY, and the gait has not
# changed, for SETTLE: while the head is still getting there, the error is lag, not droop.
STEADY = math.radians(2)
SETTLE = 0.3
# The policy moves the neck into a new gait's posture within about half a second. The trim
# blends from the old gait's to the new one's over GAIT_BLEND so the head moves with the neck,
# not ahead of it (switching at once left the camera 11 degrees high), and learns nothing
# until the blend is done.
GAIT_BLEND = 0.3
# Gaits whose trim applies the moment the move command asks for them. Walking backward, the
# policy lifts the neck within two or three state updates, past its own target; a trim
# blending down from the standing one (+15 degrees for a level gaze) kept the head aimed high
# meanwhile, and the camera rose 13-16 degrees on Ceres (pitch.jsonl, round F).
AT_ONCE = {"backward"}
WALKING = ("walking", "backward")


def gait(forward, left, turn):
    """The gait the robot walks for a move command, whose head droop its trim learns."""
    if forward or left:
        return "backward" if forward < 0 else "walking"
    return "turning" if turn else "standing"


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
        self.tables = {}  # Per gait, the trim learned at each tilt bin; None where not yet.
        self.gait = None
        self.changed = None  # When the gait last changed.
        self.blend = None  # The trims blended from since the gait changed, and when it did.
        self.tilt = None  # The pilot's tilt now, and the one its steady stretch began at.
        self.anchor = None

    def _place(self, tilt):
        position = min(TRIM_BINS - 1.0, max(0.0, tilt / TRIM_STEP + (TRIM_BINS - 1) / 2))
        low = min(TRIM_BINS - 2, int(position))
        return low, position - low

    def _bin(self, gait, index):
        """A gait's trim at a tilt bin: learned, or the standing one plus the gait's offset."""
        learned = self.tables.setdefault(gait, [None] * TRIM_BINS)[index]
        if learned is not None:
            return learned
        if gait in GAIT_OFFSET:
            value = self._bin("standing", index) + GAIT_OFFSET[gait]
            return min(TRIM_LIMIT, max(-TRIM_LIMIT, value))
        return 0.0

    def _weight(self, now):
        return 1.0 if self.blend is None else min(1.0, (now - self.blend[1]) / GAIT_BLEND)

    def __call__(self, tilt, now, gait):
        """The tilt to aim for, given the pilot's tilt and the robot's gait now."""
        if gait != self.gait:
            if self.gait is not None:
                w = self._weight(now)
                previous = [self._bin(self.gait, i) for i in range(TRIM_BINS)]
                start = self.blend[0] if self.blend else previous
                self.blend = (
                    [a * (1 - w) + b * w for a, b in zip(start, previous, strict=True)],
                    now,
                )
            if gait in AT_ONCE:
                self.blend = None
            self.gait, self.changed = gait, now
            self.anchor = (tilt, now)
        elif self.anchor is None or abs(tilt - self.anchor[0]) > STEADY:
            self.anchor = (tilt, now)
        self.tilt = tilt
        low, f = self._place(tilt)
        trim = self._bin(gait, low) * (1 - f) + self._bin(gait, low + 1) * f
        w = self._weight(now)
        if w < 1:
            start = self.blend[0]
            trim = (start[low] * (1 - f) + start[low + 1] * f) * (1 - w) + trim * w
        return tilt + trim

    def release(self):
        """The head is no longer aimed: nothing to learn from until it is again."""
        self.tilt = self.anchor = None

    def measured(self, pitch, now, dt):
        """Learn from the camera pitch robotd measured `dt` seconds after the last sample."""
        if (
            self.anchor is None
            or now - self.anchor[1] < SETTLE
            or (self.gait in WALKING and now - self.changed < GAIT_SETTLE)
            or self._weight(now) < 1
        ):
            return
        error = self.tilt - pitch
        low, f = self._place(self.tilt)
        # Shared between the two bins by their weight in the reading, scaled so the reading
        # moves by the full step wherever the tilt falls between them.
        time = WALK_TRIM_TIME if self.gait in WALKING else TRIM_TIME
        step = error * min(dt, 0.1) / time / ((1 - f) ** 2 + f**2)
        table = self.tables[self.gait]
        for index, weight in ((low, 1 - f), (low + 1, f)):
            if weight > 0:
                value = self._bin(self.gait, index) + weight * step
                table[index] = min(TRIM_LIMIT, max(-TRIM_LIMIT, value))
