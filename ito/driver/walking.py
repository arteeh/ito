"""Look is startup-relative; robot commands are body-relative, with bounded yaw speed.

Standing still, the head alone follows the gaze within its pan range. Once the gaze has
settled near or past the edge of that range, the body turns slowly toward it, and only until
the head is back inside the range with room to spare: a glance or a sweep across the edge
moves only the head, and the body never spins round to face the gaze.
"""

import math
from dataclasses import dataclass

from ito.protocol import PilotState

# A standing body follows a gaze that has held still (within STILL for SETTLE seconds) past
# SOFT_LIMIT of the head's pan range, and turns until the pan is back within RETURN of it.
# A head turn moves tens of degrees a second, so a look in progress never counts as settled:
# turning the body during one adds its yaw to the camera's and costs SLAM its matches (#30).
# A gaze kept past the pan limit itself for SETTLE is followed even while it moves: the head
# can no longer show the pilot that view, and a pilot turning round must not be left behind.
SOFT_LIMIT = 0.8
RETURN = 0.7
SETTLE = 0.6
STILL = math.radians(8)
# Following turns at a steady FOLLOW of the robot's turn speed, about half its brisk turn and
# a rate the camera's tracking keeps up with, easing off over the last TAPER and ending HOLD
# short of its target. A long fade would leave the Microduck short: its gait stands still for
# turn rates much under this one (0.44 rad/s there, yaw command about 1).
FOLLOW = 0.55
TAPER = math.radians(6)
HOLD = math.radians(2)
# A turning body reverses only when the other way round is this much shorter, so gaze
# held directly behind the robot cannot make it rock back and forth.
REVERSE = math.radians(20)


@dataclass(frozen=True)
class Walking:
    forward: float
    left: float
    turn: float
    pan: float  # Within the head's pan range.
    tilt: float


class Walker:
    def __init__(
        self,
        pan_limits: tuple[float, float],
        *,
        speed: float,
        lateral_speed: float,
        turn_speed: float,
    ):
        low, high = pan_limits
        if not low <= 0 <= high or high - low >= 2 * math.pi:
            raise ValueError("head pan limits must include zero and span less than a turn")
        self.low, self.high = low, high
        self.speed, self.lateral_speed, self.turn_speed = speed, lateral_speed, turn_speed
        self.following = 0  # +1 left, -1 right, while the body turns toward the gaze.
        # The gaze yaw the pilot holds, and since when; since when the gaze has been out of the
        # head's reach. Pilot capture times.
        self.held = self.held_since = self.out_since = None

    def _beyond(self, pan: float, direction: int, fraction: float) -> float:
        """Gaze beyond `fraction` of the pan limit on one side; negative while inside it.

        Measured up to the other side's limit the long way round, past the back: gaze swept
        behind the robot keeps the body turning the same way.
        """
        bound = fraction * (self.high if direction > 0 else -self.low)
        span = bound + (-self.low if direction > 0 else self.high)
        return (direction * pan - bound + span) % (2 * math.pi) - span

    def _settled(self, yaw: float, pan: float, when: float) -> bool:
        """Whether the gaze has held within STILL of one heading, or out of the head's reach,
        for SETTLE seconds."""
        if self.held is None or abs(math.remainder(yaw - self.held, 2 * math.pi)) > STILL:
            self.held, self.held_since = yaw, when
        if self.low <= pan <= self.high:
            self.out_since = None
        elif self.out_since is None:
            self.out_since = when
        out = self.out_since is not None and when - self.out_since >= SETTLE
        return out or when - self.held_since >= SETTLE

    def __call__(self, state: PilotState, body_yaw: float) -> Walking:
        yaw = pitch = 0.0
        if state.head:
            x, y, z, w = state.head.orientation
            norm = math.sqrt(x * x + y * y + z * z + w * w)
            x, y, z, w = (v / norm for v in (x, y, z, w))
            yaw = math.atan2(2 * (x * z + y * w), 1 - 2 * (x * x + y * y))
            pitch = math.asin(max(-1, min(1, 2 * (x * w - y * z))))
        pan = math.remainder(yaw - body_yaw, 2 * math.pi)
        forward = state.axes.get("move_y", 0.0)
        right = state.axes.get("move_x", 0.0) + state.axes.get("strafe", 0.0)
        norm = max(1, math.hypot(forward, right))
        forward, right = forward / norm, right / norm
        moving = math.hypot(forward, right) >= 1e-6
        if state.deadman and state.head and not moving:
            return self._stand(pan, pitch, self._settled(yaw, pan, state.capture_time))
        self.following = 0
        if not state.deadman or not moving:
            return Walking(0, 0, 0, self._reach(pan), pitch)
        c, s = math.cos(pan), math.sin(pan)
        vx, vy = forward * c + right * s, forward * s - right * c
        error = pan
        if self.lateral_speed == 0:
            # Wheels, or a gait with no usable sideways step: steer into the requested travel
            # direction, allowing reverse, and suppress translation until it is reachable.
            direction = -1 if forward < 0 else 1
            error = math.atan2(direction * vy, direction * vx)
            vx = direction * math.hypot(vx, vy) * max(0, math.cos(error)) ** 4
            vy = 0.0
            scale = self.speed
        else:
            # Scale both components together so asymmetric gait limits preserve direction.
            scale = min(self.speed, self.lateral_speed / max(abs(vy), 1e-9))
        return Walking(vx * scale, vy * scale, self._turn(error), self._reach(pan), pitch)

    def _stand(self, pan: float, pitch: float, settled: bool) -> Walking:
        past = {d: self._beyond(pan, d, SOFT_LIMIT) for d in (1, -1)}
        # Only a gaze past both soft limits, behind the robot, can be shorter the other way.
        f = self.following
        if f and 0 < past[-f] < past[f] - REVERSE:
            self.following = 0
        if not self.following and settled:
            sides = [d for d in (1, -1) if past[d] > 0]
            if sides:
                self.following = min(sides, key=past.get)
        if self.following:
            remaining = self._beyond(pan, self.following, RETURN)
            if remaining > HOLD:
                ease = min(1.0, remaining / TAPER)
                turn = self.following * FOLLOW * self.turn_speed * ease
                return Walking(0, 0, turn, self._reach(pan), pitch)
        # Back inside the range with room to spare, or the gaze came back first: the head
        # takes the rest, and the body stays where it is.
        self.following = 0
        return Walking(0, 0, 0, self._reach(pan), pitch)

    def _reach(self, pan: float) -> float:
        return min(self.high, max(self.low, pan))

    def _turn(self, error: float) -> float:
        return self.turn_speed * math.tanh(2 * error / self.turn_speed)
