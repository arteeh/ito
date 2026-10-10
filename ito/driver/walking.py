"""Look is startup-relative; robot commands are body-relative, with bounded yaw speed.

Standing still, the body starts turning when the pilot looks near the edge of the head's pan
range, and keeps turning until the head is back near the middle, so the next look the same way
has room before the body has to move again.
"""

import math
from dataclasses import dataclass

from ito.protocol import PilotState

# A standing body starts following a gaze this far out toward a pan limit, as a fraction of
# it, and turns until the head's pan is back within HEADROOM of the limit's way. The band
# between the two is the hysteresis: a gaze held at the edge turns the body once, not in jerks.
SOFT_LIMIT = 0.8
HEADROOM = 0.3
# Following is gentlest for a gaze just past the soft limit and fastest for one at or past
# the hard limit: its top speed is this fraction of the turn speed, scaled up to the full
# speed between them. Close to the headroom every follow slows down the same way.
GENTLE = 0.5
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

    def _beyond(self, pan: float, direction: int, fraction: float) -> float:
        """Gaze beyond `fraction` of the pan limit on one side; negative while inside it.

        Measured up to the other side's limit the long way round, past the back: gaze swept
        behind the robot keeps the body turning the same way.
        """
        bound = fraction * (self.high if direction > 0 else -self.low)
        span = bound + (-self.low if direction > 0 else self.high)
        return (direction * pan - bound + span) % (2 * math.pi) - span

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
            return self._stand(pan, pitch)
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

    def _stand(self, pan: float, pitch: float) -> Walking:
        excess = {d: self._beyond(pan, d, SOFT_LIMIT) for d in (1, -1)}
        # Only a gaze past both soft limits, behind the robot, can be shorter the other way.
        f = self.following
        if f and 0 < excess[-f] < excess[f] - REVERSE:
            self.following = 0
        past = [d for d in (1, -1) if excess[d] > 0]
        if not self.following and past:
            # Behind the robot the gaze is past both; turn the shorter way.
            self.following = min(past, key=excess.get)
        if self.following:
            remaining = self._beyond(pan, self.following, HEADROOM)
            if remaining > 0:
                bound = self.high if self.following > 0 else -self.low
                urgency = excess[self.following] / ((1 - SOFT_LIMIT) * bound)
                scale = GENTLE + (1 - GENTLE) * min(1.0, max(0.0, urgency))
                turn = self._turn(self.following * remaining, scale * self.turn_speed)
                return Walking(0, 0, turn, self._reach(pan), pitch)
        # Back within the headroom, or overshot by body inertia: never turn back toward it.
        self.following = 0
        return Walking(0, 0, 0, self._reach(pan), pitch)

    def _reach(self, pan: float) -> float:
        return min(self.high, max(self.low, pan))

    def _turn(self, error: float, speed: float | None = None) -> float:
        speed = speed or self.turn_speed
        return speed * math.tanh(2 * error / speed)
