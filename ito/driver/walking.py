import math
from dataclasses import dataclass, replace

from ito.protocol import PilotState

ENGAGE = math.radians(2)
REVERSE = math.radians(20)


@dataclass(frozen=True)
class Walking:
    forward: float
    left: float
    turn: float
    pan: float
    tilt: float
    # The pilot's head roll, left ear down positive; a head without a roll joint ignores it.
    roll: float = 0.0


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

    def _excess(self, pan: float, direction: int) -> float:
        limit = self.high if direction > 0 else self.low
        span = self.high - self.low
        return (direction * (pan - limit) + span) % (2 * math.pi) - span

    def __call__(self, state: PilotState, body_yaw: float) -> Walking:
        yaw = pitch = roll = 0.0
        if state.head:
            x, y, z, w = state.head.orientation
            norm = math.sqrt(x * x + y * y + z * z + w * w)
            x, y, z, w = (v / norm for v in (x, y, z, w))
            yaw = math.atan2(2 * (x * z + y * w), 1 - 2 * (x * x + y * y))
            pitch = math.asin(max(-1, min(1, 2 * (x * w - y * z))))
            roll = math.atan2(2 * (x * y + z * w), 1 - 2 * (x * x + z * z))
        pan = math.remainder(yaw - body_yaw, 2 * math.pi)
        forward = state.axes.get("move_y", 0.0)
        right = state.axes.get("move_x", 0.0) + state.axes.get("strafe", 0.0)
        norm = max(1, math.hypot(forward, right))
        forward, right = forward / norm, right / norm
        moving = math.hypot(forward, right) >= 1e-6
        if state.deadman and state.head and not moving:
            return replace(self._stand(pan, pitch), roll=roll)
        self.following = 0
        if not state.deadman or not moving:
            return Walking(0, 0, 0, self._reach(pan), pitch, roll)
        c, s = math.cos(pan), math.sin(pan)
        vx, vy = forward * c + right * s, forward * s - right * c
        error = pan
        if self.lateral_speed == 0:
            direction = -1 if forward < 0 else 1
            error = math.atan2(direction * vy, direction * vx)
            vx = direction * math.hypot(vx, vy) * max(0, math.cos(error)) ** 4
            vy = 0.0
            scale = self.speed
        else:
            scale = min(self.speed, self.lateral_speed / max(abs(vy), 1e-9))
        return Walking(vx * scale, vy * scale, self._turn(error), self._reach(pan), pitch, roll)

    def _stand(self, pan: float, pitch: float) -> Walking:
        excess = {d: self._excess(pan, d) for d in (1, -1)}
        if self.following and excess[-self.following] < excess[self.following] - REVERSE:
            self.following = 0
        if not self.following:
            shorter = min(excess, key=excess.get)
            if excess[shorter] > ENGAGE:
                self.following = shorter
        if self.following and excess[self.following] > 0:
            limit = self.high if self.following > 0 else self.low
            turn = self._turn(self.following * excess[self.following])
            return Walking(0, 0, turn, limit, pitch)
        self.following = 0
        return Walking(0, 0, 0, self._reach(pan), pitch)

    def _reach(self, pan: float) -> float:
        return min(self.high, max(self.low, pan))

    def _turn(self, error: float) -> float:
        return self.turn_speed * math.tanh(2 * error / self.turn_speed)
