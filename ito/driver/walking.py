"""Look is startup-relative; robot commands are body-relative, with bounded yaw speed."""

import math
from dataclasses import dataclass

from ito.protocol import PilotState


@dataclass(frozen=True)
class Walking:
    forward: float
    left: float
    turn: float
    pan: float
    tilt: float


def walking(
    state: PilotState,
    body_yaw: float,
    *,
    speed: float,
    lateral_speed: float,
    turn_speed: float,
) -> Walking:
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
    if not state.deadman or math.hypot(forward, right) < 1e-6:
        return Walking(0, 0, 0, pan, pitch)
    c, s = math.cos(pan), math.sin(pan)
    vx, vy = forward * c + right * s, forward * s - right * c
    error = pan
    if lateral_speed == 0:
        # Wheels cannot translate sideways: steer into the requested travel direction,
        # allowing reverse, and suppress translation until that direction is reachable.
        direction = -1 if forward < 0 else 1
        error = math.atan2(direction * vy, direction * vx)
        vx = direction * math.hypot(vx, vy) * max(0, math.cos(error)) ** 4
        vy = 0.0
        scale = speed
    else:
        # Scale both components together so asymmetric gait limits preserve direction.
        scale = min(speed, lateral_speed / max(abs(vy), 1e-9))
    turn = turn_speed * math.tanh(2 * error / turn_speed)
    return Walking(vx * scale, vy * scale, turn, pan, pitch)
