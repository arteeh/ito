"""The head camera's measured pose, from robotd's own forward kinematics, in Ito's world.

robotd publishes, every tick, the camera pose in the trunk frame at the measured head joints
(`robot.state` frames.camera, OpenCV axes of the upright image Ito receives) and the trunk's
IMU orientation in its gravity-aligned world. Composing the two gives the camera's real yaw,
pitch and roll: the head pans about an axis the pitch joints tilt, so pan alone is off by
degrees at wide looks and the camera rolls by up to ~30 degrees near its pan limit.

Quaternions here are scalar-first (w, x, y, z), as robotd sends them.
"""

import math

# robotd's world and trunk: x forward, y left, z up. Ito's world: x right, y up, z back.
TO_ITO = (0.5, -0.5, 0.5, 0.5)
# OpenCV camera axes (x right, y down, z ahead) to Ito's (x right, y up, looking down -z).
CAMERA_TO_ITO = (0.0, 1.0, 0.0, 0.0)


def multiply(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def rotate(q, v):
    w, x, y, z = multiply(multiply(q, (0.0, *v)), (q[0], -q[1], -q[2], -q[3]))
    return (x, y, z)


def normalized(q):
    norm = math.sqrt(sum(v * v for v in q))
    if not math.isfinite(norm) or norm < 1e-6:
        raise ValueError("Microduck sent a degenerate orientation")
    return tuple(v / norm for v in q)


def heading(yaw):
    """Rotation about robotd's vertical axis."""
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def camera_in_world(trunk, position, camera, origin):
    """Ito world-from-camera (position, xyzw quaternion), world anchored at robot startup.

    trunk: IMU trunk-to-world quaternion; position: odometry trunk position (z above the
    floor); camera: robotd's (position, quaternion) of the camera in the trunk;
    origin: (x, y, yaw) of the trunk at startup in robotd's world.
    """
    x0, y0, yaw0 = origin
    start = heading(-yaw0)
    trunk = normalized(trunk)
    offset = rotate(trunk, camera[0])
    world = rotate(
        start, (position[0] + offset[0] - x0, position[1] + offset[1] - y0, position[2] + offset[2])
    )
    w, x, y, z = normalized(
        multiply(
            TO_ITO, multiply(start, multiply(trunk, multiply(normalized(camera[1]), CAMERA_TO_ITO)))
        )
    )
    return rotate(TO_ITO, world), (x, y, z, w)


def angles(orientation):
    """Yaw (left positive), pitch (up positive) and roll (image right side up positive) of
    an Ito xyzw camera orientation, radians."""
    x, y, z, w = orientation
    q = (w, x, y, z)
    right, up, back = (rotate(q, axis) for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    yaw = math.atan2(back[0], back[2])
    pitch = math.asin(max(-1.0, min(1.0, -back[1])))
    roll = math.atan2(right[1], up[1])
    return yaw, pitch, roll
