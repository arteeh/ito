"""CPU pose policy: gravity from the robot, translation and slow yaw correction from vision."""

import math

import numpy as np

AXES = np.array([1.0, -1.0, -1.0])
# 800 paired Microduck sim poses: p95 1.32 deg, max 2.10 deg over 2.07 m.
# Six degrees leaves >2x the observed maximum; hardware/CUDA still need validation.
ROTATION_GATE = math.radians(6)
YAW_TAU = 20.0
YAW_RATE = math.radians(0.5)
YAW_LIMIT = math.radians(10)


def turn(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)))


def rotation_error(a, b):
    return math.acos(float(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1)))


def split_sim3(matrix):
    """Separate positive scale without scaling translation; reject reflections and collapse."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return None, math.nan
    determinant = np.linalg.det(matrix[:3, :3])
    scale = np.cbrt(determinant)
    if not 0.01 < scale < 100 or not np.allclose(matrix[3], (0, 0, 0, 1)):
        return None, float(scale)
    u, singular, vh = np.linalg.svd(matrix[:3, :3])
    if max(abs(singular / scale - 1)) > 0.01:
        return None, float(scale)
    pose = np.eye(4)
    pose[:3, :3] = (u @ vh) * AXES[:, None] * AXES[None, :]
    pose[:3, 3] = matrix[:3, 3] * AXES
    return pose, float(scale)


def predict(prior, previous, yaw):
    """Metric odometry is not added to an arbitrary-scale SLAM translation."""
    result = previous.copy()
    result[:3, :3] = turn(yaw) @ prior[:3, :3]
    return result


def blend(prior, visual, yaw, dt, bound=ROTATION_GATE):
    """Return (pose or None, yaw bias). Rejected observations cannot change the bias.

    SO(3) geodesic gating includes pitch and roll. Only the world-Y twist of the
    accepted visual residual corrects yaw; gravity remains entirely the IMU's.
    The complementary filter is time based, capped at 0.5 deg/s and +/-10 degrees.
    """
    expected = predict(prior, visual, yaw)
    if not np.isfinite(visual).all() or rotation_error(expected[:3, :3], visual[:3, :3]) > bound:
        return None, yaw
    residual = visual[:3, :3] @ prior[:3, :3].T
    target = math.atan2(residual[0, 2] - residual[2, 0], residual[0, 0] + residual[2, 2])
    dt = float(np.clip(dt, 0, 0.25))  # no large correction after a tracking gap
    error = math.remainder(target - yaw, math.tau)
    step = np.clip(error * -math.expm1(-dt / YAW_TAU), -YAW_RATE * dt, YAW_RATE * dt)
    yaw = float(np.clip(yaw + step, -YAW_LIMIT, YAW_LIMIT))
    return predict(prior, visual, yaw), yaw
