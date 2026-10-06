"""Column-vector rigid transforms; matrices are passed to GL in column-major order."""

import math

import numpy as np

from .scene import FloatArray


def pose(position=(0.0, 0.0, 0.0), yaw: float = 0, pitch: float = 0) -> FloatArray:
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    result = np.eye(4, dtype=np.float32)
    result[:3, :3] = ((cy, sy * sp, sy * cp), (0, cp, -sp), (-sy, cy * sp, cy * cp))
    result[:3, 3] = position
    return result


def validate_pose(value: FloatArray) -> FloatArray:
    matrix = np.asarray(value, dtype=np.float32)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-5)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-4)):
        raise ValueError("Pose must be a finite, right-handed rigid 4x4 transform")
    return matrix


def perspective(fov_y: float, aspect: float, near: float = 0.02, far: float = 1000) -> FloatArray:
    if not (0 < fov_y < math.pi and aspect > 0 and 0 < near < far
            and all(map(math.isfinite, (fov_y, aspect, near, far)))):
        raise ValueError("Invalid perspective frustum")
    f = 1 / math.tan(fov_y / 2)
    return np.array(((f / aspect, 0, 0, 0), (0, f, 0, 0),
                     (0, 0, (far + near) / (near - far), 2 * far * near / (near - far)),
                     (0, 0, -1, 0)), dtype=np.float32)
