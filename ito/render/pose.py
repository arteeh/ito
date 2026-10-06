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
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-5)
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-4)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-4)
    ):
        raise ValueError("Pose must be a finite, right-handed rigid 4x4 transform")
    return matrix


def perspective(fov_y: float, aspect: float, near: float = 0.02, far: float = 1000) -> FloatArray:
    if not (
        0 < fov_y < math.pi
        and aspect > 0
        and 0 < near < far
        and all(map(math.isfinite, (fov_y, aspect, near, far)))
    ):
        raise ValueError("Invalid perspective frustum")
    f = 1 / math.tan(fov_y / 2)
    return np.array(
        (
            (f / aspect, 0, 0, 0),
            (0, f, 0, 0),
            (0, 0, (far + near) / (near - far), 2 * far * near / (near - far)),
            (0, 0, -1, 0),
        ),
        dtype=np.float32,
    )


def quaternion(matrix: FloatArray) -> tuple[float, float, float, float]:
    """Extract xyzw without losing roll or precision near a half turn."""
    r = matrix[:3, :3]
    trace = float(np.trace(r))
    if trace > 0:
        scale = math.sqrt(trace + 1) * 2
        values = (
            (r[2, 1] - r[1, 2]) / scale,
            (r[0, 2] - r[2, 0]) / scale,
            (r[1, 0] - r[0, 1]) / scale,
            scale / 4,
        )
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        scale = math.sqrt(1 + float(r[i, i] - r[j, j] - r[k, k])) * 2
        values = [0.0] * 4
        values[i] = scale / 4
        values[j] = (r[j, i] + r[i, j]) / scale
        values[k] = (r[k, i] + r[i, k]) / scale
        values[3] = (r[k, j] - r[j, k]) / scale
    return tuple(map(float, np.asarray(values) / np.linalg.norm(values)))
