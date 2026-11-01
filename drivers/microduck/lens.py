"""The beta's head camera (GC2093, upright) and its lens, which the simulated one renders.

The lens is upstream's family calibration of the beta camera (microduck fb8875e,
robotd-params' CameraIntrinsics::beta(), solved on unit sky-gtc on the 1280x720 frame), which
mediad publishes for every real beta. The sim renders it and publishes the same figures.

Its five-term radial polynomial folds back 57.9 degrees off the axis: past that a wider ray lands
nearer the centre. That is short of the frame's right edge and corners, which the calibration's
views did not reach, so it cannot describe them. Past KNEE_DEG the sim continues the lens at its
slope there (angle linear in image radius), which puts the frame at about 114 by 60 degrees, the
module's datasheet 116 by 60. Ito's rectified view stays inside KNEE_DEG."""

import math

import numpy as np

WIDTH, HEIGHT = 1280, 720  # the calibration's frame; other sizes scale it
FX, FY = 695.6216231288809, 696.0209167299262
CX, CY = 627.4740538073111, 363.60566030048255
DISTORTION = (  # OpenCV's k1 k2 p1 p2 k3
    -0.368607360841971,
    0.13550162561831905,
    0.00012682873881955374,
    0.0008719998338708391,
    -0.022368269599378002,
)
KNEE_DEG = 54.0

K1, K2, P1, P2, K3 = DISTORTION
_KNEE_U = math.tan(math.radians(KNEE_DEG))  # pinhole radius of the knee


def _radial(r):
    return r * (1 + K1 * r**2 + K2 * r**4 + K3 * r**6)


def _radial_slope(r):
    return 1 + 3 * K1 * r**2 + 5 * K2 * r**4 + 7 * K3 * r**6


_KNEE_D = _radial(_KNEE_U)
# d(angle) / d(distorted radius) at the knee, which the lens keeps beyond it.
_KNEE_SLOPE = 1 / (1 + _KNEE_U**2) / _radial_slope(_KNEE_U)


def undistorted_radius(rd):
    """Pinhole radius (tan of the angle off the axis) of distorted normalised radius rd; inf
    for rays at or past 90 degrees."""
    rd = np.asarray(rd, np.float64)
    inner = rd <= _KNEE_D
    r = np.where(inner, rd, _KNEE_U)
    for _ in range(30):
        r = r - np.where(inner, (_radial(r) - rd) / _radial_slope(r), 0.0)
        r = np.clip(r, 0.0, _KNEE_U)
    angle = math.radians(KNEE_DEG) + (rd - _KNEE_D) * _KNEE_SLOPE
    outer = np.where(angle < math.pi / 2, np.tan(np.minimum(angle, math.pi / 2 - 1e-9)), np.inf)
    return np.where(inner, r, outer)


def rays(u, v, width, height):
    """Unit view rays (x right, y down, z forward) of pixels u, v (OpenCV's convention: integer
    coordinates are pixel centres) on a width x height frame."""
    s = width / WIDTH
    xd = (np.asarray(u, np.float64) - CX * s) / (FX * s)
    yd = (np.asarray(v, np.float64) - CY * height / HEIGHT) / (FY * height / HEIGHT)
    # The tangential terms are a pixel or two: remove them at the current estimate, then solve
    # the radial part exactly. Past the knee they stay as they are at it.
    x, y = xd, yd
    for _ in range(5):
        shrink = np.minimum(1.0, _KNEE_U / np.maximum(np.hypot(x, y), 1e-12))
        tx_, ty_ = x * shrink, y * shrink
        r2 = tx_ * tx_ + ty_ * ty_
        tx = 2 * P1 * tx_ * ty_ + P2 * (r2 + 2 * tx_ * tx_)
        ty = P1 * (r2 + 2 * ty_ * ty_) + 2 * P2 * tx_ * ty_
        ex, ey = xd - tx, yd - ty
        rd = np.hypot(ex, ey)
        ru = undistorted_radius(rd)
        scale = np.where(rd > 0, ru / np.maximum(rd, 1e-12), 1.0)
        x, y = ex * scale, ey * scale
    ray = np.stack([x, y, np.ones_like(x)], -1)
    return ray / np.linalg.norm(ray, axis=-1, keepdims=True)
