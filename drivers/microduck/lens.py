"""The beta's head camera (GC2093, upright) and its lens, which the simulated one shares.

The lens figures are the camera module's datasheet: NOMINAL, not a calibration. 1/2.9" sensor,
2.75 um pixels, 1920x1080 active; field of view 116 degrees across, 60 down, 142.2 corner to
corner (165 the lens's widest). No pinhole does that (116 across would be 84 down): the lens is a
fisheye that squeezes the edges, between equisolid and orthographic. A real beta publishes
geometry only from its own calibration; the sim renders this nominal lens and publishes it as
that calibration would."""

import numpy as np

WIDTH, HEIGHT = 1920, 1080  # the active array; the figures below are at this size
FOV_H_DEG, FOV_V_DEG, FOV_D_DEG, MAX_DFOV_DEG = 116.0, 60.0, 142.2, 165.0
# Kannala-Brandt through those three fields, principal point central: a ray theta radians off
# the axis lands F_PX * (theta + KB[0] theta^3 + KB[1] theta^5) pixels from the centre.
F_PX = 1060.0026
KB = (-0.09718408, -0.00547611)
# The same lens in OpenCV's rational model (k1 k2 p1 p2 k3 k4 k5 k6; fx = fy = F_PX, centre
# 960, 540): what calibrateCamera with CALIB_RATIONAL_MODEL makes of it, within 0.01 px of the
# curve above over the whole frame. OpenCV's plain five terms miss it by 17 px inside 50 degrees.
DISTORTION = (
    0.7803425815871512,
    0.10952142059157981,
    0.0,
    0.0,
    0.0014081666387639196,
    1.2107073041396799,
    0.339970004705214,
    0.016612549934926498,
)
RESOLUTIONS = {360: "360p30", 720: "720p30", 1080: "1080p30"}
DEFAULT_HEIGHT = 720


def width_for(height):
    return height * 16 // 9


def radius_px(theta, width=WIDTH):
    """Pixels from the image centre at which a ray theta radians off the axis lands."""
    t = np.asarray(theta, np.float64)
    return width / WIDTH * F_PX * (t + KB[0] * t**3 + KB[1] * t**5)


def angle(radius, width=WIDTH):
    """Radians off the axis of the ray landing radius pixels from the centre (radius_px's
    inverse; the curve rises monotonically to well past the frame's corners)."""
    r = np.asarray(radius, np.float64) * (WIDTH / width) / F_PX
    t = r.copy()
    for _ in range(8):
        t -= (t + KB[0] * t**3 + KB[1] * t**5 - r) / (1 + 3 * KB[0] * t**2 + 5 * KB[1] * t**4)
    return t


def rays(u, v, width, height):
    """Unit view rays (x right, y down, z forward) of pixels u, v (OpenCV's convention: integer
    coordinates are pixel centres, the axis at width / 2, height / 2)."""
    x, y = np.asarray(u, np.float64) - width / 2, np.asarray(v, np.float64) - height / 2
    r = np.hypot(x, y)
    t = angle(r, width)
    s = np.where(r > 0, np.sin(t) / np.maximum(r, 1e-12), 0.0)
    return np.stack([x * s, y * s, np.cos(t)], -1)
