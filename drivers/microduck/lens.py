"""The beta's head camera (GC2093, upright), which the simulated one has to share.

Its field of view has never been measured: HFOV_ESTIMATE_DEG (a 4 mm M12 lens on a 1/2.9"
sensor) is an estimate, to replace once a beta is calibrated. A real beta publishes geometry
only from its own calibration."""

import math

HFOV_ESTIMATE_DEG = 69.7
RESOLUTIONS = {360: "360p30", 720: "720p30", 1080: "1080p30"}
DEFAULT_HEIGHT = 720


def width_for(height):
    return height * 16 // 9


def focal_px(width, hfov_deg=HFOV_ESTIMATE_DEG):
    return width / 2 / math.tan(math.radians(hfov_deg / 2))


def vertical_fov_deg(width, height, hfov_deg=HFOV_ESTIMATE_DEG):
    return math.degrees(2 * math.atan(height / 2 / focal_px(width, hfov_deg)))
