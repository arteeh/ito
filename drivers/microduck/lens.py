"""Each board's head camera, which the simulated one has to share.

zero3: IMX219, 62 degrees across the long side, mounted a quarter turn off upright (measured).
beta (production): GC2093, upright. Its field of view has never been measured;
BETA_HFOV_ESTIMATE_DEG (69.7, a 4 mm M12 lens on a 1/2.9" sensor) is an estimate, to replace
once a beta is calibrated. A real beta publishes geometry only from its own calibration."""

import math

ZERO3_HFOV_DEG = 62.0
BETA_HFOV_ESTIMATE_DEG = 69.7
# Degrees clockwise each board's camera sits from upright: robotd-params'
# Board::camera_mount_degrees, which mediad announces as media.video's `rotate`.
MOUNT = {"zero3": 90, "beta": 0}
HFOV = {"zero3": ZERO3_HFOV_DEG, "beta": BETA_HFOV_ESTIMATE_DEG}
RESOLUTIONS = {360: "360p30", 720: "720p30", 1080: "1080p30"}
DEFAULT_HEIGHT = {"zero3": 360, "beta": 720}


def width_for(height):
    return height * 16 // 9


def focal_px(width, hfov_deg=ZERO3_HFOV_DEG):
    return width / 2 / math.tan(math.radians(hfov_deg / 2))


def vertical_fov_deg(width, height, hfov_deg=ZERO3_HFOV_DEG):
    return math.degrees(2 * math.atan(height / 2 / focal_px(width, hfov_deg)))
