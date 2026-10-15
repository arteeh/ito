"""Each board's head camera, which the simulated one has to share.

zero3 (the Radxa Zero 3W prototype): the IMX219 behind the M12 lens sees 62 degrees across the
frame's long side at every resolution mediad delivers (mediad's `sensor.rs`, solved on hardware),
mounted a quarter turn off upright (mediad announces rotate 90).

beta (Seeed's RK3566 board, the production design): the GC2093, upright (rotate 0), its only mode
1920x1080 at 30 fps that the ISP scales to the [media] quality rung, 720p30 by default. Nobody has
measured its lens: microduck 0.17.4 ships no field of view and no family solve for it, so a real
beta publishes geometry only from its own [media.intrinsics] calibration. The simulated beta stands
in for such a calibrated robot, at BETA_HFOV_ESTIMATE_DEG. That number is an estimate, not a
measurement: a 1/2.9" GC2093 (1920 x 2.9 um = 5.57 mm wide) behind a 4 mm M12 lens, the stock
lens of GC2093 board modules, sees 2 atan(5.57 / 2 / 4) = 69.7 degrees across. Vendor modules
range from 67 (4.35 mm) to 144 degrees (2.28 mm), so replace it once a beta is calibrated
(`duckslam calib intrinsics`); `microduck-sim --beta-hfov` tries another.
"""

import math

ZERO3_HFOV_DEG = 62.0
BETA_HFOV_ESTIMATE_DEG = 69.7
# Degrees clockwise each board's camera sits from upright: robotd-params'
# Board::camera_mount_degrees, which mediad announces as media.video's `rotate`.
MOUNT = {"zero3": 90, "beta": 0}
HFOV = {"zero3": ZERO3_HFOV_DEG, "beta": BETA_HFOV_ESTIMATE_DEG}
# Frame heights the simulator can render, with mediad's name for each quality.
RESOLUTIONS = {360: "360p30", 720: "720p30", 1080: "1080p30"}
# mediad's default rung is 720p30 on a real duck. The zero3 twin keeps 360p, what every
# zero3 run was recorded at.
DEFAULT_HEIGHT = {"zero3": 360, "beta": 720}


def width_for(height):
    return height * 16 // 9


def focal_px(width, hfov_deg=ZERO3_HFOV_DEG):
    """Focal length in pixels of a delivered frame `width` wide."""
    return width / 2 / math.tan(math.radians(hfov_deg / 2))


def vertical_fov_deg(width, height, hfov_deg=ZERO3_HFOV_DEG):
    return math.degrees(2 * math.atan(height / 2 / focal_px(width, hfov_deg)))
