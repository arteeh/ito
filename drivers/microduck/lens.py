"""The real head camera's lens, which the simulated one has to share.

The IMX219 behind the Microduck's M12 lens sees 62 degrees across the frame's long side at every
resolution mediad delivers (mediad's `sensor.rs`). MuJoCo's twin camera takes its vertical field
of view from the MJCF, whose 45 degree default is ~20% wider than the lens, and mediad publishes
that same 45 degrees for it.
"""

import math

HFOV_DEG = 62.0
# Frame heights the simulator can render, with mediad's name for each quality.
RESOLUTIONS = {360: "360p30", 720: "720p30", 1080: "1080p30"}


def width_for(height):
    return height * 16 // 9


def focal_px(width):
    """Focal length in pixels of a delivered frame `width` wide."""
    return width / 2 / math.tan(math.radians(HFOV_DEG / 2))


def vertical_fov_deg(width, height):
    return math.degrees(2 * math.atan(height / 2 / focal_px(width)))
