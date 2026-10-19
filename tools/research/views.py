"""Fixed viewpoints in the virtual Microduck's room, shared by truth renders (truth_views.py, the sim's
MuJoCo) and map renders (sheet.py): each a camera position and a looked-at point in MuJoCo's world
(robotd's: x forward from the spawn, y left, z up), drawn as a WIDTH x HEIGHT pinhole of HFOV."""

import math

import numpy as np

WIDTH, HEIGHT, HFOV = 960, 540, 90.0
VIEWS = [
    ("bookcase close", (1.95, 0.45, 0.22), (2.5, 0.6, 0.30)),
    ("bookcase, books", (2.05, 0.75, 0.30), (2.5, 0.55, 0.40)),
    ("table and bookcase", (1.0, 0.4, 0.25), (2.3, 0.9, 0.25)),
    ("sofa", (1.1, 0.2, 0.25), (0.2, 1.4, 0.25)),
    ("plant and art", (1.0, -0.3, 0.30), (2.4, -1.0, 0.6)),
    ("pot close", (1.45, -0.75, 0.25), (1.95, -1.18, 0.15)),
    ("room overview", (-0.3, -0.3, 0.45), (2.0, 0.6, 0.25)),
]


def focal():
    return WIDTH / 2 / math.tan(math.radians(HFOV / 2))


def mujoco_camera(position, target):
    """Camera-to-world in MuJoCo's world, camera axes x right, y up, looking down -z (Ito's)."""
    p, t = np.asarray(position, float), np.asarray(target, float)
    z = p - t
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, :3] = np.stack((x, y, z), 1)
    m[:3, 3] = p
    return m


def world_transform(frames):
    """Ito world from MuJoCo world, from a recording's truth rows (both poses of one frame)."""
    from data import pose_matrix

    estimates = []
    for f in frames[:: max(1, len(frames) // 20)]:
        if "sim" not in f:
            continue
        mj = pose_matrix(f["sim"]["position"], f["sim"]["orientation"])
        estimates.append(f["pose"] @ np.linalg.inv(mj))
    if not estimates:
        raise ValueError("recording has no sim truth")
    return estimates[len(estimates) // 2]
