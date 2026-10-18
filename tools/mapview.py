"""Draw maps a session dumped (ITO_DUMP_MAP, session.py --env) from fixed viewpoints in the
virtual Microduck's room, so maps from different runs are seen from exactly the same places.

Needs a build's Python (CUDA, gsplat). From WSL:

    cd /mnt/c/Users/me/Projects/ito/builds/BUILD/ito && cmd.exe /c "runtime\\python.exe -I
    C:\\path\\to\\tools\\mapview.py OUT.png MAP.npz [MAP.npz ...] [--labels A B ...]"

Each MAP is a map-*.npz from a session's dump folder (the last one is the map at its end).
Rows are viewpoints, columns maps. Viewpoints are in the sim room's frame (robotd's world:
x forward from where the robot starts, y left, z up), which is Ito's world for a run whose
robot started at the sim's spawn pose, as every scripted session's does.
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from gsplat import rasterization

AXES = np.diag([1.0, -1.0, -1.0, 1.0])
# robotd's world (x forward, y left, z up) to Ito's (x right, y up, z back).
TO_ITO = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
# (name, camera position, looked-at point), metres in the room's frame.
VIEWS = [
    ("bookcase close", (1.95, 0.45, 0.22), (2.5, 0.6, 0.30)),
    ("bookcase, books", (2.05, 0.75, 0.30), (2.5, 0.55, 0.40)),
    ("table and bookcase", (1.0, 0.4, 0.25), (2.3, 0.9, 0.25)),
    ("sofa", (1.1, 0.2, 0.25), (0.2, 1.4, 0.25)),
    ("plant and art", (1.0, -0.3, 0.30), (2.4, -1.0, 0.6)),
    ("room overview", (-0.3, -0.3, 0.45), (2.0, 0.6, 0.25)),
]
WIDTH, HEIGHT, HFOV = 960, 540, 90.0


def pose(position, target):
    c = TO_ITO @ np.asarray(position, float)
    t = TO_ITO @ np.asarray(target, float)
    z = c - t
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, :3] = np.stack((x, y, z), 1)
    m[:3, 3] = c
    return m


def aligned(dump):
    """A dump's splats in the robot's starting frame: an ITO_SLAM=full map holds its world
    where the pilot's keyframe is (dump "world" takes it back)."""
    records = dump["records"].copy()
    if "world" not in dump:
        return records
    from scipy.spatial.transform import Rotation

    world = dump["world"]
    records[:, 0, :3] = records[:, 0, :3] @ world[:3, :3].T + world[:3, 3]
    turn = Rotation.from_matrix(world[:3, :3])
    quats = Rotation.from_quat(records[:, 2][:, [1, 2, 3, 0]])
    records[:, 2] = (turn * quats).as_quat()[:, [3, 0, 1, 2]]
    return records


def draw(records, camera):
    r = torch.as_tensor(records, device="cuda", dtype=torch.float32)
    focal = WIDTH / 2 / math.tan(math.radians(HFOV / 2))
    K = torch.tensor([[focal, 0, WIDTH / 2], [0, focal, HEIGHT / 2], [0, 0, 1]], device="cuda")
    view = torch.as_tensor(np.linalg.inv(camera @ AXES), device="cuda", dtype=torch.float32)
    image, alpha, _ = rasterization(
        means=r[:, 0, :3],
        quats=r[:, 2],
        scales=r[:, 1, :3],
        opacities=r[:, 0, 3],
        colors=r[:, 3, :3][:, None],
        sh_degree=0,
        viewmats=view[None],
        Ks=K[None],
        width=WIDTH,
        height=HEIGHT,
        near_plane=0.02,
    )
    background = torch.tensor([0.04, 0.04, 0.06], device="cuda")
    image = image[0] + (1 - alpha[0]) * background
    rgb = (image.clamp(0, 1) * 255).byte().cpu().numpy()
    return np.ascontiguousarray(rgb[..., ::-1])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("out", type=Path)
    p.add_argument("maps", nargs="+", type=Path)
    p.add_argument("--labels", nargs="+")
    a = p.parse_args()
    labels = a.labels or [m.parent.name for m in a.maps]
    maps = [aligned(np.load(m)) for m in a.maps]
    rows = []
    for name, position, target in VIEWS:
        camera = pose(position, target)
        cells = []
        for records, label in zip(maps, labels, strict=True):
            img = draw(records, camera)
            cv2.rectangle(img, (0, HEIGHT - 30), (WIDTH, HEIGHT), (0, 0, 0), -1)
            cv2.putText(
                img,
                f"{label} | {name} | {len(records):,} splats",
                (8, HEIGHT - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (80, 230, 255),
                1,
                cv2.LINE_AA,
            )
            cells.append(img)
        rows.append(np.concatenate(cells, 1))
    cv2.imwrite(str(a.out), np.concatenate(rows, 0))
    print(a.out)


if __name__ == "__main__":
    main()
