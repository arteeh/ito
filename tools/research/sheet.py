"""Fixed-viewpoint comparison sheet: the sim's truth (truth_views.py) beside maps, one row per view
(views.py), all at one pinhole. Maps are npz files with 'records' (Ito splat records, any SH
degree) or an ITO_DUMP_MAP map-*.npz; all are in the recording's Ito world.

    sheet.py REC OUT.png MAP.npz [MAP.npz ...] [--labels ...] [--truth DIR] [--views 0,1,5]
             [--cell 640]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from data import AXES, Recording  # noqa: E402
from views import HEIGHT, VIEWS, WIDTH, focal, mujoco_camera, world_transform  # noqa: E402

TRUTH = Path.home() / "Projects/ito/research/truth-views"


def draw(records, camera, sh=None):
    from gsplat import rasterization

    r = torch.as_tensor(records, device="cuda", dtype=torch.float32)
    coefficients = r.shape[1] - 3
    degree = int(round(np.sqrt(coefficients))) - 1 if sh is None else sh
    f = focal()
    K = torch.tensor([[f, 0, WIDTH / 2], [0, f, HEIGHT / 2], [0, 0, 1]], device="cuda")
    view = torch.as_tensor(np.linalg.inv(camera @ AXES), device="cuda", dtype=torch.float32)
    colours = r[:, 3 : 3 + (degree + 1) ** 2, :3]
    image, alpha, _ = rasterization(
        means=r[:, 0, :3],
        quats=torch.nn.functional.normalize(r[:, 2], dim=-1),
        scales=r[:, 1, :3],
        opacities=r[:, 0, 3].clamp(0, 1),
        colors=colours,
        sh_degree=degree,
        viewmats=view[None],
        Ks=K[None],
        width=WIDTH,
        height=HEIGHT,
        near_plane=0.02,
        far_plane=100,
    )
    background = torch.tensor([0.04, 0.04, 0.06], device="cuda")
    image = image[0] + (1 - alpha[0]) * background
    return (image.clamp(0, 1) * 255).byte().cpu().numpy()


def label(img, text):
    cv2.rectangle(img, (0, img.shape[0] - 24), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    cv2.putText(
        img, text, (6, img.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 230, 255), 1,
        cv2.LINE_AA,
    )
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("rec")
    p.add_argument("out")
    p.add_argument("maps", nargs="+")
    p.add_argument("--labels", nargs="+")
    p.add_argument("--truth", default=str(TRUTH))
    p.add_argument("--views")
    p.add_argument("--cell", type=int, default=640)
    a = p.parse_args()
    rec = Recording(a.rec)
    transform = world_transform(rec.truth_frames())
    labels = a.labels or [Path(m).stem for m in a.maps]
    maps = []
    for m in a.maps:
        data = np.load(m)
        records = data["records"]
        if "keys" in data and records.shape[1] == 4:
            pass
        maps.append(records)
    chosen = range(len(VIEWS)) if not a.views else [int(v) for v in a.views.split(",")]
    cell = (a.cell, round(a.cell * HEIGHT / WIDTH))
    rows = []
    for i in chosen:
        name, position, target = VIEWS[i]
        camera = transform @ mujoco_camera(position, target)
        cells = []
        truth = cv2.imread(str(Path(a.truth) / f"{i}.png"))
        if truth is not None:
            cells.append(label(cv2.resize(truth, cell, interpolation=cv2.INTER_AREA), f"truth | {name}"))
        for records, text in zip(maps, labels, strict=True):
            img = draw(records, camera)[..., ::-1].copy()
            img = cv2.resize(img, cell, interpolation=cv2.INTER_AREA)
            cells.append(label(img, f"{text} | {len(records):,}"))
        rows.append(np.concatenate(cells, 1))
    cv2.imwrite(a.out, np.concatenate(rows, 0))
    print(a.out)


if __name__ == "__main__":
    main()
