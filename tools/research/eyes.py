"""Eye tests: what the pilot's eyes would see, near the robot's camera, against the sim's truth.

eyes.py poses REC OUT_DIR [--count 8] [--start S] [--end E]
    Picks COUNT moments evenly over the route (frames with truth); at each, five eyes around
    the exact camera: centre, left/right eye (-/+3.2 cm), head 8 cm forward, head turned 25 deg
    left. Writes OUT_DIR/eyes.json (Ito-world poses + labels) and OUT_DIR/truth.json (MuJoCo-world
    poses for truth_poses.py), then: MUJOCO_GL=osmesa SIM_PY truth_poses.py RL truth.json OUT_DIR/truth
eyes.py sheet OUT_DIR OUT.png MAP.npz[@POSES.npz] [...] [--labels ...] [--moments 0,3,5]
    [--eyes centre,left,right,forward,turn] [--cell 480]
    Rows: moment x eye; columns: truth, then each map rendered from that eye (gsplat, the
    rectified camera's pinhole). A map made with estimated poses is drawn from ITS estimate of
    that moment's camera (nearest seq in POSES.npz) plus the same eye offset: what the pilot,
    whose eye rides the same estimate, would see. Without @POSES the exact pose is used."""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data import AXES, Recording  # noqa: E402
from views import world_transform  # noqa: E402

SIZE = (640, 360)
EYES = {
    "centre": (np.zeros(3), 0.0),
    "left": (np.array([-0.032, 0, 0]), 0.0),
    "right": (np.array([0.032, 0, 0]), 0.0),
    "forward": (np.array([0, 0, -0.08]), 0.0),
    "turn": (np.zeros(3), 25.0),
}


def eye_pose(camera, offset, turn_deg):
    """camera: world-from-camera (Ito axes); offset in the camera frame; turn about world up."""
    out = camera.copy()
    out[:3, 3] = camera[:3, :3] @ offset + camera[:3, 3]
    if turn_deg:
        a = math.radians(turn_deg)
        up = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        out[:3, :3] = up @ camera[:3, :3]
    return out


def poses(a):
    rec = Recording(a.rec)
    frames = rec.between(a.start, a.end)
    idx = np.linspace(0, len(frames) - 1, a.count + 2)[1:-1].astype(int)
    transform = world_transform(rec.truth_frames())
    K = rec.K_at(SIZE)
    fovy = math.degrees(2 * math.atan(SIZE[1] / 2 / K[1, 1]))
    items, mj = [], []
    t0 = rec.frames[0]["received"]
    for m, i in enumerate(idx):
        f = frames[i]
        for name, (offset, turn) in EYES.items():
            pose = eye_pose(f["pose"], offset, turn)
            items.append(
                dict(moment=m, eye=name, seq=f["seq"], t=f["received"] - t0, pose=pose.tolist())
            )
            mj.append((np.linalg.inv(transform) @ pose).ravel().tolist())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eyes.json").write_text(json.dumps(dict(K=K.tolist(), size=SIZE, items=items)))
    (out / "truth.json").write_text(
        json.dumps(dict(width=SIZE[0], height=SIZE[1], fovy=fovy, poses=mj))
    )
    print(len(items), "eyes;", "fovy", round(fovy, 2))


def sheet(a):
    import torch
    from gsplat import rasterization

    spec = json.loads((Path(a.dir) / "eyes.json").read_text())
    K = torch.tensor(spec["K"], dtype=torch.float32, device="cuda")
    w, h = spec["size"]
    # --scale S renders the maps at S times the eye grid (truth is upscaled): detail finer than the
    # 640-px grid shows (the headset resolves ~1 mrad, the grid ~2.5).
    Kr = K.clone()
    Kr[:2, :2] *= a.scale
    Kr[:2, 2] = (K[:2, 2] + 0.5) * a.scale - 0.5
    wr, hr = round(w * a.scale), round(h * a.scale)
    maps, estimates = [], []
    for m in a.maps:
        if m.startswith("dir:"):
            # An outside system's own renders of these eyes: DIR/{item}.png (its renderer).
            maps.append(Path(m[4:]))
            estimates.append(None)
            continue
        path, _, given = m.partition("@")
        maps.append(np.load(path)["records"])
        if given:
            g = np.load(given)
            estimates.append((np.asarray(g["seq"]), np.asarray(g["poses"])))
        else:
            estimates.append(None)
    labels = a.labels or [Path(m.partition("@")[0].removeprefix("dir:")).stem for m in a.maps]
    moments = None if not a.moments else {int(v) for v in a.moments.split(",")}
    eyes = a.eyes.split(",")
    crop = [float(v) for v in a.crop.split(",")] if a.crop else [0, 0, 1, 1]
    cell = (a.cell, round(a.cell * h * (crop[3] - crop[1]) / (w * (crop[2] - crop[0]))))

    def fit(img):
        # --crop x0,y0,x1,y1 (fractions of the frame): a close-up of every cell, same region.
        ih, iw = img.shape[:2]
        x0, y0, x1, y1 = (
            round(crop[0] * iw),
            round(crop[1] * ih),
            round(crop[2] * iw),
            round(crop[3] * ih),
        )
        return cv2.resize(img[y0:y1, x0:x1], cell, interpolation=cv2.INTER_AREA)

    rows = []
    for i, item in enumerate(spec["items"]):
        if (moments is not None and item["moment"] not in moments) or item["eye"] not in eyes:
            continue
        truth = cv2.imread(str(Path(a.dir) / "truth" / f"{i}.png"))
        cells = [label(fit(truth), f"truth t={item['t']:.0f}s {item['eye']}")]
        offset, turn = EYES[item["eye"]]
        for records, text, estimate in zip(maps, labels, estimates, strict=True):
            if isinstance(records, Path):
                img = cv2.imread(str(records / f"{i}.png"))
                if img is None:
                    img = np.zeros((h, w, 3), np.uint8)
                cells.append(label(fit(img), text))
                continue
            pose = np.array(item["pose"])
            if estimate is not None:
                seqs, poses_ = estimate
                j = int(np.argmin(np.abs(seqs - item["seq"])))
                pose = eye_pose(poses_[j], offset, turn)
            view = torch.as_tensor(np.linalg.inv(pose @ AXES), dtype=torch.float32, device="cuda")
            r = torch.as_tensor(records, device="cuda", dtype=torch.float32)
            degree = int(round(math.sqrt(r.shape[1] - 3))) - 1
            with torch.no_grad():
                image, alpha, _ = rasterization(
                    r[:, 0, :3],
                    torch.nn.functional.normalize(r[:, 2], dim=-1),
                    r[:, 1, :3],
                    r[:, 0, 3].clamp(0, 1),
                    r[:, 3 : 3 + (degree + 1) ** 2, :3],
                    view[None],
                    Kr[None],
                    wr,
                    hr,
                    sh_degree=degree,
                    near_plane=0.02,
                    far_plane=100,
                )
            background = torch.tensor([0.04, 0.04, 0.06], device="cuda")
            image = image[0] + (1 - alpha[0]) * background
            img = (image.clamp(0, 1) * 255).byte().cpu().numpy()[..., ::-1].copy()
            cells.append(label(fit(img), text))
        rows.append(np.concatenate(cells, 1))
    cv2.imwrite(a.out, np.concatenate(rows, 0))
    print(a.out, len(rows), "rows")


def label(img, text):
    cv2.rectangle(img, (0, img.shape[0] - 20), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    cv2.putText(
        img,
        text,
        (5, img.shape[0] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (80, 230, 255),
        1,
        cv2.LINE_AA,
    )
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("poses")
    q.add_argument("rec")
    q.add_argument("out")
    q.add_argument("--count", type=int, default=8)
    q.add_argument("--start", type=float)
    q.add_argument("--end", type=float)
    s = sub.add_parser("sheet")
    s.add_argument("dir")
    s.add_argument("out")
    s.add_argument("maps", nargs="+")
    s.add_argument("--labels", nargs="+")
    s.add_argument("--moments")
    s.add_argument("--eyes", default="centre,left,right,forward,turn")
    s.add_argument("--cell", type=int, default=480)
    s.add_argument("--scale", type=float, default=1.0)
    s.add_argument("--crop", help="x0,y0,x1,y1 fractions: a close-up region of every cell")
    a = p.parse_args()
    poses(a) if a.command == "poses" else sheet(a)


if __name__ == "__main__":
    main()
