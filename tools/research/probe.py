"""Where is something a map shows from an eye? Renders MAP from eyes.json item ITEM, reads the
expected depth inside a pixel box, and lists the 3D spot and the splats there (when the map npz
carries captures/sequences from replay.py).

    probe.py EYES_DIR ITEM MAP.npz U0 U1 V0 V1 [--rec REC]"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from data import AXES  # noqa: E402

C0 = 0.28209479177387814


def main():
    p = argparse.ArgumentParser()
    p.add_argument("eyes")
    p.add_argument("item", type=int)
    p.add_argument("map")
    p.add_argument("box", type=int, nargs=4)
    p.add_argument("--rec")
    a = p.parse_args()
    from gsplat import rasterization

    spec = json.loads((Path(a.eyes) / "eyes.json").read_text())
    item = spec["items"][a.item]
    K = np.array(spec["K"])
    w, h = spec["size"]
    data = np.load(a.map)
    r = torch.as_tensor(data["records"], device="cuda", dtype=torch.float32)
    pose = np.array(item["pose"])
    view = torch.as_tensor(np.linalg.inv(pose @ AXES), dtype=torch.float32, device="cuda")
    with torch.no_grad():
        img, alpha, _ = rasterization(
            r[:, 0, :3], torch.nn.functional.normalize(r[:, 2], dim=-1), r[:, 1, :3],
            r[:, 0, 3].clamp(0, 1), r[:, 3:4, :3], view[None],
            torch.as_tensor(K, dtype=torch.float32, device="cuda")[None], w, h,
            sh_degree=0, render_mode="RGB+ED", near_plane=0.02, far_plane=100,
        )
    import cv2

    full = img[0, ..., :3].clamp(0, 1).cpu().numpy()
    a_img = alpha[0, ..., 0].cpu().numpy()
    sheet = np.concatenate([full, np.repeat(a_img[..., None], 3, -1)], 1)
    cv2.imwrite(str(Path(a.map).with_suffix(f".probe{a.item}.png")), (sheet[..., ::-1] * 255).astype(np.uint8))
    print("alpha in box p10/50/90", np.percentile(a_img[a.box[2]:a.box[3], a.box[0]:a.box[1]], [10, 50, 90]).round(2))
    u0, u1, v0, v1 = a.box
    depth = img[0, v0:v1, u0:u1, 3].cpu().numpy()
    colour = img[0, v0:v1, u0:u1, :3].cpu().numpy()
    z = float(np.median(depth))
    u, v = (u0 + u1) / 2, (v0 + v1) / 2
    local = np.array([(u - K[0, 2]) / K[0, 0] * z, -(v - K[1, 2]) / K[1, 1] * z, -z])
    spot = pose[:3, :3] @ local + pose[:3, 3]
    print(f"box colour {colour.reshape(-1, 3).mean(0).round(2)} depth p10/50/90 "
          f"{np.percentile(depth, [10, 50, 90]).round(2)} -> spot {spot.round(2)}")
    rec = data["records"]
    near = np.linalg.norm(rec[:, 0, :3] - spot, axis=1) < 0.08
    print("splats within 8 cm:", int(near.sum()))
    if "captures" in data.files and near.any():
        caps = data["captures"][near]
        seqs = data["sequences"][near]
        print("their seed/update capture times (relative to the earliest in the map):")
        t0 = data["captures"][data["captures"] > 0].min()
        hist, edges = np.histogram(caps - t0, bins=12)
        for c, e0, e1 in zip(hist, edges[:-1], edges[1:], strict=True):
            if c:
                print(f"  {e0:6.1f}..{e1:6.1f} s: {c}")
        print("sequences (most common):", np.unique(seqs, return_counts=True)[0][:10])
        lum = (0.5 + C0 * rec[near, 3, :3]).mean(1)
        print("luminance p10/50/90", np.percentile(lum, [10, 50, 90]).round(2),
              "opacity p50", np.median(rec[near, 0, 3]).round(2),
              "scale p50", np.median(rec[near, 1, :3], 0).round(4))


if __name__ == "__main__":
    main()
