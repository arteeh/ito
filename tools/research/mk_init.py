"""Initial points for gs.py from Ito's own per-frame depth (an ITO_DUMP folder written during a
replay: one npz per mapped frame, named source sequence = recording seq + 1).

    mk_init.py DUMP REC OUT.npz [--depth estimated|mapped] [--poses truth|dump] [--voxel 0.01]

estimated: the network's depth at the frame's metric scale; mapped: what Ito actually seeded
(after consensus/fusion). Poses: the exact ones (to isolate depth error) or the dump's own."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data import Recording  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dump")
    p.add_argument("rec")
    p.add_argument("out")
    p.add_argument("--depth", default="estimated")
    p.add_argument("--poses", default="truth")
    p.add_argument("--voxel", type=float, default=0.01)
    a = p.parse_args()
    rec = Recording(a.rec)
    by_seq = {f["seq"]: f for f in rec.truth_frames()}
    points, colors = [], []
    for path in sorted(Path(a.dump).glob("*.npz")):
        d = np.load(path)
        seq = int(path.stem) - 1
        if a.poses == "truth":
            if seq not in by_seq:
                continue
            pose = by_seq[seq]["pose"]
        else:
            pose = d["pose"]
        depth = d[a.depth]
        image = d["image"].astype(np.float32) / 255
        fx, fy, cx, cy = d["intrinsics"]
        h, w = depth.shape
        v, u = np.mgrid[:h, :w].astype(np.float64)
        ok = (depth > 0.05) & (depth < 8)
        local = np.stack(((u - cx) / fx * depth, -(v - cy) / fy * depth, -depth), -1)[ok]
        points.append(local @ pose[:3, :3].T + pose[:3, 3])
        colors.append(image[ok])
    points, colors = np.concatenate(points), np.concatenate(colors)
    keys = np.floor(points / a.voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    np.savez(a.out, points=points[first].astype(np.float32), colors=colors[first].astype(np.float32))
    print(len(first), "points from", len(list(Path(a.dump).glob("*.npz"))), "frames")


if __name__ == "__main__":
    main()
