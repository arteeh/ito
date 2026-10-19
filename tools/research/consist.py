"""How much do two views' depth guesses disagree about the same surface? (CPU, supporting numbers)

    consist.py REC DUMP [--pairs 300] [--min-gap 2] [--depth estimated|mapped|truth]

For random pairs of dumped frames (ITO_DUMP) more than --min-gap seconds apart that look at the same
place, each pixel of A is placed in 3D with A's depth and the EXACT pose, projected into B, and its
depth there compared with B's own depth at that pixel (both exact poses: only depth disagreement
remains). Reports the median and p90 of |dA->B - dB| / dB over pixels both see (a 3D gap of that share
of the distance = a doubled surface), and the same for the exact depth (the floor of the method:
resampling, occlusion edges)."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data import Recording  # noqa: E402


def load(dump, rec, kind):
    by = {f["seq"]: f for f in rec.truth_frames()}
    out = []
    for path in sorted(Path(dump).glob("*.npz")):
        seq = int(path.stem) - 1
        if seq not in by:
            continue
        d = np.load(path)
        K = d["intrinsics"]
        depth = d["estimated"] if kind == "estimated" else d["mapped"]
        h, w = depth.shape
        truth = rec.depth(by[seq], (w, h)) if by[seq].get("depth") else None
        # The dump's grid (cloud intrinsics) is the frame's rectified pinhole scaled to (w, h).
        out.append(dict(seq=seq, t=by[seq]["received"], pose=by[seq]["pose"], K=K, depth=depth,
                        truth=truth))
    return out


def disagreement(a, b, key):
    da, db = a[key], b[key]
    if da is None or db is None:
        return None
    fx, fy, cx, cy = a["K"]
    h, w = da.shape
    v, u = np.mgrid[:h, :w]
    ok = da > 0.05
    z = da[ok]
    # Ito camera axes: x right, y up, looking down -z.
    pts = np.stack(((u[ok] - cx) / fx * z, -(v[ok] - cy) / fy * z, -z), -1)
    world = pts @ a["pose"][:3, :3].T + a["pose"][:3, 3]
    local = (world - b["pose"][:3, 3]) @ b["pose"][:3, :3]
    zb = -local[:, 2]
    fx, fy, cx, cy = b["K"]
    ub = np.rint(cx + fx * local[:, 0] / np.maximum(zb, 1e-6)).astype(int)
    vb = np.rint(cy - fy * local[:, 1] / np.maximum(zb, 1e-6)).astype(int)
    inside = (zb > 0.05) & (ub >= 0) & (ub < w) & (vb >= 0) & (vb < h)
    if inside.sum() < 2000:
        return None
    seen = db[vb[inside], ub[inside]]
    good = seen > 0.05
    # The nearer of the two is the visible one; only compare where they could be one surface
    # (within 30%): beyond that it is occlusion, not disagreement.
    rel = (zb[inside][good] - seen[good]) / seen[good]
    rel = rel[np.abs(rel) < 0.3]
    return rel if len(rel) > 1000 else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rec")
    p.add_argument("dump")
    p.add_argument("--pairs", type=int, default=300)
    p.add_argument("--min-gap", type=float, default=2.0)
    p.add_argument("--depth", default="estimated")
    a = p.parse_args()
    rec = Recording(a.rec)
    frames = load(a.dump, rec, a.depth)
    rng = np.random.default_rng(0)
    rows = {"network": [], "exact": []}
    tried = 0
    while len(rows["network"]) < a.pairs and tried < 50 * a.pairs:
        tried += 1
        i, j = rng.integers(0, len(frames), 2)
        A, B = frames[i], frames[j]
        if abs(A["t"] - B["t"]) < a.min_gap:
            continue
        if A["pose"][:3, 2] @ B["pose"][:3, 2] < np.cos(np.radians(40)):
            continue
        r = disagreement(A, B, "depth")
        if r is None:
            continue
        rows["network"].append((np.median(np.abs(r)), np.percentile(np.abs(r), 90), np.median(r)))
        rt = disagreement(A, B, "truth")
        if rt is not None:
            rows["exact"].append((np.median(np.abs(rt)), np.percentile(np.abs(rt), 90), np.median(rt)))
    for k, v in rows.items():
        if v:
            v = np.array(v)
            print(f"{k:8s} pairs {len(v):4d}: |rel| median {100 * np.median(v[:, 0]):.1f}%  "
                  f"p90 {100 * np.median(v[:, 1]):.1f}%  signed bias per pair: median "
                  f"{100 * np.median(v[:, 2]):+.1f}%, spread (p10..p90) "
                  f"{100 * np.percentile(v[:, 2], 10):+.1f}..{100 * np.percentile(v[:, 2], 90):+.1f}%")


if __name__ == "__main__":
    main()
