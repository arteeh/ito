"""Sanity check of a recording: warp frame A into frame B by A's exact depth and both exact poses;
the warped image should line up with B. Writes OUT.png (B | warped A | |difference|)."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data import AXES, Recording  # noqa: E402


def warp(rec, a, b, size):
    K = rec.K_at(size)
    w, h = size
    depth = rec.depth(a, size)
    rgb_a = rec.rgb(a, size)
    u, v = np.meshgrid(np.arange(w) + 0.0, np.arange(h) + 0.0)
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    points = rays * depth[..., None]  # OpenCV camera A
    A = a["pose"] @ AXES
    B = b["pose"] @ AXES
    rel = np.linalg.inv(B) @ A
    p = points.reshape(-1, 3) @ rel[:3, :3].T + rel[:3, 3]
    z = p[:, 2]
    ok = (depth.ravel() > 0) & (z > 0.05)
    x = K[0, 0] * p[:, 0] / np.maximum(z, 1e-6) + K[0, 2]
    y = K[1, 1] * p[:, 1] / np.maximum(z, 1e-6) + K[1, 2]
    out = np.zeros((h, w, 3), np.uint8)
    zb = np.full((h, w), np.inf)
    xi, yi = np.rint(x).astype(int), np.rint(y).astype(int)
    ok &= (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(-z[idx])]
    out[yi[order], xi[order]] = rgb_a.reshape(-1, 3)[order]
    zb[yi[order], xi[order]] = z[order]
    return out, np.isfinite(zb)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("recording")
    p.add_argument("a", type=int, help="index into frames with depth")
    p.add_argument("b", type=int, help="index into frames with depth")
    p.add_argument("out")
    args = p.parse_args()
    rec = Recording(args.recording)
    deep = [f for f in rec.truth_frames() if f.get("depth")]
    a, b = deep[args.a], deep[args.b]
    size = (640, 360)
    warped, mask = warp(rec, a, b, size)
    target = rec.rgb(b, size)
    diff = np.where(mask[..., None], np.abs(warped.astype(int) - target.astype(int)), 0).astype(
        np.uint8
    )
    moved = np.linalg.norm(a["pose"][:3, 3] - b["pose"][:3, 3])
    print(
        f"A seq {a['seq']} B seq {b['seq']} moved {moved:.3f} m, mean |diff| over warped "
        f"{diff[mask].mean():.1f}, covered {mask.mean():.2f}"
    )
    cv2.imwrite(args.out, np.concatenate([target, warped, diff * 3], 1)[..., ::-1])


if __name__ == "__main__":
    main()
