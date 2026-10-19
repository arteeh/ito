"""A point-cloud reconstruction (PLY: x y z red green blue) as small Gaussians in the room's frame, to
look at it with the same eye tests as splat maps.

    ply2map.py IN.ply PLACEMENT.npz OUT.npz [--size 0.006] [--voxel 0.004]

PLACEMENT: an export.py import npz (scale, rotation, translation: Ito world = s R x + t)."""

import argparse

import numpy as np
from plyfile import PlyData

C0 = 0.28209479177387814


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ply")
    p.add_argument("placement")
    p.add_argument("out")
    p.add_argument("--size", type=float, default=0.006)
    p.add_argument("--voxel", type=float, default=0.004)
    a = p.parse_args()
    v = PlyData.read(a.ply)["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    names = v.dtype.names
    rgb = np.stack([v[n] for n in ("red", "green", "blue")], 1).astype(np.float32)
    if rgb.max() > 1.5:
        rgb /= 255
    g = np.load(a.placement)
    xyz = (g["scale"] * (g["rotation"] @ xyz.T)).T + g["translation"]
    keys = np.floor(xyz / a.voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    xyz, rgb = xyz[first], rgb[first]
    n = len(xyz)
    records = np.zeros((n, 4, 4), np.float32)
    records[:, 0, :3] = xyz
    records[:, 0, 3] = 0.9
    records[:, 1, :3] = a.size
    records[:, 2, 0] = 1
    records[:, 3, :3] = (rgb - 0.5) / C0
    np.savez(a.out, records=records)
    print(n, "points", names)


if __name__ == "__main__":
    main()
