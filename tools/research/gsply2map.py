"""A standard 3DGS PLY (an outside system's map) as Ito splat records in the room's frame.

    gsply2map.py IN.ply PLACEMENT.npz OUT.npz [--sh 0]

PLACEMENT: an export.py import npz of the same system's trajectory (Ito world = s R x + t of the
system's world, OpenCV axes; Gaussians are axis-free, so only the similarity applies)."""

import argparse

import numpy as np
from plyfile import PlyData
from scipy.spatial.transform import Rotation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ply")
    p.add_argument("placement")
    p.add_argument("out")
    p.add_argument("--sh", type=int, default=0)
    a = p.parse_args()
    v = PlyData.read(a.ply)["vertex"].data
    names = set(v.dtype.names)
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    opacity = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
    scales = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1).astype(np.float64))
    rot = np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(np.float64)  # wxyz
    rot /= np.linalg.norm(rot, axis=1, keepdims=True)
    dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], 1)
    g = np.load(a.placement)
    s, R, t = float(g["scale"]), g["rotation"], g["translation"]
    xyz = (s * (R @ xyz.T)).T + t
    scales = scales * s
    turned = Rotation.from_matrix(R) * Rotation.from_quat(rot[:, [1, 2, 3, 0]])
    rot = turned.as_quat()[:, [3, 0, 1, 2]]
    n = len(xyz)
    coefficients = (a.sh + 1) ** 2
    records = np.zeros((n, 3 + coefficients, 4), np.float32)
    records[:, 0, :3] = xyz
    records[:, 0, 3] = opacity
    records[:, 1, :3] = scales
    records[:, 2] = rot
    records[:, 3, :3] = dc
    if coefficients > 1:
        rest = sorted([k for k in names if k.startswith("f_rest_")], key=lambda k: int(k[7:]))
        extra = np.stack([v[k] for k in rest], 1).reshape(n, 3, -1).transpose(0, 2, 1)
        # Rotating SH bands with the similarity is not done: degree 0 only is exact.
        records[:, 4:, :3] = extra[:, : coefficients - 1]
    keep = np.isfinite(records).all(axis=(1, 2))
    np.savez(a.out, records=records[keep])
    print(int(keep.sum()), "Gaussians")


if __name__ == "__main__":
    main()
