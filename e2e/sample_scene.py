"""Generate a deterministic standard 3DGS PLY of a colored sculpture courtyard."""

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def write_scene(path: Path, *, text: bool = False, degree: int = 3) -> int:
    rng = np.random.default_rng(8)
    points, colors, scales, rotations = [], [], [], []

    def add(xyz, rgb, scale, quaternion=(1, 0, 0, 0)):
        points.append(xyz)
        colors.append(rgb)
        scales.append(scale)
        rotations.append(quaternion)

    # Checkerboard floor with splats shaped like small horizontal tiles.
    for x in np.linspace(-4, 4, 45):
        for z in np.linspace(-9, -1, 45):
            light = (int(np.floor(x)) + int(np.floor(z))) % 2
            add(
                (x, -1.1, z),
                (0.24, 0.31, 0.39) if light else (0.10, 0.17, 0.23),
                (0.095, 0.015, 0.095),
            )
    for center, rgb, radius in (
        ((-1.15, -0.15, -3.5), (0.95, 0.24, 0.07), 0.78),
        ((1.1, -0.4, -4.4), (0.05, 0.65, 0.92), 0.65),
        ((0.05, 0.9, -5.1), (0.91, 0.69, 0.08), 0.6),
    ):
        for _ in range(800):
            normal = rng.normal(size=3)
            normal /= np.linalg.norm(normal)
            p = np.array(center) + normal * radius
            shade = 0.60 + 0.4 * max(0, float(normal @ np.array((-0.4, 0.8, 0.4))))
            add(p, np.array(rgb) * shade, (0.055, 0.055, 0.055))
    # Tilted, anisotropic green rods behind the spheres.
    for side in (-1, 1):
        for y in np.linspace(-0.9, 1.8, 35):
            add(
                (side * (2.1 - 0.25 * y), y, -6),
                (0.08, 0.7, 0.35),
                (0.045, 0.12, 0.045),
                (np.cos(0.13), 0, 0, side * np.sin(0.13)),
            )
    rest_count = 3 * ((degree + 1) ** 2 - 1)
    names = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
    names += [f"f_rest_{i}" for i in range(rest_count)]
    names += ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    vertices = np.zeros(len(points), dtype=[(name, "f4") for name in names])
    for i, name in enumerate(("x", "y", "z")):
        vertices[name] = np.asarray(points)[:, i]
    for i in range(3):
        vertices[f"f_dc_{i}"] = (np.asarray(colors)[:, i] - 0.5) / 0.2820947918
        vertices[f"scale_{i}"] = np.log(np.asarray(scales)[:, i])
    if rest_count:
        for channel in range(3):
            vertices[f"f_rest_{channel * (rest_count // 3) + 2}"] = 0.06
    for i in range(4):
        vertices[f"rot_{i}"] = np.asarray(rotations)[:, i]
    vertices["opacity"] = 3.0
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")], text=text, byte_order="<").write(path)
    return len(points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(f"Wrote {write_scene(args.path)} Gaussians to {args.path}")
