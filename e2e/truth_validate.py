"""Validate a source dataset against independent continuous surfaces and saved RGB hashes."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from truth_surface import Surface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    root = args.dataset
    model = mujoco.MjModel.from_binary_path(str(root / "scene.mjb"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    surface = Surface(model, data)
    k = json.loads((root / "camera.json").read_text())
    rows = [json.loads(line) for line in (root / "frames.jsonl").read_text().splitlines()]
    clouds, errors = [], []
    y, x = np.mgrid[: k["height"] : 8, : k["width"] : 8]
    rays = np.stack(((x - k["cx"]) / k["fx"], -(y - k["cy"]) / k["fy"], -np.ones_like(x)), -1)
    for row in rows[:: max(1, len(rows) // 30)]:
        rgb = np.asarray(Image.open(root / row["rgb"]))
        assert hashlib.sha256(rgb.tobytes()).hexdigest() == row["rgb_sha256"]
        depth = np.load(root / row["depth"])[::8, ::8]
        camera = np.asarray(row["camera"])
        points = (rays * depth[..., None]) @ camera[:3, :3].T + camera[:3, 3]
        valid = (depth > 0) & (depth < model.vis.map.zfar * model.stat.extent * 0.99)
        clouds.append(points[valid])
        for point in points[valid][::100]:
            ray = point - camera[:3, 3]
            length = np.linalg.norm(ray)
            hit = mujoco.mj_ray(
                model,
                data,
                camera[:3, 3],
                ray / length,
                np.array([1, 1, 0, 0, 1, 1], np.uint8),
                True,
                -1,
                np.zeros(1, np.int32),
            )
            if hit >= 0:
                errors.append(abs(hit - length))
    report = dict(
        frames=len(rows),
        seconds=rows[-1]["sim_time"] - rows[0]["sim_time"],
        geometry=surface.metrics(np.concatenate(clouds)),
        ray_p95_m=float(np.percentile(errors, 95)),
    )
    report["passed"] = report["geometry"]["ghost_fraction"] < 0.001 and report["ray_p95_m"] < 0.001
    (root / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    assert report["passed"], "Source geometry is not valid"


if __name__ == "__main__":
    main()
