"""CPU RGB-D pose-policy A/B using #32's independent MuJoCo surface scorer.

Requires e2e/scene_truth.py from fix/scene-truth (kept separate from #33).
uv run --with scipy python e2e/imu_scene.py
Noise is an explicit experiment, not an assertion about hardware sensor accuracy.
"""

import json
from pathlib import Path

import numpy as np
from scene_truth import Room, geometry, points, unique
from scipy.spatial import cKDTree
from splat_quality import drive

from ito.reconstruction.prior import blend, turn
from ito.reconstruction.rgbd import RGBDBackend


def main():
    out = Path("e2e/out/imu_scene")
    out.mkdir(parents=True, exist_ok=True)
    room = Room()
    rng = np.random.default_rng(33)
    names = ("truth", "noisy_prior", "visual", "gated")
    backends = {name: RGBDBackend(131072, room.intrinsics, device="cpu") for name in names}
    surfaces = []
    rejected = 0
    yaw = 0.0
    try:
        for index, t in enumerate(np.linspace(0, 20, 160)):
            rgb, depth, truth = room.view((*drive(t, 20), 0.19, 0, 0))
            surfaces.append(points(depth, truth, 65))
            if len(surfaces) >= 20:
                surfaces = [unique(np.concatenate(surfaces))]
            prior = truth.copy()
            prior[:3, :3] = turn(np.radians(rng.normal(0, 1) + 0.05 * t)) @ truth[:3, :3]
            prior[:3, 3] += rng.normal(0, 0.005, 3) + np.array([0.001 * t, 0, 0])
            visual = truth.copy()
            # Ambiguous visual registration: occasional 25-degree wrong-room matches.
            visual[:3, :3] = (
                turn(np.radians(25 if index % 10 == 5 else rng.normal(0, 3))) @ truth[:3, :3]
            )
            visual[:3, 3] += rng.normal(0, 0.01, 3)
            fused, yaw = blend(prior, visual, yaw, 1 / 8)
            rejected += fused is None
            for name, pose in zip(names, (truth, prior, visual, fused), strict=True):
                backend = backends[name]
                backend.expire(t, float("inf"))
                if pose is not None:
                    backend.integrate(rgb, depth, pose, t)
        surface = unique(np.concatenate(surfaces))
        tree = cKDTree(surface)
        results = {}
        for name, backend in backends.items():
            live = (backend.keys >= 0) & ~backend.retiring
            centres = backend.records[live, 0, :3]
            results[name] = geometry(centres, surface, tree, cKDTree(centres))
        summary = dict(
            seed=33,
            frames=160,
            rejected=int(rejected),
            noise=(
                "prior: 1 deg yaw RMS + 0.05 deg/s drift, 5 mm RMS + 1 mm/s drift; "
                "visual: 3 deg/1 cm RMS, 25 deg outlier every 10 frames"
            ),
            results=results,
        )
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        assert rejected >= 16, summary
        assert results["gated"]["ghost_fraction"] < results["visual"]["ghost_fraction"], summary
    finally:
        room.close()


if __name__ == "__main__":
    main()
