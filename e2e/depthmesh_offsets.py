"""Stress retained coverage at 102-degree FOV, 60-degree turns and 30cm translation."""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from depthmesh_evaluate import sheet
from measure_gl import View, projection
from measure_images import Metrics, Room
from PIL import Image
from scipy.spatial.transform import Rotation
from truth import align

from ito.proto.depthmesh import MeshWorld
from ito.protocol import Intrinsics


def main():
    root = Path("../p47-out").resolve()
    work = Path("..").resolve()
    out = root / "offsets"
    out.mkdir(exist_ok=True)
    i = Intrinsics(
        width=640,
        height=360,
        fx=360 / (2 * np.tan(np.radians(35))),
        fy=360 / (2 * np.tan(np.radians(35))),
        cx=320,
        cy=180,
    )
    room = Room(work / "truth/data/scan/scene.mjb", i)
    main_view = View(i)
    ctx = main_view.context
    with ctx:
        mesh = MeshWorld.load(ctx, root / "rendered/metric-small-robot-cap64/sweep180-world.npz")
        target = ctx.simple_framebuffer((i.width, i.height))
    snap = np.load(work / "m1-out/verified/repeat-1/sweep180/snapshot-000692.npz")
    poses = np.load(work / "gs/out/sweep180/data.npz")["poses"]
    world = align(snap["first_pose"], poses[int(snap["first_index"])])
    main_view.upload(snap["records"])
    metrics = Metrics()
    panels, captions, rows = [], [], []
    for yaw, offset in ((0, 0), (-60, 0), (60, 0), (0, -0.3), (0, 0.3)):
        camera = poses[692].copy()
        camera[:3, 3] += camera[:3, 0] * offset
        camera[:3, :3] = camera[:3, :3] @ Rotation.from_euler("y", yaw, degrees=True).as_matrix()
        truth = room.draw(camera)
        main, main_covered = main_view.draw(world @ camera)
        with ctx:
            mesh.draw(camera, projection(i), target)
            rgba = (
                np.frombuffer(target.read(components=4), "u1")
                .reshape(i.height, i.width, 4)[::-1]
                .copy()
            )
        candidate = rgba[:, :, :3].astype("f4") / 255
        covered = rgba[:, :, 3] > 127
        # Wider-angle coverage only; the reference eye runs have independent common-seen masks.
        scores = metrics.score(truth, candidate, covered, np.ones(covered.shape, bool))
        for key in (
            "previously_seen_holes",
            "common_visible_fraction",
            "never_seen_fraction",
            "never_seen_covered_fraction",
        ):
            scores.pop(key)
        rows.append(
            dict(
                yaw_deg=yaw,
                translation_m=offset,
                main_coverage=float(main_covered.mean()),
                mesh_coverage=float(covered.mean()),
                **scores,
            )
        )
        image = (
            (np.concatenate((truth, main, candidate), axis=1) * 255)
            .round()
            .clip(0, 255)
            .astype("u1")
        )
        panels.append(image)
        captions.append(
            f"sweep #693 | yaw {yaw} deg, shift {offset:+.2f}m, wide FOV | truth / main / mesh"
        )
        Image.fromarray(image).save(out / f"yaw{yaw}-shift{offset:+.2f}.png")
    sheet(panels, captions, out / "wide-offsets.jpg")
    (out / "metrics.json").write_text(
        json.dumps(
            dict(scope="Wide/off-axis static truth; coverage, not prior-seen mask", rows=rows),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(rows, indent=2))
    with ctx:
        mesh.close()
        target.release()
    room.renderer.close()
    main_view.close()


if __name__ == "__main__":
    main()
