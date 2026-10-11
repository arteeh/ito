"""Validate geometry against the Microduck room's rendered triangles and MuJoCo rays.

python e2e/truth_oracle.py --out ../out/oracle
This is a static geometry calibration, not a live Microduck or SLAM benchmark.
"""

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from truth_surface import Surface, wall_histogram

from drivers.microduck.scene import ASSETS, _draw_as_meshes


def room():
    scene = ET.parse(ASSETS / "room.xml")
    asset = scene.find("asset")
    for texture in asset.findall("texture"):
        if "file" in texture.attrib:
            texture.set("file", str((ASSETS / texture.get("file")).resolve()))
    stretched = {m.get("name") for m in asset.iter("material") if m.get("texuniform") == "false"}
    for body in scene.getroot().iter("body"):
        _draw_as_meshes(body, asset, stretched)
    world = scene.find("worldbody")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="2 1.8 .05",
        material="room_oak",
        pos=".7 0 0",
    )
    ET.SubElement(world, "light", pos=".3 -.5 2.1", dir=".2 .3 -1", diffuse=".65 .62 .56")
    ET.SubElement(world, "camera", name="truth", pos="0 0 .2", fovy="65")
    visual = ET.SubElement(scene.getroot(), "visual")
    ET.SubElement(visual, "global", offwidth="640", offheight="480")
    ET.SubElement(visual, "map", znear=".0003", zfar="10")
    # Multisampled depth resolves a subpixel minimum, not the pixel-centre ray.
    ET.SubElement(visual, "quality", offsamples="0")
    ET.SubElement(scene.getroot(), "statistic", center=".7 0 .5", extent="4")
    model = mujoco.MjModel.from_xml_string(ET.tostring(scene.getroot(), encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def unproject(depth, camera, focal):
    height, width = depth.shape
    y, x = np.indices(depth.shape)
    local = np.stack(
        ((x + 0.5 - width / 2) / focal, (height / 2 - y - 0.5) / focal, -np.ones_like(x)), -1
    )
    return (local * depth[..., None]) @ camera[:3, :3].T + camera[:3, 3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    model, data = room()
    surface = Surface(model, data)
    width, height = 320, 180
    focal = height / (2 * math.tan(math.radians(65) / 2))
    option = mujoco.MjvOption()
    option.geomgroup[:] = [1, 1, 0, 0, 1, 1]
    renderer = mujoco.Renderer(model, height=height, width=width)
    rows, clouds = [], []
    try:
        for i, yaw in enumerate(np.linspace(0, math.tau, 8, endpoint=False)):
            # Camera right, up, back; MuJoCo world Z is up.
            rotation = np.array(
                [[math.sin(yaw), 0, -math.cos(yaw)], [-math.cos(yaw), 0, -math.sin(yaw)], [0, 1, 0]]
            )
            position = np.array([0.3, -0.1, 0.22 if i % 2 == 0 else 0.65])
            camera = np.eye(4)
            camera[:3, :3], camera[:3, 3] = rotation, position
            model.cam_pos[0] = position
            mujoco.mju_mat2Quat(model.cam_quat[0], rotation.ravel())
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=0, scene_option=option)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            renderer.disable_depth_rendering()
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            depth = renderer.render().copy()
            points = unproject(depth, camera, focal)
            valid = np.isfinite(depth) & (depth > 0) & (depth < 39)
            selected = valid.copy()
            selected[:] = False
            selected[::4, ::4] = valid[::4, ::4]
            sample = points[selected]
            distance = surface.distance(sample)
            # Independent physics ray intersections catch flips, range-vs-Z and pose errors.
            ray_errors = []
            for point in sample[::20]:
                ray = point - position
                length = np.linalg.norm(ray)
                hit = mujoco.mj_ray(
                    model,
                    data,
                    position,
                    ray / length,
                    option.geomgroup,
                    True,
                    -1,
                    np.zeros(1, np.int32),
                )
                if hit >= 0:
                    ray_errors.append(abs(hit - length))
            clouds.append(sample)
            row = surface.metrics(sample)
            row.update(view=i, ray_error_p95_m=float(np.percentile(ray_errors, 95)))
            rows.append(row)
            error = np.zeros_like(rgb)
            error[selected] = np.uint8(np.clip(distance[:, None] / 0.05, 0, 1) * [255, 0, 0])
            sheet = Image.new("RGB", (width * 2, height + 24))
            sheet.paste(Image.fromarray(rgb), (0, 24))
            sheet.paste(Image.fromarray(error), (width, 24))
            ImageDraw.Draw(sheet).text((5, 5), "MuJoCo truth | surface error (red = 5 cm)")
            sheet.save(args.out / f"oracle-view-{i:02d}.png")
    finally:
        renderer.close()
    cloud = np.concatenate(clouds)
    y, z = np.meshgrid(np.linspace(-1, 1, 200), np.linspace(1.65, 2.1, 80))
    wall = np.column_stack((np.full(y.size, 2.7), y.ravel(), z.ravel()))
    duplicate = wall - [0.2, 0, 0]
    wall_args = dict(axis=0, face=2.7, bounds=((-1, 1), (1.65, 2.1)), inward=-1)
    clean = wall_histogram(wall, **wall_args)
    doubled = wall_histogram(np.concatenate((wall, duplicate)), **wall_args)
    report = dict(
        scope="Static Microduck room geometry calibration; no live capture or SLAM",
        room_sha256=hashlib.sha256((ASSETS / "room.xml").read_bytes()).hexdigest(),
        mujoco_version=mujoco.__version__,
        depth_multisampling=0,
        units="metres; MuJoCo world; camera OpenGL axes; depth axial -Z",
        surface="exact rendered triangles plus infinite floor plane",
        O1_geometry=surface.metrics(cloud),
        views=rows,
        injected_wall=surface.metrics(np.concatenate((wall, duplicate))),
        clean_wall=clean,
        doubled_wall=doubled,
    )
    histogram = Image.new("RGB", (800, 300), "white")
    draw = ImageDraw.Draw(histogram)
    for column, (name, result) in enumerate((("Clean wall", clean), ("20 cm duplicate", doubled))):
        left = column * 400 + 35
        draw.text((left, 8), name + " (signed distance, metres)", fill="black")
        counts = np.array(result["counts"])
        for index, count in enumerate(counts):
            x = left + index * 2
            draw.line((x, 260, x, 260 - 220 * count / max(1, counts.max())), fill="navy")
        draw.text((left, 275), "-0.4                0                +0.4", fill="black")
    histogram.save(args.out / "injected-wall-histograms.png")
    report["passed"] = bool(
        report["O1_geometry"]["ghost_fraction"] < 0.001
        and report["O1_geometry"]["p95_m"] < 0.001
        and max(r["ray_error_p95_m"] for r in rows) < 0.001
        and not clean["doubled"]
        and doubled["doubled"]
        and report["injected_wall"]["ghost_fraction"] > 0.45
    )
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"views", "clean_wall", "doubled_wall"}},
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit("Geometry oracle failed; do not use its scores")


if __name__ == "__main__":
    main()
