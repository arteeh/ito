"""Render the virtual Microduck's room (the sim's own furnished scene, robot hidden) from the fixed
viewpoints in views.py: the truth column of every comparison sheet. Run with the sim's Python:

    MUJOCO_GL=osmesa SIM/microduck_rl/.venv/bin/python tools/research/truth_views.py RL OUT_DIR

RL: the sim prefix's microduck_rl checkout. Writes OUT_DIR/<index>.png per view."""

import math
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from views import HEIGHT, HFOV, VIEWS, WIDTH  # noqa: E402

from drivers.microduck.scene import furnish  # noqa: E402


def main():
    rl, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    source = rl / "src/mjlab_microduck/robot/microduck/scene.xml"
    with tempfile.TemporaryDirectory() as temporary:
        state = Path(temporary)
        for asset in source.parent.iterdir():
            if asset.name not in {"scene.xml", "robot_groundcontact.xml"}:
                (state / asset.name).symlink_to(asset, target_is_directory=asset.is_dir())
        robot = ET.parse(source.parent / "robot_groundcontact.xml")
        for geom in robot.findall(".//worldbody//geom"):
            geom.set("group", "2" if geom.get("class") == "visual" else "3")
        robot.write(state / "robot_groundcontact.xml")
        scene = furnish(source)
        ET.SubElement(scene.find("visual"), "quality", shadowsize="256", offsamples="4")
        scene.write(state / "scene.xml")
        model = mujoco.MjModel.from_xml_path(str(state / "scene.xml"))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, HEIGHT)
    model.vis.global_.fovy = math.degrees(
        2 * math.atan(math.tan(math.radians(HFOV / 2)) * HEIGHT / WIDTH)
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    option = mujoco.MjvOption()
    option.geomgroup[2] = 0
    option.geomgroup[5] = 1
    for i, (name, position, target) in enumerate(VIEWS):
        p, t = np.array(position), np.array(target)
        d = t - p
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = t
        camera.distance = float(np.linalg.norm(d))
        camera.azimuth = math.degrees(math.atan2(d[1], d[0]))
        camera.elevation = math.degrees(math.asin(d[2] / np.linalg.norm(d)))
        renderer.update_scene(data, camera=camera, scene_option=option)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        rgb = renderer.render()
        cv2.imwrite(str(out / f"{i}.png"), rgb[..., ::-1])
        print(i, name)


if __name__ == "__main__":
    main()
