"""Render the sim's room (robot hidden) from arbitrary camera poses: the truth for eye tests.
Run with the sim's Python:

    MUJOCO_GL=osmesa SIM/microduck_rl/.venv/bin/python truth_poses.py RL POSES.json OUT_DIR

POSES.json: {"width": W, "height": H, "fovy": degrees, "poses": [[16 floats, row-major
camera-to-world in MuJoCo's world, camera axes x right, y up, looking down -z], ...]}.
Writes OUT_DIR/<index>.png."""

import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from drivers.microduck.scene import furnish  # noqa: E402


def model_with_eye(rl):
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
        ET.SubElement(scene.find("worldbody"), "camera", name="eye", pos="0 0 0", mode="fixed")
        scene.write(state / "scene.xml")
        return mujoco.MjModel.from_xml_path(str(state / "scene.xml"))


def main():
    rl, spec, out = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    model = model_with_eye(rl)
    w, h = spec["width"], spec["height"]
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, w)
    model.vis.global_.offheight = max(model.vis.global_.offheight, h)
    eye = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "eye")
    model.cam_fovy[eye] = spec["fovy"]
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=h, width=w)
    option = mujoco.MjvOption()
    option.geomgroup[2] = 0
    option.geomgroup[5] = 1
    for i, flat in enumerate(spec["poses"]):
        m = np.array(flat, float).reshape(4, 4)
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, m[:3, :3].ravel())
        model.cam_pos[eye] = m[:3, 3]
        model.cam_quat[eye] = q
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=eye, scene_option=option)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        cv2.imwrite(str(out / f"{i}.png"), renderer.render()[..., ::-1])
    print(len(spec["poses"]), "views")


if __name__ == "__main__":
    main()
