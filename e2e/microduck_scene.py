"""Render the bundled room and verify Pollen's robot and contact floor stay unchanged.

uv run --extra microduck python e2e/microduck_scene.py --rl /path/to/microduck_rl
"""

import argparse
import json
import sys
import tempfile
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import cv2
import mujoco
import numpy as np
from PIL import Image

from drivers.microduck.scene import furnish

OUT = Path("e2e/out/microduck-scene")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl", type=Path, required=True)
    args = parser.parse_args()
    source = args.rl.resolve() / "src/mjlab_microduck/robot/microduck/scene.xml"
    OUT.mkdir(parents=True, exist_ok=True)
    original = mujoco.MjModel.from_xml_path(str(source))
    with tempfile.TemporaryDirectory(prefix="ito-room-") as temp:
        state = Path(temp)
        for asset in source.parent.iterdir():
            if asset.name not in {"scene.xml", "robot_groundcontact.xml"}:
                (state / asset.name).symlink_to(asset, target_is_directory=asset.is_dir())
        robot = ET.parse(source.parent / "robot_groundcontact.xml")
        for geom in robot.findall(".//worldbody//geom"):
            geom.set("group", "2" if geom.get("class") == "visual" else "3")
        robot.write(state / "robot_groundcontact.xml")
        scene = furnish(source)
        ET.SubElement(scene.find("visual"), "quality", shadowsize="256", offsamples="1")
        scene.find("visual/global").set("offwidth", "1280")
        scene.find("visual/global").set("offheight", "960")
        scene.write(state / "scene.xml")
        model = mujoco.MjModel.from_xml_path(str(state / "scene.xml"))
    # Model arrays compare the real compiled assets, including all robot inertials and actuators.
    preserved = (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "jnt_range",
        "jnt_pos",
        "jnt_axis",
        "dof_damping",
        "dof_frictionloss",
        "dof_armature",
        "actuator_gear",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_ctrlrange",
        "key_qpos",
        "key_ctrl",
    )
    for field in preserved:
        expected, actual = getattr(original, field), getattr(model, field)
        np.testing.assert_array_equal(actual[: len(expected)], expected, err_msg=field)
    assert (model.nq, model.nv, model.nu) == (original.nq, original.nv, original.nu)
    for field in ("geom_type", "geom_friction", "geom_condim", "geom_contype", "geom_conaffinity"):
        expected = getattr(original, field)[original.geom("floor").id]
        actual = getattr(model, field)[model.geom("floor").id]
        np.testing.assert_array_equal(actual, expected, err_msg=field)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("STAND").id)
    mujoco.mj_forward(model, data)
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    option.geomgroup[5] = 0
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.65, 0.1, 0.42]
    camera.distance, camera.azimuth, camera.elevation = 5.9, 45, -42
    with mujoco.Renderer(model, height=960, width=1280) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        Image.fromarray(renderer.render()).save(OUT / "room-overview.png")
    sys.path.insert(0, str(args.rl.resolve() / "src"))
    from drivers.microduck.sim_body import HeadCamera

    head = HeadCamera(model, "head_camera")
    world = SimpleNamespace(data=data, lock=threading.Lock())
    try:
        for name, yaw in (
            ("forward", 0),
            ("left", np.pi / 2),
            ("right", -np.pi / 2),
            ("back", np.pi),
        ):
            data.qpos[:] = model.key("STAND").qpos
            data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            mujoco.mj_forward(model, data)
            head.render(world)
            packed = np.frombuffer(head.latest, dtype=np.uint8).reshape(360, 640, 2)
            rgb = cv2.cvtColor(packed, cv2.COLOR_YUV2RGB_UYVY)
            Image.fromarray(np.rot90(rgb, k=3)).save(OUT / f"head-{name}.png")
    finally:
        head.renderer.close()
    summary = {
        "robot_physics_preserved": list(preserved),
        "original_nq_nv_nu": [original.nq, original.nv, original.nu],
        "room_nq_nv_nu": [model.nq, model.nv, model.nu],
        "floor_contact_preserved": True,
        "near_clip_m": model.vis.map.znear * model.stat.extent,
        "room_geoms": model.ngeom - original.ngeom,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"PASS: room and unchanged robot rendered to {OUT}")


if __name__ == "__main__":
    main()
