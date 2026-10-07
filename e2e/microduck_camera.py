"""Render the original Microduck camera at controlled poses and isolate geom visibility.

LD_LIBRARY_PATH=/path/to/osmesa /path/to/rl/.venv/bin/python \
    e2e/microduck_camera.py --rl /path/to/rl
"""

import argparse
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("e2e/out/microduck-camera"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(args.rl.resolve() / "src")]

    import mujoco
    import numpy as np
    from mjlab_microduck.sim.body_server import HOME_TRUNK_Z, Body, World
    from PIL import Image, ImageDraw

    from drivers.microduck.sim_body import HeadCamera

    source = args.rl.resolve() / "src/mjlab_microduck/robot/microduck"
    with tempfile.TemporaryDirectory(dir=args.out, prefix="scene-") as temporary:
        state = Path(temporary)
        for asset in source.iterdir():
            if asset.name not in {"scene.xml", "robot_groundcontact.xml"}:
                (state / asset.name).symlink_to(asset, target_is_directory=asset.is_dir())
        robot = ET.parse(source / "robot_groundcontact.xml")
        for geom in robot.findall(".//worldbody//geom"):
            geom.set("group", "2" if geom.get("class") == "visual" else "3")
        robot.write(state / "robot_groundcontact.xml")
        scene = ET.parse(source / "scene.xml")
        ET.SubElement(scene.find("visual"), "quality", shadowsize="256", offsamples="1")
        scene.write(state / "scene.xml")

        world = World(state / "scene.xml")
        body = Body(world, 0)
        body.place(None, HOME_TRUNK_Z, 0)
        world.bodies.append(body)
        camera = HeadCamera(world.model, "head_camera")
        render = camera.renderer.render
        captured = []

        def capture_rgb(*args, **kwargs):
            rgb = render(*args, **kwargs)
            captured.append(rgb.copy())
            return rgb

        camera.renderer.render = capture_rgb
        home = world.data.qpos.copy()
        addresses = {
            name: world.model.jnt_qposadr[world.model.joint(name).id]
            for name in ("head_yaw", "head_pitch")
        }
        poses = [("home", 0.0, 0.3491)] + [
            (f"yaw{yaw:+.1f}-pitch{pitch:+.1f}", yaw, pitch)
            for yaw in (-1.0, 0.0, 1.0)
            for pitch in (-0.6, 0.0, 0.9)
        ]
        rows = []
        thumbnails = []
        try:
            for name, yaw, pitch in poses:
                world.data.qpos[:] = home
                world.data.qpos[addresses["head_yaw"]] = yaw
                world.data.qpos[addresses["head_pitch"]] = pitch
                mujoco.mj_forward(world.model, world.data)
                camera.render(world)
                actual = captured.pop()
                camera.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
                baseline = render()
                images = {"shadow-on": baseline, "actual": actual}
                for label, group3 in (("hide3", 0), ("show3", 1)):
                    option = mujoco.MjvOption()
                    option.geomgroup[2] = 0
                    option.geomgroup[3] = group3
                    with world.lock:
                        camera.renderer.update_scene(
                            world.data, camera=camera.camera, scene_option=option
                        )
                    images[label] = render()
                for label, flag in (
                    ("no-reflection", mujoco.mjtRndFlag.mjRND_REFLECTION),
                    ("no-shadow", mujoco.mjtRndFlag.mjRND_SHADOW),
                ):
                    option = mujoco.MjvOption()
                    option.geomgroup[2] = 0
                    with world.lock:
                        camera.renderer.update_scene(
                            world.data, camera=camera.camera, scene_option=option
                        )
                    previous = camera.renderer.scene.flags[flag]
                    camera.renderer.scene.flags[flag] = False
                    images[label] = render()
                    camera.renderer.scene.flags[flag] = previous
                old_shadow_size = world.model.vis.quality.shadowsize
                try:
                    world.model.vis.quality.shadowsize = 2048
                    with mujoco.Renderer(world.model, height=360, width=640) as detailed:
                        option = mujoco.MjvOption()
                        option.geomgroup[2] = 0
                        detailed.update_scene(world.data, camera=camera.camera, scene_option=option)
                        images["shadow2048"] = detailed.render()
                finally:
                    world.model.vis.quality.shadowsize = old_shadow_size
                # Preserve sensor orientation as well as upright inspection images.
                Image.fromarray(actual).save(args.out / f"{name}-sensor.png")
                row = {"pose": name, "yaw_rad": yaw, "pitch_rad": pitch}
                for label, rgb in images.items():
                    delta = np.abs(rgb.astype(int) - baseline.astype(int))
                    row[label] = {
                        "changed_pixels": int(np.any(delta, axis=2).sum()),
                        "max_channel_delta": int(delta.max()),
                        "dark_pixels": int(np.all(rgb < 12, axis=2).sum()),
                    }
                    upright = Image.fromarray(np.rot90(rgb, -1))
                    upright.save(args.out / f"{name}-{label}.png")
                    tile = Image.new("RGB", (180, 346), "#202830")
                    tile.paste(upright.resize((180, 320)), (0, 26))
                    ImageDraw.Draw(tile).text((4, 4), f"{name} {label}", fill="white")
                    thumbnails.append(tile)
                assert row["hide3"]["changed_pixels"] == 0, row
                assert np.array_equal(images["actual"], images["no-shadow"]), row
                rows.append(row)
        finally:
            camera.renderer.close()
        columns = len(images)
        sheet = Image.new("RGB", (180 * columns, 346 * len(poses)))
        for i, tile in enumerate(thumbnails):
            sheet.paste(tile, ((i % columns) * 180, (i // columns) * 346))
        sheet.save(args.out / "contact-sheet.png")
        report = {
            "mujoco": mujoco.__version__,
            "default_geomgroup": mujoco.MjvOption().geomgroup.tolist(),
            "poses": rows,
            "result": (
                "Actual head camera equals the shadow-disabled control at every pose. "
                "Explicitly hiding group 3 leaves the shadow-enabled control unchanged."
            ),
        }
        (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        print(f"PASS: raw MuJoCo camera visibility sweep; images in {args.out}")


if __name__ == "__main__":
    main()
