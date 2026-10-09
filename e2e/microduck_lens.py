"""Check the simulated Microduck camera sees what the real lens sees, at every resolution.

/path/to/rl/.venv/bin/python e2e/microduck_lens.py --rl /path/to/rl \\
    --driver-python /path/to/driver/venv/bin/python [--out DIR]

The renderer (mujoco) and the driver (av, cv2) live in different environments, so the driver's
intrinsics come from a subprocess.

Red markers sit at known angles off the head camera's axis. Each is rendered by the simulator's
head camera, at 360p, 720p and 1080p, and must land where the intrinsics Ito's Microduck driver
derives from mediad's description of the twin (45 degrees vertical, which is wider than the lens)
say it will; the long side must span the real lens's 62 degrees.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

# Horizontal and vertical angles off the optical axis, in degrees. The long side is the lens's 62.
MARKERS = [(0, 0), (28, 0), (-28, 0), (0, 15), (0, -15), (25, 14), (-25, -14)]
DISTANCE = 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl", type=Path, required=True)
    parser.add_argument("--driver-python", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("e2e/out/microduck-lens"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(args.rl.resolve() / "src")]

    import mujoco
    import numpy as np
    from mjlab_microduck.sim.body_server import HOME_TRUNK_Z, Body, World
    from PIL import Image

    from drivers.microduck import lens
    from drivers.microduck.sim_body import HeadCamera

    source = args.rl.resolve() / "src/mjlab_microduck/robot/microduck"
    failures = []
    for height in sorted(lens.RESOLUTIONS):
        os.environ["ITO_MICRODUCK_CAMERA_HEIGHT"] = str(height)
        width = lens.width_for(height)
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
            for index in range(len(MARKERS)):
                # Mocap bodies, because MuJoCo only places static geoms at load.
                mocap = ET.SubElement(
                    scene.find("worldbody"), "body", name=f"marker{index}", mocap="true"
                )
                ET.SubElement(
                    mocap,
                    "geom",
                    type="sphere",
                    size="0.012",
                    rgba="1 0 0 1",
                    contype="0",
                    conaffinity="0",
                )
            scene.write(state / "scene.xml")
            world = World(state / "scene.xml")
            body = Body(world, 0)
            body.place(None, HOME_TRUNK_Z, 0)
            world.bodies.append(body)
            camera = HeadCamera(world.model, "head_camera")
            try:
                mujoco.mj_forward(world.model, world.data)
                origin = world.data.cam_xpos[camera.camera].copy()
                axes = world.data.cam_xmat[camera.camera].reshape(
                    3, 3
                )  # columns: x right, y up, z back
                for index, (across, up) in enumerate(MARKERS):
                    ray = np.array(
                        [math.tan(math.radians(across)), math.tan(math.radians(up)), -1.0]
                    )
                    world.data.mocap_pos[index] = origin + axes @ (ray * DISTANCE)
                mujoco.mj_forward(world.model, world.data)
                camera.render(world)
                rgb = camera.renderer.render()
            finally:
                camera.renderer.close()
            assert rgb.shape[:2] == (height, width), rgb.shape

            # The driver's view of the twin: mediad's Intrinsics::sim, a 45 degree vertical field.
            nominal = height / 2 / math.tan(math.radians(22.5))
            info = {
                "width": width,
                "height": height,
                "rotate": 90,
                "intrinsics": {
                    "source": "sim",
                    "fx": nominal,
                    "fy": nominal,
                    "cx": width / 2,
                    "cy": height / 2,
                },
            }
            reported = subprocess.run(
                [
                    args.driver_python,
                    "-c",
                    "import json,sys;from drivers.microduck.camera import Camera;"
                    "k=Camera(json.loads(sys.argv[1])).intrinsics;print(json.dumps([k.fx,k.fy]))",
                    json.dumps(info),
                ],
                env=os.environ | {"PYTHONPATH": str(Path(__file__).resolve().parents[1])},
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            # The driver rotates intrinsics with the frame; the render is still unrotated.
            fy, fx = json.loads(reported)
            hfov = math.degrees(2 * math.atan(width / 2 / fx))
            Image.fromarray(rgb).save(args.out / f"lens-{height}p.png")
            worst = 0.0
            seen = 0
            for index, (across, up) in enumerate(MARKERS):
                red = (
                    (rgb[..., 0] > 80)
                    & (rgb[..., 0] > 3 * rgb[..., 1])
                    & (rgb[..., 0] > 3 * rgb[..., 2])
                )
                # Isolate this marker: the nearest blob to where it should be.
                u = width / 2 + fx * math.tan(math.radians(across))
                v = height / 2 - fy * math.tan(math.radians(up))
                near = np.zeros_like(red)
                r = max(8, height // 30)
                near[max(0, int(v) - r) : int(v) + r, max(0, int(u) - r) : int(u) + r] = True
                ys, xs = np.nonzero(red & near)
                if len(xs) < 0.6 * math.pi * (fx * 0.012 / DISTANCE) ** 2:
                    # Something clips the +across edge of the view; skip those markers.
                    print(f"  marker {index} ({across},{up}) occluded in the render; skipped")
                    continue
                seen += 1
                error = math.hypot(xs.mean() - u, ys.mean() - v)
                worst = max(worst, error)
            print(
                f"{height}p {width}x{height}: driver hfov {hfov:.1f} deg, "
                f"worst marker error {worst:.1f} px ({worst / height * 100:.2f}% of height)"
            )
            if seen < 5:
                failures.append(f"{height}p only {seen} markers visible")
            if abs(hfov - lens.HFOV_DEG) > 0.1:
                failures.append(f"{height}p driver hfov {hfov:.2f}")
            if worst > max(2.0, height / 180):
                failures.append(f"{height}p markers off by {worst:.1f} px")
    if failures:
        raise SystemExit("FAIL: " + "; ".join(failures))
    print("PASS: markers at 62 degrees across the frame land where the driver's intrinsics say")


if __name__ == "__main__":
    main()
