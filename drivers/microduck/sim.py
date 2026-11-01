"""The virtual Microduck's body: the beta in MuJoCo (microduck_rl's body server, in our furnished
room, with the beta's head camera lens), on this machine so its viewer can open a window. The
robot side runs in compose.yaml and finds it on ports 7801 (bus) and 7901 (head camera).

    docker compose up -d
    uv run --extra sim python -m drivers.microduck.sim [--viewer]
"""

import argparse
import os
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from drivers.microduck.robot import BODY_PORT, FRAME_PORT
from drivers.microduck.scene import furnish

ROOT = Path(__file__).resolve().parents[2]
RL = ROOT / "third_party/microduck_rl/src"
SCENES = RL / "mjlab_microduck/robot/microduck"


def scene():
    """Upstream's scene and robot, furnished, with the robot's visual and contact geoms in
    groups 2 and 3 (the head camera draws the visual ones).

    The eye ring around the lens goes in group 4, which only the viewer draws: the head camera
    sits at the sensor, 8.5 mm behind the lens's front, and from there saw the ring as a solid
    border round the frame's sides and corners (a fifth of the frame), which Ito then painted
    onto the room. A real lens looks out from its front, past the ring."""
    state = Path(tempfile.gettempdir()) / "ito-microduck-scene"
    shutil.rmtree(state, ignore_errors=True)
    state.mkdir()
    for asset in SCENES.iterdir():
        if asset.name not in {"scene.xml", "robot_groundcontact.xml"}:
            (state / asset.name).symlink_to(asset, target_is_directory=asset.is_dir())
    robot = ET.parse(SCENES / "robot_groundcontact.xml")
    for geom in robot.findall(".//worldbody//geom"):
        geom.set("group", "2" if geom.get("class") == "visual" else "3")
        if geom.get("mesh") == "noenoeil" and geom.get("class") == "visual":
            geom.set("group", "4")
    robot.write(state / "robot_groundcontact.xml")
    furnish(SCENES / "scene.xml").write(state / "scene.xml")
    return state / "scene.xml"


def environment(viewer):
    # With a display (WSLg), GLFW renders the head camera on the GPU; headless EGL is software.
    display = viewer or bool(os.environ.get("DISPLAY")) or Path("/tmp/.X11-unix/X0").exists()
    env = os.environ | {
        "PYTHONPATH": os.pathsep.join([str(ROOT), str(RL)]),
        "MUJOCO_GL": "glfw" if display else "egl",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    if display:
        env.setdefault("DISPLAY", ":0")
    if display and Path("/dev/dxg").exists():
        # WSL's Mesa defaults to llvmpipe; d3d12 renders the viewer and head camera on the GPU.
        env.setdefault("GALLIUM_DRIVER", "d3d12")
        import glfw

        # pyGLFW prefers the system's GLFW, and Ubuntu 26.04's (3.4) opens the viewer on WSLg's
        # Wayland: no title bar or window decorations. The wheel's X11 build keeps it on XWayland.
        env.setdefault("PYGLFW_LIBRARY", str(Path(glfw.__file__).parent / "x11/libglfw.so"))
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--viewer", action="store_true", help="open MuJoCo's viewer")
    args = parser.parse_args()
    if not SCENES.is_dir():
        sys.exit("third_party/microduck_rl is missing: git submodule update --init --depth 1")
    command = [sys.executable, "-m", "drivers.microduck.sim_body"]
    command += [] if args.viewer else ["--headless"]
    command += ["--cameras", "a", "--port", str(BODY_PORT), "--frame-port", str(FRAME_PORT)]
    command += ["--scene", str(scene()), "--board", "beta", "--imu-noise", "1"]
    os.execve(sys.executable, command, environment(args.viewer))


if __name__ == "__main__":
    main()
