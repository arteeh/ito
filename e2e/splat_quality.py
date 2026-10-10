"""Drive the bundled MuJoCo room through live RGB-D reconstruction and grade the splats.

DISPLAY=:97 LIBGL_ALWAYS_SOFTWARE=1 uv run --with pillow python e2e/splat_quality.py

A scripted robot loops the room in real time while the real reconstruction worker fuses
its camera. Afterwards the splat scene is rendered from robot poses (current view, the
start and middle of the drive, a wall) next to MuJoCo's own image from the same pose.
Coverage says whether the room is still there; grain compares high-frequency energy with
the ground truth, so visible individual splats read as grain above 1. Passes when the room
seen a lap ago is still there, the scene only grew until the budget was full, and a view
through a briefly misplaced camera leaves no ghost room once the right pose returns.
"""

import argparse
import json
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__" and sys.argv[1:] == ["--serve"]:
    # MuJoCo renders in its own process: OSMesa and the pilot's GL context do not mix.
    os.environ["MUJOCO_GL"] = os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import numpy as np
import pygame
from PIL import Image

from ito import clock
from ito.app.frames import camera_matrix
from ito.reconstruction import Reconstruction
from ito.render import GaussianRenderer, current_context, perspective, pose

OUTPUT = Path("e2e/out/splat_quality")
SIZE = (320, 240)
FOV = 65  # The bundled head camera's fovy.


class Room:
    def __init__(self):
        self.process = subprocess.Popen(
            [sys.executable, __file__, "--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        self.intrinsics = pickle.load(self.process.stdout)

    def place(self, *place):
        pickle.dump(place, self.process.stdin)
        self.process.stdin.flush()
        return pickle.load(self.process.stdout)

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=5)


class Pipe:
    def __init__(self):
        self.input, self.output = sys.stdin.buffer, sys.stdout.buffer

    def send(self, value):
        pickle.dump(value, self.output)
        self.output.flush()

    def recv(self):
        try:
            return pickle.load(self.input)
        except EOFError:
            return None


def serve():
    import mujoco

    from drivers.mujoco.adapter import MujocoAdapter

    adapter = MujocoAdapter(width=SIZE[0], height=SIZE[1], gl="osmesa")
    model, data = adapter.model, adapter.data
    renderer = mujoco.Renderer(model, SIZE[1], SIZE[0])
    free = model.jnt_qposadr[model.body_jntadr[adapter.base_id]]
    tilt_joint = model.jnt_qposadr[model.joint("head_tilt_joint").id]
    far = model.vis.map.zfar * model.stat.extent
    pipe = Pipe()
    pipe.send(adapter.description.cameras[0].intrinsics)
    while (place := pipe.recv()) is not None:
        x, y, yaw, height, tilt = (*place, 0.19, 0.0)[:5] if len(place) == 3 else place
        data.qpos[free : free + 3] = x, y, height
        data.qpos[free + 3 : free + 7] = math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)
        data.qpos[tilt_joint] = tilt
        mujoco.mj_forward(model, data)
        camera = camera_matrix(adapter.camera_pose(data))
        renderer.update_scene(data, camera=adapter.camera_id)
        renderer.disable_depth_rendering()
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        depth = renderer.render().astype(np.float32)
        depth[~(np.isfinite(depth) & (depth > 0) & (depth < far * 0.999))] = 0
        pipe.send((rgb, depth, camera))
    renderer.close()


def drive(t, period):
    """One slow loop around the furniture, the camera along the direction of travel."""
    angle = 2 * math.pi * t / period
    x, y = 0.3 + 2.4 * math.cos(angle + math.pi), 1.5 * math.sin(angle + math.pi)
    dx, dy = -2.4 * math.sin(angle + math.pi), 1.5 * math.cos(angle + math.pi)
    # Look a little outward so the walls are seen, not only the next stretch of floor.
    return x, y, math.atan2(dy, dx) - 0.5


def grade(name, splats, truth, alpha):
    def grain(image):
        gray = image.astype(float).mean(axis=2)
        laplacian = 4 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1]
        laplacian -= gray[1:-1, :-2] + gray[1:-1, 2:]
        return np.abs(laplacian)

    covered = alpha > 0.5
    inner = covered[1:-1, 1:-1]
    result = dict(
        coverage=float(covered.mean()),
        error=float(np.abs(splats.astype(float) - truth)[covered].mean()) if covered.any() else 255,
        grain=float(grain(splats)[inner].mean() / max(grain(truth)[inner].mean(), 1e-6))
        if inner.any()
        else 0.0,
    )
    Image.fromarray(np.concatenate((truth, splats), axis=1)).save(OUTPUT / f"{name}.png")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--budget", type=int, default=131_072)
    parser.add_argument("--rate", type=float, default=8, help="Camera frames per second")
    parser.add_argument("--label", default="run")
    args = parser.parse_args()
    global OUTPUT
    OUTPUT = OUTPUT / args.label
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pygame.display.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_mode((64, 64), pygame.OPENGL | pygame.DOUBLEBUF | pygame.HIDDEN)
    context = current_context()
    renderer = GaussianRenderer(context)
    target = context.simple_framebuffer(SIZE)
    projection = perspective(math.radians(FOV), SIZE[0] / SIZE[1])
    room = Room()
    counts = []
    report = dict(label=args.label, budget=args.budget, seconds=args.seconds)
    try:
        with Reconstruction(room.intrinsics, max_splats=args.budget) as reconstruction:
            began = clock.now()
            next_frame = began
            while (elapsed := clock.now() - began) < args.seconds:
                if clock.now() >= next_frame:
                    rgb, depth, camera = room.place(*drive(elapsed, args.seconds))
                    reconstruction.submit(rgb, depth, camera)
                    next_frame += 1 / args.rate
                while (packet := reconstruction.poll()) is not None:
                    renderer.apply(packet)
                    counts.append((round(elapsed, 2), packet.count))
                pygame.time.wait(5)

            def hold(seconds, camera_turn=0.0):
                # Keep looking from the last pose, optionally through a misplaced camera.
                place = drive(args.seconds, args.seconds)
                until = clock.now() + seconds
                while clock.now() < until:
                    rgb, depth, camera = room.place(*place)
                    turned = camera.copy()
                    c, s_ = math.cos(camera_turn), math.sin(camera_turn)
                    turned[:3, :3] = np.array(((c, 0, s_), (0, 1, 0), (-s_, 0, c))) @ camera[:3, :3]
                    reconstruction.submit(rgb, depth, turned)
                    while (packet := reconstruction.poll()) is not None:
                        renderer.apply(packet)
                        counts.append((round(clock.now() - began, 2), packet.count))
                    pygame.time.wait(int(1000 / args.rate))

            def view(name, place):
                truth, _, camera = room.place(*place)
                renderer.draw(camera, pose(), projection, target, clear=(0, 0, 0, 0))
                pixels = np.frombuffer(target.read(components=4), np.uint8)
                pixels = pixels.reshape(SIZE[1], SIZE[0], 4)[::-1]
                alpha = pixels[..., 3] / 255
                # Show what a pilot sees: the renderer's own clear colour behind the splats.
                background = np.array((0.025, 0.035, 0.055)) * 255
                shown = pixels[..., :3] + background * (1 - alpha[..., None])
                report[name] = grade(name, np.uint8(np.clip(shown, 0, 255)), truth, alpha)

            # Let the last frames land, the way a pilot keeps looking after the robot stops.
            hold(2)
            views = dict(
                current=drive(args.seconds, args.seconds),
                start=drive(0, args.seconds),
                halfway=drive(args.seconds / 2, args.seconds),
                wall=(0.5, 0.0, math.pi / 2, 0.6, 0.0),
            )
            for name, place in views.items():
                view(name, place)
            # A map misplaced by a few degrees for a moment (SLAM realigning) leaves a
            # second, turned room; looking on from the right place must clear it again.
            hold(1.5, math.radians(10))
            view("ghost", views["current"])
            hold(3)
            view("cleared", views["current"])
            assert context.error == "GL_NO_ERROR"
    finally:
        room.close()
    report["max_count"] = max(count for _, count in counts)
    report["final_count"] = counts[-1][1]
    report["count_by_second"] = {
        str(int(t)): c for t, c in {int(t): c for t, c in counts}.items() if int(t) % 5 == 0
    }
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUTPUT / "counts.json").write_text(json.dumps(counts) + "\n")
    print(json.dumps(report, indent=2))
    for name in ("start", "halfway", "wall"):
        assert report[name]["coverage"] > 0.75, f"The room seen at {name} faded away"
    assert report["cleared"]["error"] < report["current"]["error"] * 1.2 + 0.5, (
        "A misplaced view left a ghost room in front of the real one"
    )
    # A lap of the room shows four times what its first sixth does; a kept room grows.
    lap = [count for t, count in counts if t <= args.seconds]
    early = max(count for t, count in counts if t <= args.seconds / 6)
    full = lap[-1] >= 0.95 * args.budget
    assert full or (lap[-1] > 0.9 * max(lap) and lap[-1] > 2 * early), (
        f"The scene did not build up: {early} after a sixth of the lap, {lap[-1]} at its end"
    )
    print("PASS: the scene builds up, keeps the room it has seen and clears a ghost room")


if __name__ == "__main__":
    serve() if sys.argv[1:] == ["--serve"] else main()
