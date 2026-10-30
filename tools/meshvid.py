"""Render the mesh world (ito/render/mesh.py, lane mesh) along a tape into an MP4, offline.

    python tools/meshvid.py TAPE OUT.mp4 [--memory tsdf|ibr|atlas|none] [--dump DIR]
        [--every 2] [--start S] [--end E] [--latency 0.3] [--eye x,y,z,yaw,pitch] [--panel 640x360]

TAPE: a driver recording (tools/research/data.py). Without --dump the world is built from the
sim's exact depth and camera poses (the upper bound: what the mesh does when poses and depth are
right). With --dump DIR (ITO_DUMP of a replay: one npz per mapped frame) it is built from what
Ito estimated: that frame's depth and its placed pose.

Each video frame shows three panels: the camera frame, the mesh world (NOW + memory) and the
memory alone, both drawn from the camera pose then moved by --eye (metres right/up/back,
degrees yaw left/pitch up, in the camera's own axes), so the 3D shows. Frames reach the world
--latency seconds after capture, every --every-th frame with depth, as the live worker would.

Needs a GL 4.3 context: in WSL run with GALLIUM_DRIVER=d3d12 (the RTX through WSLg's D3D12) and
EGL, under the shared GPU lock.
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "research"), str(HERE.parent)]

import cv2  # noqa: E402
import moderngl  # noqa: E402
import numpy as np  # noqa: E402
from data import Recording  # noqa: E402

FFMPEG = os.environ.get("ITO_FFMPEG", str(Path.home() / ".unii/work/HappyParrot/ffmpeg"))
# The world's input, as the live worker sends it: depth on a coarse grid, RGB finer.
DEPTH_SIZE = (320, 180)
RGB_SIZE = (960, 540)


def offset_pose(x, y, z, yaw, pitch):
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    yaw_m = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
    pitch_m = np.array(((1, 0, 0), (0, cp, -sp), (0, sp, cp)))
    m = np.eye(4)
    m[:3, :3] = yaw_m @ pitch_m
    m[:3, 3] = (x, y, z)
    return m


def perspective(fov_y, aspect, near=0.02, far=100.0):
    f = 1 / math.tan(fov_y / 2)
    return np.array(
        (
            (f / aspect, 0, 0, 0),
            (0, f, 0, 0),
            (0, 0, (far + near) / (near - far), 2 * far * near / (near - far)),
            (0, 0, -1, 0),
        ),
        np.float32,
    )


def orthonormal(m):
    m = np.array(m, np.float64)
    u, _, vt = np.linalg.svd(m[:3, :3])
    m[:3, :3] = u @ vt
    return m


class Keyframes:
    """A frame is a keyframe once it moved 15 cm or turned 12 degrees from the last one."""

    def __init__(self):
        self.last = None
        self.count = 0

    def __call__(self, pose):
        if self.last is not None:
            moved = np.linalg.norm(pose[:3, 3] - self.last[:3, 3])
            cos = (np.trace(self.last[:3, :3].T @ pose[:3, :3]) - 1) / 2
            if moved < 0.15 and math.degrees(math.acos(np.clip(cos, -1, 1))) < 12:
                return -1
        self.last = pose
        self.count += 1
        return self.count - 1


def tape_inputs(rec, frames, every):
    """(capture time, MeshFrame factory) from the sim's exact depth and pose."""
    from ito.render.mesh import MeshFrame

    K = rec.K_at(DEPTH_SIZE)
    intrinsics = (K[0, 0], K[1, 1], K[0, 2] + 0.5, K[1, 2] + 0.5)
    with_depth = [f for f in frames if f.get("depth")][::every]
    keys = Keyframes()
    out = []
    for f in with_depth:
        pose = orthonormal(f["pose"])
        key = keys(pose)

        def make(f=f, pose=pose, key=key):
            return MeshFrame(
                rgb=rec.rgb(f, RGB_SIZE),
                depth=rec.depth(f, DEPTH_SIZE),
                intrinsics=intrinsics,
                pose=pose,
                captured=f["exposed"],
                key=key,
            )

        out.append((f["exposed"], make))
    return out


def dump_inputs(rec, folder, every):
    """From a replay's ITO_DUMP: the frame's estimated depth and the pose it was placed at."""
    from ito.render.mesh import MeshFrame

    by_seq = {f["seq"]: f for f in rec.frames}
    files = sorted(Path(folder).glob("[0-9]*.npz"))[::every]
    keys = Keyframes()
    out = []
    for path in files:
        seq = int(path.stem)
        f = by_seq.get(seq)
        if f is None:
            continue
        with np.load(path) as z:
            pose = orthonormal(z["pose"])
            depth = z["mapped"].astype(np.float32)
            fx, fy, cx, cy = (float(v) for v in np.ravel(z["intrinsics"])[:4])
        key = keys(pose)

        def make(f=f, pose=pose, depth=depth, cal=(fx, fy, cx, cy), key=key):
            return MeshFrame(
                rgb=rec.rgb(f, RGB_SIZE),
                depth=depth,
                intrinsics=cal,
                pose=pose,
                captured=f.get("exposed", f["received"]),
                key=key,
            )

        out.append((f.get("exposed", f["received"]), make, pose))
    return out


def label(image, text):
    cv2.putText(image, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(
        image, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tape")
    p.add_argument("out")
    p.add_argument("--memory", default=os.environ.get("ITO_MESH_MEMORY", "tsdf"))
    p.add_argument("--dump")
    p.add_argument("--every", type=int, default=2)
    p.add_argument("--start", type=float)
    p.add_argument("--end", type=float)
    p.add_argument("--latency", type=float, default=0.3)
    p.add_argument("--eye", default="0.15,0.10,0.30,15,-5")
    p.add_argument("--panel", default="640x360")
    p.add_argument("--fov", type=float, default=80)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--stills", help="folder: also save every 2 s of video as a jpg (for agents)")
    args = p.parse_args()

    from ito.render.mesh import MeshWorld

    rec = Recording(args.tape)
    frames = rec.between(args.start, args.end)
    if args.dump:
        inputs = dump_inputs(rec, args.dump, args.every)
        poses_t = np.array([t for t, _, _ in inputs])
        poses = np.array([pose for _, _, pose in inputs])
        inputs = [(t, make) for t, make, _ in inputs]
        t0, t1 = frames[0]["exposed"], frames[-1]["exposed"]
        inputs = [(t, m) for t, m in inputs if t0 - 1 <= t <= t1]
    else:
        inputs = tape_inputs(rec, frames, args.every)
        poses_t = poses = None
    pw, ph = (int(v) for v in args.panel.split("x"))
    eye = offset_pose(*(float(v) for v in args.eye.split(",")))
    ctx = moderngl.create_standalone_context(require=430, backend="egl")
    world = MeshWorld(ctx, memory=args.memory)
    target = ctx.simple_framebuffer((pw, ph))
    projection = perspective(2 * math.atan(math.tan(math.radians(args.fov) / 2) * ph / pw), pw / ph)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(
        [
            FFMPEG,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{3 * pw}x{ph}",
            "-r",
            str(args.fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            args.out,
        ],
        stdin=subprocess.PIPE,
    )
    step = max(1, round(30 / args.fps))
    cursor = 0
    stills = Path(args.stills) if args.stills else None
    if stills:
        stills.mkdir(parents=True, exist_ok=True)
    written = 0
    t_first = frames[0]["exposed"]
    for f in frames[::step]:
        now = f["exposed"]
        while cursor < len(inputs) and inputs[cursor][0] + args.latency <= now:
            world.add(inputs[cursor][1]())
            cursor += 1
        if poses is not None:
            i = int(np.clip(np.searchsorted(poses_t, now), 1, len(poses_t) - 1))
            a = (now - poses_t[i - 1]) / max(poses_t[i] - poses_t[i - 1], 1e-6)
            camera = poses[i - 1] if a < 0.5 else poses[i]
        else:
            camera = orthonormal(f["pose"])
        world_from_eye = (camera @ eye).astype(np.float32)
        panels = [cv2.resize(rec.rgb(f, (pw, ph)), (pw, ph))]
        for memory_only in (False, True):
            if memory_only:
                saved, world.show_now = world.show_now, False
            world.draw(world_from_eye, projection, target, clear=(0.02, 0.025, 0.04, 1))
            if memory_only:
                world.show_now = saved
            pixels = np.frombuffer(target.read(components=3, alignment=1), np.uint8)
            panels.append(np.flipud(pixels.reshape(ph, pw, 3)).copy())
        label(panels[0], f"camera  t={now - t_first:5.1f}s")
        label(panels[1], f"mesh NOW+{args.memory}")
        label(panels[2], f"{args.memory} only")
        frame = np.concatenate(panels, 1)
        encoder.stdin.write(frame.tobytes())
        if stills is not None and written % (2 * args.fps) == 0:
            cv2.imwrite(str(stills / f"{now - t_first:06.1f}.jpg"), frame[..., ::-1])
        written += 1
    encoder.stdin.close()
    encoder.wait()
    print(json.dumps(dict(out=args.out, frames=written, inputs=cursor, keys=len(world.keys))))


if __name__ == "__main__":
    main()
