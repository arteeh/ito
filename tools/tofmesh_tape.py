"""Feed a ToF mesh worker recording (ITO_TOFMESH_TAPE, ito/tofmesh/tape.py) through Fusion again,
at the times the live worker got each input, and render stills of the mesh at chosen moments:

    uv run --extra microduck python tools/tofmesh_tape.py TAPE OUT.png T1,T2,... [--side 0.5]
        [--target 1.2]

T are seconds after the recording's first input. Each moment is one row of six pictures: the
textured mesh (the app's MeshRenderer) from the robot camera and from eyes moved `--side` metres
left and right of it (and 0.2 m up) looking at the point `--target` metres ahead of the camera,
and below them the same three as plain shaded geometry. Environment switches (ITO_TOF_GUIDE=0,
ITO_GUESS_DEPTH=0, ...) apply as in the app, so the same recording rendered twice shows a
feature off and on with identical inputs and poses. Runs on the CPU (llvmpipe) or a GPU,
whichever moderngl's standalone context gets.
"""

import argparse
import io
import sys
from pathlib import Path

import cv2
import moderngl
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tofmesh_film import look_at, perspective  # noqa: E402

from ito.tofmesh.fusion import Fusion  # noqa: E402
from ito.tofmesh.render import MeshRenderer  # noqa: E402
from ito.tofmesh.tape import load  # noqa: E402

SHADE_VERTEX = """#version 330
uniform mat4 mvp;
in vec3 position;
in vec3 normal;
out vec3 n;
void main() { n = normal; gl_Position = mvp * vec4(position, 1.0); }
"""
SHADE_FRAGMENT = """#version 330
in vec3 n;
out vec4 f;
void main() {
    vec3 m = normalize(n);
    float l = 0.35 + 0.65 * abs(dot(m, normalize(vec3(0.3, 0.8, 0.5))));
    f = vec4(mix(vec3(l), (0.5 + 0.5 * m) * l, 0.45), 1.0);
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tape")
    parser.add_argument("out")
    parser.add_argument("times", help="seconds after the first input, comma separated")
    parser.add_argument("--side", type=float, default=0.5)
    parser.add_argument("--target", type=float, default=1.2)
    args = parser.parse_args()
    items = load(args.tape)
    t0 = items[0][0]
    times = sorted(float(t) for t in args.times.split(","))
    w, h = 640, 360
    ctx = moderngl.create_standalone_context(require=430)
    renderer = MeshRenderer(ctx)
    fbo = ctx.framebuffer([ctx.renderbuffer((w, h), 4)], ctx.depth_renderbuffer((w, h)))
    shade = ctx.program(vertex_shader=SHADE_VERTEX, fragment_shader=SHADE_FRAGMENT)
    fusion = Fusion()
    mesh, fy, rows, i, now = None, None, [], 0, t0
    while times and (i < len(items) or now - t0 < times[-1]):
        now += 1 / 60
        while i < len(items) and items[i][0] <= now:
            item = items[i][1]
            i += 1
            if item[0] == "motion":
                fusion.motion(item[1])
            elif item[0] == "ranges":
                fusion.ranges(item[1])
            elif item[0] == "imu":
                fusion.imu(item[1])
            elif item[0] == "frame_jpeg":
                image = np.asarray(Image.open(io.BytesIO(item[1])).convert("RGB"))
                fusion.frame(image, *item[2:])
                fy = item[3][1] * w / image.shape[1]
        news = fusion.step(now)
        if news is not None:
            renderer.upload(news)
            if news.positions is not None:
                mesh = news
        if now - t0 < times[0] or mesh is None or fusion.camera is None:
            continue
        t = times.pop(0)
        camera = fusion.camera
        projection = perspective(2 * np.arctan(h / (2 * fy)), w / h)
        position, ahead, right = camera[:3, 3], -camera[:3, 2], camera[:3, 0]
        target = position + ahead * args.target
        views = [camera] + [
            look_at(position + right * s + np.array([0, 0.2, 0]), target)
            for s in (-args.side, args.side)
        ]
        vbo = ctx.buffer(np.hstack([mesh.positions, mesh.normals]).astype("f4").tobytes())
        ibo = ctx.buffer(mesh.triangles.astype("u4").tobytes())
        vao = ctx.vertex_array(shade, [(vbo, "3f 3f", "position", "normal")], ibo)
        textured, geometry = [], []
        for view in views:
            renderer.draw(view, np.eye(4), projection, fbo, viewport=(0, 0, w, h))
            textured.append(np.frombuffer(fbo.read(components=3), np.uint8).reshape(h, w, 3)[::-1])
            fbo.use()
            fbo.clear(0.05, 0.05, 0.08, 1.0, depth=1.0)
            ctx.enable_only(moderngl.DEPTH_TEST)
            shade["mvp"].write((projection @ np.linalg.inv(view)).T.astype("f4").tobytes())
            vao.render()
            geometry.append(np.frombuffer(fbo.read(components=3), np.uint8).reshape(h, w, 3)[::-1])
        for o in (vao, vbo, ibo):
            o.release()
        row = np.vstack([np.hstack(textured), np.hstack(geometry)])
        cv2.putText(row, f"{t:.1f} s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        rows.append(row)
    cv2.imwrite(args.out, np.vstack(rows)[..., ::-1])
    print(args.out, len(rows), "moments")


if __name__ == "__main__":
    main()
