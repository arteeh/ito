"""Films the ToF mesh during a tape replay (tofmesh_replay.py) with the build's own renderer
(ito/tofmesh/render.py) in an offscreen OpenGL context: a 2x2 H.264 video,
  top left: the robot's camera frame       top right: the mesh from the robot's camera
  bottom left: the mesh from behind the robot   bottom right: a fixed view over the start
"""

import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import moderngl
import numpy as np

from ito.tofmesh.render import MeshRenderer


def perspective(fov_y, aspect, near=0.05, far=50.0):
    f = 1 / np.tan(fov_y / 2)
    p = np.zeros((4, 4))
    p[0, 0], p[1, 1] = f / aspect, f
    p[2, 2] = (far + near) / (near - far)
    p[2, 3] = 2 * far * near / (near - far)
    p[3, 2] = -1
    return p


def look_at(eye, target, up=(0, 1, 0)):
    eye, target, up = (np.asarray(v, np.float64) for v in (eye, target, up))
    back = eye - target
    back /= np.linalg.norm(back)
    right = np.cross(up, back)
    right /= np.linalg.norm(right)
    true_up = np.cross(back, right)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, true_up, back, eye
    return m


class Film:
    def __init__(self, out, size, fps, K, frame_size, overview=""):
        self.w, self.h = size[0] // 2, size[1] // 2
        self.size = (self.w * 2, self.h * 2)
        self.ctx = moderngl.create_standalone_context(require=430)
        self.renderer = MeshRenderer(self.ctx)
        self.colour = self.ctx.renderbuffer((self.w, self.h), 4)
        self.depth = self.ctx.depth_renderbuffer((self.w, self.h))
        self.fbo = self.ctx.framebuffer([self.colour], self.depth)
        self.K, self.frame_size = K, frame_size
        self.overview = [float(v) for v in overview.split(",")] if overview else None
        self.start = None
        self.image = None
        self.camera = None
        self.proc = subprocess.Popen(
            [
                imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "rawvideo",
                "-pix_fmt", "rgb24", "-s", f"{self.size[0]}x{self.size[1]}", "-r", str(fps),
                "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                str(Path(out) / "tofmesh.mp4"),
            ],
            stdin=subprocess.PIPE,
        )  # fmt: skip

    def apply(self, news):
        self.renderer.upload(news)
        for p in news.projectors:
            if p.slot == 0:
                self.image = p.image
        if news.camera is not None:
            self.camera = news.camera
            if self.start is None:
                self.start = news.camera.copy()

    def _view(self, world_from_eye, projection):
        self.renderer.draw(world_from_eye, np.eye(4), projection, self.fbo, viewport=(0, 0, self.w, self.h))
        data = self.fbo.read(components=3, viewport=(0, 0, self.w, self.h))
        return np.frombuffer(data, np.uint8).reshape(self.h, self.w, 3)[::-1]

    def draw(self, t, fusion):
        canvas = np.zeros((self.size[1], self.size[0], 3), np.uint8)
        if self.camera is not None:
            fx, fy, cx, cy = self.K
            fw, fh = self.frame_size
            fov_y = 2 * np.arctan(fh / (2 * fy))
            # Panels are 16:9 like the camera; the pilot panel uses the camera's own field of view.
            canvas[: self.h, self.w :] = self._view(self.camera, perspective(fov_y, self.w / self.h))
            pos = self.camera[:3, 3]
            forward = -self.camera[:3, 2]
            flat = np.array([forward[0], 0, forward[2]])
            flat /= max(np.linalg.norm(flat), 1e-6)
            chase = look_at(pos - flat * 1.4 + np.array([0, 0.9, 0]), pos + flat * 1.2)
            canvas[self.h :, : self.w] = self._view(chase, perspective(np.radians(70), self.w / self.h))
            if self.overview:
                ex, ey, ez, tx, ty, tz = self.overview
                over = look_at((ex, ey, ez), (tx, ty, tz))
            else:
                s = self.start[:3, 3]
                f0 = -self.start[:3, 2]
                f0 = np.array([f0[0], 0, f0[2]]) / max(np.linalg.norm([f0[0], f0[2]]), 1e-6)
                over = look_at(s - f0 * 0.8 + np.array([0, 2.6, 0]), s + f0 * 1.8)
            canvas[self.h :, self.w :] = self._view(over, perspective(np.radians(80), self.w / self.h))
        if self.image is not None:
            canvas[: self.h, : self.w] = cv2.resize(self.image, (self.w, self.h), interpolation=cv2.INTER_AREA)
        stats = fusion.stats
        label = (
            f"t {t:5.1f}s  tris {stats['triangles']}  blocks {stats['blocks']}  "
            f"fuse {stats['fuse_ms']:.0f}ms  mesh {stats['mesh_ms']:.0f}ms"
        )
        cv2.putText(canvas, label, (10, self.size[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for text, x, y in (
            ("camera", 8, 18), ("ToF mesh, robot view", self.w + 8, 18),
            ("ToF mesh, behind", 8, self.h + 18), ("ToF mesh, overview", self.w + 8, self.h + 18),
        ):  # fmt: skip
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        self.proc.stdin.write(canvas.tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()
        self.renderer.close()
