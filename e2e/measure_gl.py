"""The pilot's Gaussian renderer in an offscreen target; readback is evaluation only."""

import os
import sys
import time

import numpy as np


def projection(i, near=0.01, far=1000):
    p = np.zeros((4, 4), np.float32)
    p[0, 0], p[1, 1] = 2 * i.fx / i.width, 2 * i.fy / i.height
    p[0, 2], p[1, 2] = 1 - 2 * i.cx / i.width, 2 * i.cy / i.height - 1
    p[2, 2], p[2, 3], p[3, 2] = -(far + near) / (far - near), -2 * far * near / (far - near), -1
    return p


class View:
    def __init__(self, intrinsics):
        import moderngl

        self.window = None
        if sys.platform == "win32":
            import pygame

            pygame.display.init()
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
            pygame.display.gl_set_attribute(
                pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
            )
            pygame.display.set_mode((64, 64), pygame.OPENGL | pygame.DOUBLEBUF | pygame.HIDDEN)
            self.context = moderngl.create_context(require=430)
            self.window = pygame
        else:
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
            self.context = moderngl.create_standalone_context(require=430, backend="egl")
        from ito.render.renderer import GaussianRenderer

        with self.context:
            self.renderer = GaussianRenderer(self.context)
            self.size = (intrinsics.width, intrinsics.height)
            self.target = self.context.simple_framebuffer(self.size, components=4, dtype="f4")
            self.device = self.context.info["GL_RENDERER"]
        self.projection = projection(intrinsics)
        self.draw_ms, self.upload_ms = [], []

    def upload(self, records):
        from ito.render.scene import GaussianBuffer

        with self.context:
            began = time.perf_counter()
            self.renderer.upload(GaussianBuffer(np.ascontiguousarray(records, np.float32)))
            self.context.finish()
            self.upload_ms.append((time.perf_counter() - began) * 1000)

    def draw(self, camera, *, sort=True, alpha=False):
        with self.context:
            began = time.perf_counter()
            self.renderer.draw(
                np.asarray(camera, np.float32),
                np.eye(4, dtype=np.float32),
                self.projection,
                self.target,
                clear=(0, 0, 0, 0),
                sort=sort,
            )
            self.context.finish()
            self.draw_ms.append((time.perf_counter() - began) * 1000)
            w, h = self.size
            pixels = np.frombuffer(self.target.read(components=4, dtype="f4"), np.float32)
        rgba = pixels.reshape(h, w, 4)[::-1].copy()
        return rgba[..., :3].clip(0, 1), rgba[..., 3] if alpha else rgba[..., 3] > 0.5

    def close(self):
        with self.context:
            self.target.release()
            self.renderer.close()
        self.context.release()
        if self.window:
            self.window.quit()
