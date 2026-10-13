"""Fly a saved view-cache world with Ito's desktop input and Dear ImGui.

python -m ito.proto.viewcache PATH-world.npz
"""

import argparse
import json
import logging
import math
import time
from pathlib import Path

import moderngl
import numpy as np
import pygame
from imgui_bundle import imgui
from OpenGL import GL

from ito.desktop.input import DesktopInput
from ito.desktop.overlay import Overlay
from ito.render import perspective

from .renderer import ViewCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world", type=Path)
    parser.add_argument("--size", type=int, nargs=2, default=(1280, 720))
    parser.add_argument("--fov", type=float, default=70)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--capture-dir", type=Path, default=Path("captures/viewcache"))
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()
    if min(args.size) < 64 or not 1 <= args.fov <= 150 or args.frames < 0:
        parser.error("Invalid size, field of view or frame count")
    pygame.display.init()
    world = overlay = inputs = ctx = None
    metrics = None
    try:
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
        )
        pygame.display.set_mode(args.size, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
        pygame.display.set_caption("Ito — View cache inspection")
        ctx = moderngl.create_context(require=430)
        world = ViewCache.load(ctx, args.world)
        anchor_file = args.world.with_name(args.world.name.replace("-world.npz", "-camera.npy"))
        anchor = (
            np.load(anchor_file)
            if anchor_file != args.world and anchor_file.exists()
            else world.sources[-1].pose.copy()
        )
        overlay, inputs = Overlay(ctx, 1), DesktopInput(speed=0.5)
        pacing, stereo, number = pygame.time.Clock(), False, 0
        if args.metrics:
            args.metrics.parent.mkdir(parents=True, exist_ok=True)
            metrics = args.metrics.open("w")
        while not args.frames or number < args.frames:
            dt = min(pacing.tick(90) / 1000, 0.1)
            events = pygame.event.get()
            size = pygame.display.get_window_size()
            if min(size) == 0:
                continue
            io = overlay.begin(events, size, inputs.captured)
            pilot = inputs.poll(
                dt, events, mouse_ui=io.want_capture_mouse, keyboard_ui=io.want_capture_keyboard
            )
            if pilot.quit:
                break
            camera = anchor @ pilot.head
            start = time.perf_counter()
            projection = perspective(
                math.radians(args.fov), (size[0] / 2 if stereo else size[0]) / size[1]
            )
            world.prepare(camera, projection)
            if stereo:
                width, height = size[0] // 2, size[1]
                # Independent eye depth buffers; an ImGui divider does not alter eye separation.
                ctx.screen.clear(0, 0, 0, 1, depth=1)
                for side in (-1, 1):
                    target = ctx.simple_framebuffer((width, height))
                    eye = camera.copy()
                    eye[:3, 3] += eye[:3, 0] * side * 0.032
                    world.draw(eye, projection, target)
                    # Blit into the corresponding half with OpenGL's framebuffer API.
                    GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, target.glo)
                    GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, 0)
                    x = 0 if side == -1 else width
                    GL.glBlitFramebuffer(
                        0,
                        0,
                        width,
                        height,
                        x,
                        0,
                        x + width,
                        height,
                        GL.GL_COLOR_BUFFER_BIT,
                        GL.GL_NEAREST,
                    )
                    target.release()
                ctx.screen.use()
            else:
                world.draw(camera, projection, ctx.screen)
            imgui.set_next_window_pos((12, 12), imgui.Cond_.always)
            imgui.begin("View cache", flags=imgui.WindowFlags_.always_auto_resize)
            imgui.text(f"{len(world.sources)} sources | {pacing.get_fps():.0f} fps | saved room")
            imgui.text("WASD move | PgUp/PgDn rise/fall | Home recenter")
            imgui.text("Tab mouse-look | F12 screenshot | close window to leave")
            _, stereo = imgui.checkbox("Stereo: 64 mm eye separation", stereo)
            _, args.fov = imgui.slider_float("Vertical FOV", args.fov, 30, 110)
            if imgui.button("Recenter"):
                inputs.position[:] = 0
                inputs.yaw = inputs.pitch = 0
            imgui.end()
            imgui.render()
            overlay.backend.render(imgui.get_draw_data())
            if pilot.screenshot:
                args.capture_dir.mkdir(parents=True, exist_ok=True)
                GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, 0)
                GL.glReadBuffer(GL.GL_BACK)
                GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
                pixels = GL.glReadPixels(0, 0, *size, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
                image = pygame.image.frombytes(pixels, size, "RGB")
                pygame.image.save(
                    pygame.transform.flip(image, False, True),
                    args.capture_dir / f"frame-{number:06d}.png",
                )
            pygame.display.flip()
            if metrics:
                metrics.write(
                    json.dumps(
                        dict(
                            frame=number,
                            window=pygame.display.get_wm_info().get("window"),
                            camera=camera.tolist(),
                            stereo=stereo,
                            cpu_draw_submit_ms=(time.perf_counter() - start) * 1000,
                        )
                    )
                    + "\n"
                )
                metrics.flush()
            number += 1
        return 0
    except (OSError, ValueError, RuntimeError, pygame.error, moderngl.Error) as exc:
        logging.error("%s", exc)
        return 1
    finally:
        if metrics:
            metrics.close()
        if inputs:
            inputs.close()
        if overlay:
            overlay.close()
        if world:
            world.close()
        if ctx:
            ctx.release()
        pygame.display.quit()


if __name__ == "__main__":
    raise SystemExit(main())
