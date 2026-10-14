"""Exercise the real menus at different sizes/scales and click the splat step controls.

ITO_E2E_OUT=/path xvfb-run -a uv run python e2e/menu_layout.py
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import moderngl
import pygame
from imgui_bundle import imgui
from OpenGL import GL

from ito import diagnostics
from ito.app import connect
from ito.desktop.overlay import Overlay

OUT = Path(os.environ.get("ITO_E2E_OUT", "e2e/out/menu-layout"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CONFIG_HOME"] = os.environ["APPDATA"] = str(OUT / "config")
    (OUT / "config/ito/settings.json").unlink(missing_ok=True)
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_mode((1280, 960), pygame.OPENGL | pygame.DOUBLEBUF)
    context = moderngl.create_context()
    overlay = Overlay(context, 1_000_000)
    menu = connect.Menu(SimpleNamespace())
    menu.headset = True
    menu.recent = [dict(name="Microduck", address="172.29.91.193:8081", credential=None)]
    menu.address = menu.recent[0]["address"]
    facts = []

    def frame(size, *, pointer=None, shot=None):
        overlay.begin([], size, False, pointer=pointer)
        menu.draw(size, xr_mode=size == (768, 768))
        context.screen.use()
        context.viewport = (0, 0, *size)
        context.clear(0.05, 0.06, 0.07)
        overlay.render()
        if shot:
            screenshot(size, shot)
        pygame.display.flip()

    def screenshot(size, name):
        GL.glReadBuffer(GL.GL_BACK)
        pixels = GL.glReadPixels(0, 0, *size, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        image = pygame.image.frombytes(pixels, size, "RGB")
        pygame.image.save(pygame.transform.flip(image, False, True), OUT / f"{name}.png")

    def click(name, size=(1280, 960)):
        x, y = connect.layout[name]
        for down in (False, True, False):
            frame(size, pointer=(x, y, down))
        for _, future in menu.settings.writes:
            future.result(timeout=5)
        menu.settings.poll()

    try:
        style = imgui.get_style()
        for size, scale in (((1000, 800), 1), ((1280, 960), 1.25), ((768, 768), 1)):
            pygame.display.set_mode(size, pygame.OPENGL | pygame.DOUBLEBUF)
            style.font_scale_main = scale
            for page in ("main", "settings"):
                menu.page = page
                for _ in range(3):
                    frame(size)
                facts.append(
                    dict(size=size, scale=scale, page=page, controls=connect.layout.copy())
                )
                frame(size, shot=f"{page}-{size[0]}x{size[1]}-{scale}")
        pygame.display.set_mode((1280, 960), pygame.OPENGL | pygame.DOUBLEBUF)
        style.font_scale_main = 1
        menu.page = "settings"
        for value, action, expected in (
            (150_000, "plus", 200_000),
            (150_000, "minus", 100_000),
            (124_928, "plus", 200_000),
            (124_928, "minus", 100_000),
            (200_000, "plus", 300_000),
            (100_000, "minus", 1),
            (1, "plus", 100_000),
            ((1 << 31) - 2, "plus", (1 << 31) - 1),
        ):
            menu.settings.change(max_splats=value)
            frame((1280, 960))
            click(f"splats {action}")
            stored = json.loads((OUT / "config/ito/settings.json").read_text())
            assert stored["max_splats"] == expected, (value, action, stored)
            facts.append(dict(before=value, action=action, after=expected))
        (OUT / "layout.json").write_text(json.dumps(facts, indent=2))
        print("PASS: centred/scaled menu screenshots and real splat button steps")
    finally:
        menu.settings.close()
        diagnostics.close()
        overlay.close()
        context.release()
        pygame.quit()


if __name__ == "__main__":
    main()
