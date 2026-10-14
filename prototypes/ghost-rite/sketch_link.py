"""Sketch: a theme's link_thread() in its three states, headless, to one PNG.

    uv run python sketch_link.py themes/ghost-rite [--out shots/ghost-rite/5-link.png]
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import moderngl  # noqa: E402
from imgui_bundle import imgui  # noqa: E402
from imgui_bundle.python_backends.opengl_backend_programmable import (  # noqa: E402
    ProgrammablePipelineRenderer,
)
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "lab"))
from theme_api import Theme  # noqa: E402

STATES = (
    ("taut", "CONNECTED | RTT 23 ms"),
    ("slack", "LAGGING | RTT 240 ms"),
    ("cut", "糸が切れた | LINK LOST"),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("theme")
    parser.add_argument("--out")
    args = parser.parse_args()
    width, height = 1200, 440
    theme = Theme(args.theme)
    out = Path(args.out or ROOT / "shots" / theme.dir.name / "5-link.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    ctx = moderngl.create_standalone_context(backend="egl", require=330)
    fbo = ctx.simple_framebuffer((width, height))
    imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename("")
    io.display_size = (width, height)
    theme.setup()
    renderer = ProgrammablePipelineRenderer()
    for _ in range(4):
        io.delta_time = 1 / 60
        imgui.new_frame()
        imgui.set_next_window_pos((40, 30), imgui.Cond_.always)
        imgui.set_next_window_size((width - 80, height - 60), imgui.Cond_.always)
        with theme.font("title"):
            imgui.begin("Link", flags=imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_collapse)
        dl = imgui.get_window_draw_list()
        for state, label in STATES:
            imgui.dummy((1, 18))
            with theme.font("numeric"):
                imgui.text(label)
            p = imgui.get_cursor_screen_pos()
            imgui.dummy((1, 64))
            theme.mod.link_thread(dl, (p.x + 10, p.y + 20), (p.x + width - 160, p.y + 20), state, 1.0)
        imgui.end()
        imgui.render()
        fbo.use()
        ctx.clear(*theme.color("clear")[:3])
        renderer.render(imgui.get_draw_data())
    image = Image.frombytes("RGB", (width, height), fbo.read(components=3)).transpose(Image.FLIP_TOP_BOTTOM)
    image.save(out)
    renderer.shutdown()
    imgui.destroy_context()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
