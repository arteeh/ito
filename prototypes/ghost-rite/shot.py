"""Render Ito's screens with a theme, headless (EGL), to PNGs.

    uv run python shot.py themes/<name> [--out shots/<name>] [--size 1600x900] [--scale 1.0]

Writes 1-menu.png, 2-settings.png, 3-hud.png, 4-hud-alert.png and sheet.png (all four, 2x2).
Menus draw on the theme's clear colour (Ito shows no scene behind them); the HUD shots draw
over a fake splat scene. --scale multiplies the font size (125% Windows scaling: --scale 1.25).
--time T sets the animation clock passed to the hooks (default 0.37 s).
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import moderngl  # noqa: E402
import numpy as np  # noqa: E402
from imgui_bundle import imgui  # noqa: E402
from imgui_bundle.python_backends.opengl_backend_programmable import (  # noqa: E402
    ProgrammablePipelineRenderer,
)
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "lab"))
import screens  # noqa: E402
from theme_api import Theme  # noqa: E402


def scene(width, height, seed=7):
    """A dim room of soft blobs, standing in for Ito's splat view."""
    rng = np.random.default_rng(seed)
    w, h = 320, int(320 * height / width)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    horizon = h * 0.55
    img = np.zeros((h, w, 3), np.float32)
    wall = np.clip(1 - y / horizon, 0, 1)[..., None]
    img += np.array([0.20, 0.19, 0.18]) * (1 - wall) * (y > horizon)[..., None]
    img += np.array([0.32, 0.30, 0.27]) * (y <= horizon)[..., None] * (0.6 + 0.4 * wall)
    for _ in range(260):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(2, 18)
        color = rng.uniform(0.1, 0.7, 3) * np.array([1.0, 0.9, 0.8])
        a = rng.uniform(0.15, 0.6)
        g = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * r * r))[..., None] * a
        img = img * (1 - g) + color * g
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(img).resize((width, height), Image.BICUBIC)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("theme")
    parser.add_argument("--out")
    parser.add_argument("--size", default="1600x900")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--time", type=float, default=0.37)
    args = parser.parse_args()
    width, height = map(int, args.size.split("x"))
    out = Path(args.out or ROOT / "shots" / Path(args.theme).resolve().name)
    out.mkdir(parents=True, exist_ok=True)

    ctx = moderngl.create_standalone_context(backend="egl", require=330)
    fbo = ctx.simple_framebuffer((width, height))
    backdrop = ctx.texture((width, height), 3, scene(width, height).transpose(Image.FLIP_TOP_BOTTOM).tobytes())
    quad = ctx.program(
        vertex_shader="#version 330\nin vec2 p; out vec2 uv;"
        "void main(){uv=p*0.5+0.5; gl_Position=vec4(p,0,1);}",
        fragment_shader="#version 330\nuniform sampler2D t; in vec2 uv; out vec4 c;"
        "void main(){c=texture(t,uv);}",
    )
    vao = ctx.vertex_array(
        quad, ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4").tobytes()), "p"
    )

    theme = Theme(args.theme)
    theme.fade = 0.0  # stills show every panel settled, never mid-fade
    screens.THEME = theme
    imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename("")
    io.display_size = (width, height)
    theme.setup()
    if args.scale != 1.0:
        imgui.get_style().font_scale_main = args.scale
    renderer = ProgrammablePipelineRenderer()

    shots = []
    for name, (draw, over_scene, hover) in screens.SHOTS.items():
        for frame in range(6):  # auto-resized windows settle after two frames
            theme.t = args.time + frame / 60 - 5 / 60
            io.delta_time = 1 / 60
            target = screens.layout.get(hover) if hover and frame >= 2 else None
            io.add_mouse_pos_event(*(target or (-imgui.FLT_MAX, -imgui.FLT_MAX)))
            screens.layout.clear()
            imgui.new_frame()
            theme.frame_begin((width, height))
            draw((width, height), screens.State)
            theme.frame_end((width, height))
            imgui.render()
            fbo.use()
            ctx.viewport = (0, 0, width, height)
            if over_scene:
                backdrop.use(0)
                vao.render(moderngl.TRIANGLE_STRIP)
            else:
                ctx.clear(*theme.color("clear")[:3])
            renderer.render(imgui.get_draw_data())
        pixels = fbo.read(components=3)
        image = Image.frombytes("RGB", (width, height), pixels).transpose(Image.FLIP_TOP_BOTTOM)
        image.save(out / f"{name}.png")
        shots.append(image)
    sheet = Image.new("RGB", (width, height))
    for index, image in enumerate(shots):
        sheet.paste(image.resize((width // 2, height // 2), Image.LANCZOS),
                    ((index % 2) * width // 2, (index // 2) * height // 2))
    sheet.save(out / "sheet.png")
    renderer.shutdown()
    imgui.destroy_context()
    print(f"{theme.name}: wrote {len(shots)} shots and sheet.png to {out}")


if __name__ == "__main__":
    main()
