"""Render a theme's motion as a video, headless (EGL): what Ito would show over ~16 s.

    uv run python demo.py themes/<name> [--out shots/<name>-demo] [--size 1600x900] [--fps 60]

Timeline: startup (intro), Settings and back (no intro), a session (HUD over the scene),
the "Robot stopped" prompt coming and going, then back to the main menu after the session
(intro again). Every screen change fades (theme FADE). Writes intro.mp4 (H.264), frames.png
(stills from the first two seconds) and fades.png (stills through two quick fades). The small grey caption bottom-left
is the lab's, not the theme's.
"""

import argparse
import subprocess
from pathlib import Path

from shot import ROOT, scene  # also sets up EGL and the lab path

import moderngl
import numpy as np
from imgui_bundle import imgui
from imgui_bundle.python_backends.opengl_backend_programmable import ProgrammablePipelineRenderer
from PIL import Image

import screens
from theme_api import Theme

# (start s, what happens, caption). ("show", screen, kind, intro) changes screen; ("prompt", on)
# shows or hides the "Robot stopped" prompt during the session.
TIMELINE = [
    (0.0, ("show", "menu", "menu", True), "startup: intro"),
    (4.0, ("show", "settings", "menu", False), "Settings: quick fade, no intro"),
    (6.5, ("show", "menu", "menu", False), "back to the main menu: quick fade, no intro"),
    (8.5, ("show", "hud", "session", False), "session: menu fades out, HUD fades in"),
    (9.8, ("prompt", True), "robot stopped: prompt fades in"),
    (11.3, ("prompt", False), "resumed: prompt fades out"),
    (12.3, ("show", "menu", "menu", True), "session ended: HUD fades out, menu intro"),
]
END = 17.0
STILLS = (0.1, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.2)
# The quick fades, Settings round trip and leaving for the session: fades.png
FADE_STILLS = (4.0, 4.04, 4.08, 4.12, 4.16, 4.2, 8.5, 8.55, 8.62)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("theme")
    parser.add_argument("--out")
    parser.add_argument("--size", default="1600x900")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()
    width, height = map(int, args.size.split("x"))
    out = Path(args.out or ROOT / "shots" / (Path(args.theme).resolve().name + "-demo"))
    out.mkdir(parents=True, exist_ok=True)

    ctx = moderngl.create_standalone_context(backend="egl", require=330)
    fbo = ctx.simple_framebuffer((width, height))
    backdrop = ctx.texture((width, height), 3,
                           scene(width, height).transpose(Image.FLIP_TOP_BOTTOM).tobytes())
    quad = ctx.program(
        vertex_shader="#version 330\nin vec2 p; out vec2 uv;"
        "void main(){uv=p*0.5+0.5; gl_Position=vec4(p,0,1);}",
        fragment_shader="#version 330\nuniform sampler2D t; in vec2 uv; out vec4 c;"
        "void main(){c=texture(t,uv);}",
    )
    vao = ctx.vertex_array(
        quad, ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4").tobytes()), "p")

    theme = Theme(args.theme)
    screens.THEME = theme
    imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename("")
    io.display_size = (width, height)
    theme.setup()
    if args.scale != 1.0:
        imgui.get_style().font_scale_main = args.scale
    renderer = ProgrammablePipelineRenderer()
    prompt = [False]
    draws = {
        "menu": (lambda size: screens.menu(size, screens.State), False),
        "settings": (lambda size: screens.settings(size, screens.State), False),
        "hud": (lambda size: screens.hud(size, nudged=False, prompt=prompt[0]), True),
    }

    video = out / "intro.mp4"
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", str(args.fps), "-i", "-", "-vf", "vflip",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow",
         "-movflags", "+faststart", str(video)],
        stdin=subprocess.PIPE)
    stills = {}
    segment = -1
    caption = ""
    frames = int(END * args.fps)
    for frame in range(frames):
        now = frame / args.fps
        theme.t = now
        while segment + 1 < len(TIMELINE) and now >= TIMELINE[segment + 1][0]:
            segment += 1
            _, event, caption = TIMELINE[segment]
            if event[0] == "show":
                theme.show(event[1], event[2], intro=event[3])
            else:
                prompt[0] = event[1]
        io.delta_time = 1 / args.fps
        io.add_mouse_pos_event(-imgui.FLT_MAX, -imgui.FLT_MAX)
        screens.layout.clear()
        imgui.new_frame()
        theme.frame_begin((width, height))
        over_scene = False
        for name, alpha, leaving in theme.showing():
            draw, over_scene = draws[name]
            with theme.faded(alpha, leaving):
                draw((width, height))
        theme.frame_end((width, height))
        imgui.get_foreground_draw_list().add_text(
            (14, height - 28), imgui.color_convert_float4_to_u32(imgui.ImVec4(0.6, 0.6, 0.6, 0.6)),
            f"lab  {now:5.2f} s  {caption}")
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
        ffmpeg.stdin.write(pixels)
        for s in STILLS + FADE_STILLS:
            if frame == round(s * args.fps):
                stills[s] = Image.frombytes("RGB", (width, height), pixels).transpose(
                    Image.FLIP_TOP_BOTTOM)
    ffmpeg.stdin.close()
    ffmpeg.wait()
    cols = 3
    tw, th = width // cols, height // cols
    for name, picks in (("frames.png", STILLS), ("fades.png", FADE_STILLS)):
        sheet = Image.new("RGB", (tw * cols, th * ((len(picks) + cols - 1) // cols)))
        for index, s in enumerate(picks):
            sheet.paste(stills[s].resize((tw, th), Image.LANCZOS),
                        ((index % cols) * tw, (index // cols) * th))
        sheet.save(out / name)
    renderer.shutdown()
    imgui.destroy_context()
    print(f"{theme.name}: wrote {video} ({frames} frames, {args.fps} fps) frames.png and fades.png, rc={ffmpeg.returncode}")


if __name__ == "__main__":
    main()
