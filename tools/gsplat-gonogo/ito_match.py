# ruff: noqa: E402
"""Windows OpenGL check of unmodified Ito load_ply and GaussianRenderer."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "repo"))
import moderngl
import numpy as np
import pygame
from PIL import Image, ImageDraw

from ito.render.renderer import GaussianRenderer
from ito.render.scene import load_ply

run = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else "true"
folder = ROOT / "out" / run / mode
camera = np.load(folder / "match-reference.npz")
pose = camera["pose"].astype(np.float32)
K = camera["K"]
ref = camera["antialiased"]
classic = camera["classic"]
h, w = ref.shape[:2]
pygame.init()
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
pygame.display.set_mode((w, h), pygame.OPENGL | pygame.DOUBLEBUF | pygame.HIDDEN)
ctx = moderngl.create_context(require=430)
renderer = GaussianRenderer(ctx)
scene = load_ply(folder / "scene.ply")
renderer.upload(scene)
p = np.zeros((4, 4), np.float32)
p[0, 0] = 2 * K[0, 0] / w
p[1, 1] = 2 * K[1, 1] / h
p[0, 2] = 1 - 2 * K[0, 2] / w
p[1, 2] = 2 * K[1, 2] / h - 1
p[2, 2] = -(1000 + 0.01) / (1000 - 0.01)
p[2, 3] = -2 * 1000 * 0.01 / (1000 - 0.01)
p[3, 2] = -1
fbo = ctx.simple_framebuffer((w, h), components=4, dtype="f4")
renderer.draw(pose, np.eye(4, dtype=np.float32), p, fbo, clear=(0, 0, 0, 1))
ctx.finish()
rgb = (
    np.frombuffer(fbo.read(components=3, dtype="f4"), np.float32)
    .reshape(h, w, 3)[::-1]
    .copy()
    .clip(0, 1)
)
# Ito's own GL renderer speed: one sort + two eye draws per pair at a Quest-3-class eye size.
W, H = 1832, 1920
eye = ctx.simple_framebuffer((W, H), components=4)
proj = np.zeros((4, 4), np.float32)
proj[0, 0] = H / W
proj[1, 1] = 1.0
proj[2, 2] = p[2, 2]
proj[2, 3] = p[2, 3]
proj[3, 2] = -1
heads = []
for dx in (-0.032, 0.032):
    head = np.eye(4, dtype=np.float32)
    head[0, 3] = dx
    heads.append(head)
for n in range(70):
    if n == 10:
        ctx.finish()
        began = time.perf_counter()
    renderer.draw(pose, heads[0], proj, eye, sort=True)
    renderer.draw(pose, heads[1], proj, eye, sort=False)
ctx.finish()
ito_pairs = 60 / (time.perf_counter() - began)
report = {
    "ito_gl_stereo_pairs_per_s": ito_pairs,
    "ito_gl_eye_size": [W, H],
    "gl_renderer": ctx.info["GL_RENDERER"],
    "gl_version": ctx.info["GL_VERSION"],
    "count": scene.count,
    "sh_degree": scene.sh_degree,
    "frame": int(camera["frame"]),
    "antialiased_psnr": float(-10 * np.log10(((rgb - ref) ** 2).mean())),
    "classic_psnr": float(-10 * np.log10(((rgb - classic) ** 2).mean())),
    "mae": float(np.abs(rgb - ref).mean()),
}
image = Image.fromarray(
    (np.concatenate([ref, rgb, np.abs(ref - rgb) * 5], 1).clip(0, 1) * 255).astype(np.uint8)
)
ImageDraw.Draw(image).text(
    (8, 8),
    "gsplat AA | Ito load_ply renderer | absolute difference x5",
    fill="white",
    stroke_width=2,
    stroke_fill="black",
)
image.save(folder / "ito-match.png")
(folder / "ito-match.json").write_text(json.dumps(report, indent=2))
np.save(folder / "ito-render.npy", rgb)
print(json.dumps(report), flush=True)
pygame.quit()
