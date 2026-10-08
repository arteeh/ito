"""The flat camera feed stays straight ahead of the pilot, wherever they look.

LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a uv run python e2e/flat_panel.py

The real desktop window shows a camera frame in flat-feed mode while the pilot's head turns
to 0, 90, 180 and -90 degrees and pitches up and down. Every drawn frame must show the panel
centred, the same size, and as wide as the camera sees; a camera wider than the view must
shrink to fit inside it.
"""

import json
import math
from pathlib import Path

import numpy as np
import pygame
from OpenGL import GL

from ito import clock
from ito.desktop import DesktopState, DesktopWindow
from ito.render import pose

OUT = Path("e2e/out/flat-panel")
SIZE = (640, 480)
FOV_Y = 70
HEADS = [(yaw, pitch) for yaw in (0, 90, 180, -90) for pitch in (0, 35, -35)]


class Empty:
    def poll(self):
        return None


def frame():
    # A plain bright frame with a dark marker in its top-left quarter: the panel must
    # not be mirrored or rotated either.
    rgb = np.full((240, 320, 3), (240, 200, 40), dtype=np.uint8)
    rgb[20:100, 20:120] = (20, 60, 220)
    return rgb


def panel(pixels):
    yellow = (pixels[..., 0] > 200) & (pixels[..., 1] > 160) & (pixels[..., 2] < 90)
    blue = (pixels[..., 2] > 180) & (pixels[..., 0] < 60)
    rows, columns = np.nonzero(yellow | blue)
    if not len(rows):
        return None
    marker_rows, marker_columns = np.nonzero(blue)
    return {
        "left": int(columns.min()),
        "right": int(columns.max()) + 1,
        "top": int(rows.min()),
        "bottom": int(rows.max()) + 1,
        "marker_top_left": bool(
            len(marker_rows)
            and marker_rows.mean() < rows.mean()
            and marker_columns.mean() < columns.mean()
        ),
    }


def measure(fov_degrees):
    rgb = frame()
    began = clock.now()
    shots = {}
    with DesktopWindow(SIZE, fps=60, fov=FOV_Y, capture_dir=OUT) as window:
        draw = window.draw_view

        def state():
            return DesktopState(
                video=rgb,
                video_time=clock.now(),
                video_fov=math.radians(fov_degrees),
                flat_video=True,
            )

        def turned(current, head, projection, target, viewport, *args, **kwargs):
            index = int((clock.now() - began) * 4)
            if index >= len(HEADS):
                window.overlay.leave = True
                index = len(HEADS) - 1
            yaw, pitch = HEADS[index]
            head = pose((0.0, 0.0, 0.0), math.radians(yaw), math.radians(pitch))
            draw(current, head, projection, target, viewport, *args, **kwargs)
            GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
            data = GL.glReadPixels(0, 0, *viewport[2:], GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
            pixels = np.frombuffer(data, np.uint8).reshape(viewport[3], viewport[2], 3)[::-1]
            if (yaw, pitch) not in shots or not shots[(yaw, pitch)]:
                shots[(yaw, pitch)] = panel(pixels)
                if pitch == 0:
                    image = pygame.image.frombytes(pixels.tobytes(), viewport[2:], "RGB")
                    pygame.image.save(image, OUT / f"fov{fov_degrees}-yaw{yaw}.png")

        window.draw_view = turned
        window.run(Empty(), state=state, save_settings=lambda _: None)
    return shots


def check(shots, fov_degrees):
    width, height = SIZE
    half_x = math.tan(math.radians(FOV_Y) / 2) * width / height
    expected = min(math.tan(math.radians(fov_degrees) / 2) / half_x, 0.9) * width
    assert set(shots) == set(HEADS), f"missing head poses: {set(HEADS) - set(shots)}"
    for head, box in shots.items():
        assert box, f"no panel visible at yaw/pitch {head}"
        centre = ((box["left"] + box["right"]) / 2, (box["top"] + box["bottom"]) / 2)
        assert abs(centre[0] - width / 2) <= 2 and abs(centre[1] - height / 2) <= 2, (head, box)
        assert abs((box["right"] - box["left"]) - expected) <= 3, (head, box, expected)
        assert box["bottom"] - box["top"] <= 0.9 * height + 1, (head, box)
        assert box["marker_top_left"], (head, box)
    return {
        "panel_width_px": shots[(0, 0)]["right"] - shots[(0, 0)]["left"],
        "expected_width_px": round(expected, 1),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    for fov in (60, 140):
        result[f"camera_fov_{fov}"] = check(measure(fov), fov)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    print("PASS: the flat feed stays centred and the same size at every head pose")


if __name__ == "__main__":
    main()
