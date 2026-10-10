"""Every camera frame reaches the 3D view, hung where the camera looked (#29).

LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a uv run python e2e/live_frame.py

The real desktop window draws the scene view (not the flat feed) while a new camera frame
arrives every other drawn frame. The pilot's head and the camera each look at several yaws:
the frame must sit where the camera looked relative to the pilot's view (world-locked, so a
head that trails the pilot shows as the frame trailing the view), upright and unmirrored,
as wide as the camera sees. Every new frame must be counted as shown, and a feed that stops
must fade out of the view instead of freezing in it.
"""

import json
import math
from pathlib import Path

import numpy as np
import pygame
from OpenGL import GL

from ito import clock
from ito.desktop import DesktopState, DesktopWindow
from ito.protocol import Intrinsics
from ito.render import pose

OUT = Path("e2e/out/live-frame")
SIZE = (640, 480)
FOV_Y = 70
CAMERA = Intrinsics(width=320, height=240, fx=400, fy=400, cx=160, cy=120)
# (pilot head yaw, camera yaw), degrees, left positive.
VIEWS = [(0, 0), (15, 0), (-15, 0), (30, 30), (0, -12), (55, 45)]
FRAMES_PER_VIEW = 6
STALE = 1.5  # Seconds: well past the fade, the frame must be gone.


class Empty:
    def poll(self):
        return None


def frame():
    rgb = np.full((CAMERA.height, CAMERA.width, 3), (240, 200, 40), dtype=np.uint8)
    rgb[10:70, 10:90] = (20, 60, 220)  # Top-left marker: not mirrored, not upside down.
    return rgb


def found(pixels):
    yellow = (pixels[..., 0] > 120) & (pixels[..., 1] > 100) & (pixels[..., 2] < 90)
    blue = (pixels[..., 2] > 110) & (pixels[..., 0] < 60)
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rgb = frame()
    drawn = 0
    stamps = 0
    shots, stale_shot = {}, "not drawn"
    stamp = clock.now()
    with DesktopWindow(SIZE, fps=60, fov=FOV_Y, capture_dir=OUT) as window:
        draw = window.draw_view

        def view():
            return VIEWS[min(drawn // FRAMES_PER_VIEW, len(VIEWS) - 1)]

        def state():
            nonlocal stamp, stamps
            if drawn >= len(VIEWS) * FRAMES_PER_VIEW:
                return DesktopState(
                    video=rgb,
                    video_time=clock.now() - STALE,
                    video_orientation=pose(),
                    video_intrinsics=CAMERA,
                )
            if drawn % 2 == 0:
                stamp, stamps = clock.now(), stamps + 1
            return DesktopState(
                video=rgb,
                video_time=stamp,
                video_orientation=pose(yaw=math.radians(view()[1])),
                video_intrinsics=CAMERA,
            )

        def turned(current, head, projection, target, viewport, *args, **kwargs):
            nonlocal drawn, stale_shot
            stale = drawn >= len(VIEWS) * FRAMES_PER_VIEW
            key = view()
            drawn += 1
            if drawn > len(VIEWS) * FRAMES_PER_VIEW + 2:
                window.overlay.leave = True
            head = pose((0.0, 0.0, 0.0), math.radians(key[0]))
            draw(current, head, projection, target, viewport, *args, **kwargs)
            GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
            data = GL.glReadPixels(0, 0, *viewport[2:], GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
            pixels = np.frombuffer(data, np.uint8).reshape(viewport[3], viewport[2], 3)[::-1]
            if stale:
                stale_shot = found(pixels)
            elif not shots.get(key):
                shots[key] = found(pixels)
                image = pygame.image.frombytes(pixels.tobytes(), viewport[2:], "RGB")
                pygame.image.save(image, OUT / f"head{key[0]}-camera{key[1]}.png")

        window.draw_view = turned
        window.run(Empty(), state=state, save_settings=lambda _: None)
        shown = window.shown

    width, height = SIZE
    focal = height / 2 / math.tan(math.radians(FOV_Y) / 2)
    result = {}
    for (head_yaw, camera_yaw), box in shots.items():
        assert box, f"no live frame visible for head/camera yaw {head_yaw}/{camera_yaw}"
        # The camera's edge rays, seen from the pilot's turned head.
        away, half = math.radians(head_yaw - camera_yaw), math.atan(CAMERA.cx / CAMERA.fx)
        left, right = (width / 2 + focal * math.tan(away + side * half) for side in (-1, 1))
        expected, span = (left + right) / 2, right - left
        centre = (box["left"] + box["right"]) / 2
        result[f"head{head_yaw}_camera{camera_yaw}"] = {
            "centre_px": centre,
            "expected_px": round(expected, 1),
            "width_px": box["right"] - box["left"],
            "expected_width_px": round(span, 1),
        }
        assert abs(centre - expected) <= 4, (head_yaw, camera_yaw, box, expected)
        # The feathered edge fades the outermost few percent into the scene.
        assert abs((box["right"] - box["left"]) - span) <= 0.12 * span, (box, span)
        assert box["marker_top_left"], (head_yaw, camera_yaw, box)
        assert abs((box["top"] + box["bottom"]) / 2 - height / 2) <= 4, box
    # Every new frame drawn is counted once, however many times it is redrawn.
    result["frames_sent"] = stamps
    result["frames_counted"] = len(shown.times)
    assert len(shown.times) == stamps, (len(shown.times), stamps)
    assert stale_shot is None, f"a stalled feed stayed in view: {stale_shot}"
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    print(
        "PASS: every camera frame shows in the 3D view where the camera looked, upright, "
        "as wide as the camera sees, and a stalled feed fades out"
    )


if __name__ == "__main__":
    main()
