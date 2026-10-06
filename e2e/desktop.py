"""Launch and drive the desktop: xvfb-run -a uv run python e2e/desktop.py.

Uses SDL event injection. Captures and frame metrics are saved under e2e/out/desktop.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import numpy as np
import pygame
from plyfile import PlyData
from sample_scene import write_scene

OUTPUT = Path("e2e/out/desktop")


def main():
    from pygame._sdl2 import Window

    from ito.desktop import DesktopWindow
    from ito.desktop.__main__ import FileScene

    OUTPUT.mkdir(parents=True, exist_ok=True)
    captures = OUTPUT / "captures"
    captures.mkdir(exist_ok=True)
    for old in captures.glob("capture-*.png"):
        old.unlink()
    scene = OUTPUT / "scene.ply"
    count = write_scene(scene)
    metrics = OUTPUT / "metrics.jsonl"
    env = dict(os.environ, LIBGL_ALWAYS_SOFTWARE="1", SDL_VIDEODRIVER="x11")
    ticks = 0

    def key(keycode, down=True):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN if down else pygame.KEYUP, key=keycode))

    def capture():
        key(pygame.K_F12)
        key(pygame.K_F12, False)

    def drive(pilot):
        nonlocal ticks
        ticks += 1
        if ticks in (15, 60, 85, 95, 115, 130):
            capture()
        if ticks == 20:
            key(pygame.K_w)
        elif ticks == 58:
            key(pygame.K_w, False)
        elif ticks == 65:
            key(pygame.K_TAB)
        elif ticks == 66:
            key(pygame.K_TAB, False)
        elif ticks == 70:
            pygame.event.post(
                pygame.event.Event(
                    pygame.MOUSEMOTION, pos=(480, 400), rel=(95, -25), buttons=(0, 0, 0)
                )
            )
        elif ticks == 88:
            key(pygame.K_TAB)
            key(pygame.K_TAB, False)
            key(pygame.K_e)
        elif ticks == 98:
            key(pygame.K_e, False)
            key(pygame.K_r)
            key(pygame.K_SPACE)
        elif ticks == 105:
            Window.from_display_module().size = (800, 600)
        elif ticks == 120:
            key(pygame.K_HOME)
        elif ticks == 135:
            key(pygame.K_ESCAPE)

    with (
        metrics.open("w") as log,
        DesktopWindow((960, 720), fps=60, capture_dir=captures) as window,
    ):
        window.run(FileScene(scene), on_input=drive, metrics=log, max_frames=150)

    def read(number):
        path = captures / f"capture-{number:03d}.png"
        assert path.exists(), f"No capture {number}"
        return np.transpose(pygame.surfarray.array3d(pygame.image.load(path)), (1, 0, 2))

    before, moved, looked, stopped, resized = map(read, range(1, 6))
    # Exclude the overlay: input must actually change the rendered 3D scene.
    for name, image in (("before", before), ("moved", moved), ("looked", looked)):
        crop = image[220:]
        saturated = crop.max(axis=2).astype(int) - crop.min(axis=2).astype(int) > 60
        assert saturated.sum() > 12000, f"{name}: missing colored splat sculptures"
    assert np.abs(before[220:].astype(float) - moved[220:]).mean() > 5, "W did not change view"
    assert np.abs(moved[220:].astype(float) - looked[220:]).mean() > 5, "Mouse did not change view"
    assert np.abs(stopped[:200].astype(float) - looked[:200]).mean() > 0.3, (
        "Command overlay did not change"
    )
    assert resized.shape == (600, 800, 3), resized.shape
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    captured = [row for row in rows if row["capture"]]
    assert all(row["gaussians"] == count for row in rows)
    assert all("llvmpipe" in row["renderer"].lower() for row in rows)
    assert captured[1]["head"][2][3] < -0.3
    assert abs(captured[2]["head"][0][2]) > 0.1
    assert np.allclose(captured[-1]["head"], np.eye(4))
    commands = [command for row in rows for command in row["commands"]]
    assert all(command in commands for command in ("e_stop", "resume", "stop")), commands

    # Bad files must produce an actionable error, never a traceback or blank window.
    vertices = PlyData.read(scene)
    vertices["vertex"]["scale_0"][0] = np.nan
    invalid = OUTPUT / "invalid.ply"
    vertices.write(invalid)
    result = subprocess.run(
        [sys.executable, "-m", "ito.desktop", str(invalid), "--frames", "1"],
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
        check=False,
    )
    assert (
        result.returncode == 1
        and "non-finite" in result.stderr
        and "Traceback" not in result.stderr
    )
    print(
        f"PASS: {count} splats visible; WASD, mouse-look, recenter, resize, commands, invalid PLY"
    )
    median_ms = np.median([row["frame_ms"] for row in rows])
    print(f"Captures: {captures}; median frame time: {median_ms:.1f} ms")


if __name__ == "__main__":
    main()
