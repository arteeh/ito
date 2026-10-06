"""Launch and drive the desktop: xvfb-run -a uv run python e2e/desktop.py.

Requires xdotool. Captures and frame metrics are saved under e2e/out/desktop.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import numpy as np
import pygame
from plyfile import PlyData
from sample_scene import write_scene

OUTPUT = Path("e2e/out/desktop")


def wait_for(check, message, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(0.05)
    raise AssertionError(message)


def xdo(*args):
    return subprocess.check_output(["xdotool", *map(str, args)], text=True, timeout=10).strip()


def main():
    if not shutil.which("xdotool"):
        raise SystemExit("Install xdotool, then run under xvfb-run -a")
    xdo("--version")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    captures = OUTPUT / "captures"
    captures.mkdir(exist_ok=True)
    for old in captures.glob("capture-*.png"):
        old.unlink()
    scene = OUTPUT / "scene.ply"
    count = write_scene(scene)
    metrics = OUTPUT / "metrics.jsonl"
    env = dict(os.environ, LIBGL_ALWAYS_SOFTWARE="1", SDL_VIDEODRIVER="x11")
    with (OUTPUT / "desktop.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-m", "ito.desktop", str(scene),
                                    "--size", "960", "720", "--capture-dir", str(captures),
                                    "--metrics", str(metrics)], stdout=log, stderr=log, env=env)
        try:
            def find_window():
                if process.poll() is not None:
                    raise AssertionError((OUTPUT / "desktop.log").read_text())
                result = subprocess.run(["xdotool", "search", "--pid", str(process.pid),
                                         "--name", "Ito"], capture_output=True, text=True,
                                        check=False, timeout=5)
                return result.stdout.strip().splitlines() if result.returncode == 0 else None

            window = wait_for(find_window, "Desktop window did not open")[0]
            xdo("windowfocus", "--sync", window)
            wait_for(lambda: metrics.exists() and metrics.stat().st_size, "No rendered frames")
            time.sleep(0.5)

            def capture(number):
                xdo("key", "F12")
                path = captures / f"capture-{number:03d}.png"
                wait_for(lambda: path.exists() and path.stat().st_size > 5000, f"No capture {number}")
                wait_for(lambda: str(path) in metrics.read_text(), "Capture not flushed to metrics")
                return np.transpose(pygame.surfarray.array3d(pygame.image.load(path)), (1, 0, 2))

            before = capture(1)
            xdo("keydown", "w")
            time.sleep(0.65)
            xdo("keyup", "w")
            moved = capture(2)
            xdo("mousemove", "--window", window, "480", "400")
            xdo("click", "1")
            time.sleep(0.15)
            xdo("mousemove_relative", "--sync", "95", "-25")
            time.sleep(0.3)
            looked = capture(3)
            xdo("key", "Tab")
            xdo("key", "e")
            stopped = capture(4)
            xdo("key", "r", "space")
            xdo("windowsize", "--sync", window, "800", "600")
            time.sleep(0.4)
            resized = capture(5)
            xdo("key", "Home")
            time.sleep(0.2)
            capture(6)
            xdo("key", "Escape")
            assert process.wait(timeout=10) == 0, (OUTPUT / "desktop.log").read_text()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    # Exclude the overlay: input must actually change the rendered 3D scene.
    for name, image in (("before", before), ("moved", moved), ("looked", looked)):
        crop = image[220:]
        saturated = crop.max(axis=2).astype(int) - crop.min(axis=2).astype(int) > 60
        assert saturated.sum() > 12000, f"{name}: missing colored splat sculptures"
    assert np.abs(before[220:].astype(float) - moved[220:]).mean() > 5, "W did not change view"
    assert np.abs(moved[220:].astype(float) - looked[220:]).mean() > 5, "Mouse did not change view"
    assert np.abs(stopped[:200].astype(float) - looked[:200]).mean() > 0.3, "Command overlay did not change"
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
    result = subprocess.run([sys.executable, "-m", "ito.desktop", str(invalid), "--frames", "1"],
                            capture_output=True, text=True, env=env, timeout=10, check=False)
    assert result.returncode == 1 and "non-finite" in result.stderr and "Traceback" not in result.stderr
    print(f"PASS: {count} splats visible; WASD, mouse-look, recenter, resize, commands, invalid PLY")
    print(f"Captures: {captures}; median frame time: {np.median([row['frame_ms'] for row in rows]):.1f} ms")


if __name__ == "__main__":
    main()
