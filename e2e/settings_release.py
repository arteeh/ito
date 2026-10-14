"""Click Settings in the real ito.exe release, restart it, and connect to its simulation.

Windows: python e2e/settings_release.py --release C:/path/ito --out C:/path/output
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from e2e import release


def button_rows(image):
    controls = release.control_pixels(image)
    rows = np.flatnonzero(controls.sum(axis=1) > 50)
    bands = [b for b in np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1) if len(b)]
    tallest = max(map(len, bands))
    result = []
    for band in bands:
        if len(band) < tallest * 0.7:
            continue
        columns = np.flatnonzero(controls[band].sum(axis=0) > len(band) * 0.5)
        groups = np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1)
        groups = [group for group in groups if len(group) > 40]
        centres = []
        for index, group in enumerate(groups):
            if index and group[0] - groups[index - 1][-1] > 40:
                # The numeric input is dark glass between the two cyan step buttons.
                centres.append(float((group[0] + groups[index - 1][-1]) / 2))
            centres.append(float((group[0] + group[-1]) / 2))
        result.append([(x, float((band[0] + band[-1]) / 2)) for x in centres])

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    release.OUT = args.out.resolve()
    release.OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ | {"APPDATA": str(release.OUT / "config")}
    path = release.OUT / "config/ito/settings.json"
    path.unlink(missing_ok=True)
    app = None
    try:
        app = release.App(args.release.resolve(), env, "settings", "--size", "1000", "800")
        time.sleep(2)
        image = app.screenshot("main-menu")
        buttons = release.menu_buttons(image)
        app.click(*buttons[-2 if release.runtimes.installed() else -1])
        time.sleep(1)
        app.screenshot("settings-defaults")
        shot = 0

        def control(row, column=0):
            nonlocal shot
            shot += 1
            rows = button_rows(app.screenshot(f"controls-{shot:02d}"))
            assert len(rows) == 5, rows
            app.click(*rows[row][column])

        def key(value):
            scan = release.user32.MapVirtualKeyW(value, 0)
            param = 1 | (scan << 16) | ((1 << 24) if value == 0x23 else 0)
            release.user32.PostMessageW(app.window, 0x100, value, param)
            time.sleep(0.1)
            release.user32.PostMessageW(app.window, 0x101, value, param | 0xC0000000)
            time.sleep(0.1)

        def type_splats(value):
            control(2, 1)
            key(0x23)  # End
            for _ in range(10):
                key(0x08)
            for character in str(value):
                release.user32.PostMessageW(app.window, 0x102, ord(character), 0)
                time.sleep(0.1)
            time.sleep(0.3)
            assert json.loads(path.read_text())["max_splats"] == value

        control(0, 1)  # Half
        for _ in range(3):
            control(1, 0)
        for value, column, expected in (
            (150_000, 2, 200_000),
            (150_000, 0, 100_000),
            (124_928, 2, 200_000),
            (124_928, 0, 100_000),
            (200_000, 2, 300_000),
            (100_000, 0, 1),
        ):
            type_splats(value)
            control(2, column)
            time.sleep(0.3)
            assert json.loads(path.read_text())["max_splats"] == expected
        type_splats(375_000)
        control(3)
        time.sleep(1)
        app.screenshot("settings-lowest")
        values = json.loads(path.read_text())
        assert values == dict(
            precision="half", input_resolution=25, max_splats=375_000, diagnostic_logging=True
        ), values
        app.close()
        app = release.App(args.release.resolve(), env, "restart", "--size", "1000", "800")
        time.sleep(2)
        buttons = release.menu_buttons(app.screenshot("restart-menu"))
        app.click(*buttons[-2 if release.runtimes.installed() else -1])
        time.sleep(1)
        app.screenshot("settings-persisted")
        assert json.loads(path.read_text()) == values
        control(4, 0)  # Reset
        time.sleep(0.5)
        assert json.loads(path.read_text()) == dict(
            precision="full", input_resolution=100, max_splats=1_000_000, diagnostic_logging=False
        )
        control(4, 1)  # Back
        time.sleep(0.5)
        image = app.screenshot("back")
        app.click(*release.simulated_robot_button(image))
        app.wait(
            "connected room",
            lambda rows: release.live(rows, flat=False) and rows[-1]["gaussians"] > 1000,
            90,
        )
        app.screenshot("runtime")
        app.close()
        app = None
        print("PASS: released Settings, save, restart, reset, Back, simulation and runtime overlay")
    finally:
        if app and app.process.poll() is None:
            app.close()


if __name__ == "__main__":
    main()
