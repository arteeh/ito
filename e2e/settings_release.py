"""Click Settings in the real ito.exe release, restart it, and connect to its simulation.

Windows: python e2e/settings_release.py --release C:/path/ito --out C:/path/output
"""

import argparse
import json
import os
import time
from pathlib import Path

from e2e import release


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
        # The settings panel is centred and uses fixed-size button rows.
        app.click(575, 248)  # Half
        for _ in range(3):
            app.click(344, 326)  # Resolution -
        app.click(344, 428)  # Splats -
        # Type a different global count through the real desktop edit box.
        app.click(455, 417)

        def key(value):
            scan = release.user32.MapVirtualKeyW(value, 0)
            param = 1 | (scan << 16) | ((1 << 24) if value == 0x23 else 0)
            release.user32.PostMessageW(app.window, 0x100, value, param)
            time.sleep(0.1)
            release.user32.PostMessageW(app.window, 0x101, value, param | 0xC0000000)
            time.sleep(0.1)

        key(0x23)  # End
        for _ in range(6):
            key(0x08)  # Backspace
        for character in "375000":
            release.user32.PostMessageW(app.window, 0x102, ord(character), 0)
            time.sleep(0.2)
        app.click(385, 545)  # Logging on
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
        app.click(432, 611)  # Reset
        time.sleep(0.5)
        assert json.loads(path.read_text()) == dict(
            precision="full", input_resolution=100, max_splats=1_000_000, diagnostic_logging=False
        )
        app.click(623, 611)  # Back
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
