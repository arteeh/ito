"""The robot camera and each run's pilot view at the same moments, to judge by eye.

    python e2e/view_grid.py OUT.jpg SECONDS RUN_DIR[:HFOV] [RUN_DIR[:HFOV] ...]

SECONDS is a comma-separated list (from each recording's start). Row per moment; per run, its
robot camera frame and its pilot view, each at its own aspect (a zero3 camera is portrait, a
beta's landscape). The pilot's eye sits at the robot camera, so where they look the same way
the scene should match the camera: things the scene shows farther, nearer, smeared or shifted
stand out. The desktop pilot view is wider than any robot camera (70 degrees vertical); with
:HFOV, the camera's field across its long side, the pilot view is cropped to the camera's
field, so the two show the same angles and sizes compare directly.
"""

import json
import math
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HEIGHT = int(os.environ.get("GRID_HEIGHT", 300))
PILOT_VFOV = math.radians(70)  # ito/desktop/window.py's default


def stream(run: Path, name: str):
    rec = run / "rec"
    start = json.loads((rec / "recording.json").read_text())["started"]
    rows = [json.loads(s) for s in (rec / name / "frames.jsonl").read_text().splitlines()]
    return rec / name, start, rows


def at(source, second):
    folder, start, rows = source
    shown = [r for r in rows if r["t"] - start <= second]
    return Image.open(folder / shown[-1]["file"]) if shown else None


def main():
    out, seconds, *runs = sys.argv[1:]
    seconds = [float(s) for s in seconds.split(",")]
    columns = []
    for argument in runs:
        run, _, hfov = argument.partition(":")
        run = Path(run)
        crop = None
        for kind in ("camera", "pilot"):
            source = stream(run, kind)
            folder, _, rows = source
            with Image.open(folder / rows[len(rows) // 2]["file"]) as sample:
                aspect = sample.width / sample.height
            if kind == "camera" and hfov:
                # Half-angle tangents of the camera's field against the pilot view's.
                long = math.tan(math.radians(float(hfov)) / 2)
                across, up = (long, long / aspect) if aspect >= 1 else (long * aspect, long)
                crop = (across, up)
            if kind == "pilot" and crop:
                aspect = crop[0] / crop[1]
            columns.append(
                (
                    f"{run.name} {kind}",
                    source,
                    round(HEIGHT * aspect),
                    crop if kind == "pilot" else None,
                )
            )
    sheet = Image.new(
        "RGB", (60 + sum(c[2] for c in columns) + 4 * len(columns), 24 + len(seconds) * HEIGHT)
    )
    draw = ImageDraw.Draw(sheet)
    x = 60
    for name, _, width, _ in columns:
        draw.text((x + 4, 4), name, fill="white")
        x += width + 4
    for row, second in enumerate(seconds):
        y = 24 + row * HEIGHT
        draw.text((4, y + 4), f"{second:g}s", fill="white")
        x = 60
        for _, source, width, crop in columns:
            image = at(source, second)
            if image is not None and crop:
                up = math.tan(PILOT_VFOV / 2)
                w, h = image.size
                half_w = w / 2 * crop[0] / (up * w / h)
                half_h = h / 2 * crop[1] / up
                image = image.crop(
                    (
                        round(w / 2 - half_w),
                        round(h / 2 - half_h),
                        round(w / 2 + half_w),
                        round(h / 2 + half_h),
                    )
                )
            if image is not None:
                image = image.resize((width, HEIGHT))
                sheet.paste(image, (x, y))
            x += width + 4
    sheet.save(out, quality=88)


if __name__ == "__main__":
    main()
