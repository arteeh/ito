"""The robot camera and each run's pilot view at the same moments, to judge by eye.

    python e2e/view_grid.py OUT.jpg SECONDS RUN_DIR [RUN_DIR ...]

SECONDS is a comma-separated list (from each recording's start). Row per moment: the first
run's robot camera frame, then every run's pilot view. The pilot's eye sits at the robot
camera, so where they look the same way the scene should match the camera: things the
scene shows farther, nearer, smeared or shifted stand out.
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HEIGHT = 300


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
    columns = [("camera", stream(Path(runs[0]), "camera"))]
    columns += [(Path(r).name, stream(Path(r), "pilot")) for r in runs]
    widths = [HEIGHT * 9 // 16 if name == "camera" else HEIGHT * 16 // 9 for name, _ in columns]
    sheet = Image.new("RGB", (60 + sum(widths) + 4 * len(widths), 24 + len(seconds) * HEIGHT))
    draw = ImageDraw.Draw(sheet)
    x = 60
    for (name, _), width in zip(columns, widths, strict=True):
        draw.text((x + 4, 4), name, fill="white")
        x += width + 4
    for row, second in enumerate(seconds):
        y = 24 + row * HEIGHT
        draw.text((4, y + 4), f"{second:g}s", fill="white")
        x = 60
        for (_, source), width in zip(columns, widths, strict=True):
            image = at(source, second)
            if image is not None:
                image.thumbnail((width, HEIGHT))
                sheet.paste(image, (x, y))
            x += width + 4
    sheet.save(out, quality=88)


if __name__ == "__main__":
    main()
