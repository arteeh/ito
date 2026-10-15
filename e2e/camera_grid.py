"""Robot camera frames of by-eye runs side by side, one row per second, to judge by eye.

    python e2e/camera_grid.py OUT.jpg FROM TO RUN_DIR [RUN_DIR ...]

Each RUN_DIR is an e2e/by_eye.sh output (its rec/camera stream); seconds count from the
start of each recording. Shows what the robot's head camera saw, so a head that moves on
its own, a camera that drifts or a feed that stalls shows up next to the pilot's own view.
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

CELL = (180, 320)


def frames(run: Path):
    rec = run / "rec"
    start = json.loads((rec / "recording.json").read_text())["started"]
    rows = [json.loads(line) for line in (rec / "camera/frames.jsonl").read_text().splitlines()]
    return rec / "camera", start, rows


def main():
    out, first, last, *runs = sys.argv[1:]
    first, last = float(first), float(last)
    sources = [frames(Path(r)) for r in runs]
    seconds = range(int(first), int(last) + 1)
    sheet = Image.new("RGB", (60 + CELL[0] * len(seconds), 24 + CELL[1] * len(runs)), "black")
    draw = ImageDraw.Draw(sheet)
    for column, second in enumerate(seconds):
        draw.text((60 + column * CELL[0] + 4, 4), f"{second}s", fill="white")
    for row, (folder, start, rows) in enumerate(sources):
        draw.text((4, 24 + row * CELL[1] + 4), Path(runs[row]).name[:9], fill="white")
        for column, second in enumerate(seconds):
            shown = [r for r in rows if r["t"] - start <= second]
            if not shown:
                continue
            image = Image.open(folder / shown[-1]["file"])
            image.thumbnail(CELL)
            sheet.paste(image, (60 + column * CELL[0], 24 + row * CELL[1]))
    sheet.save(out, quality=88)


if __name__ == "__main__":
    main()
