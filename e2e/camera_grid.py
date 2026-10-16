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
    # Each run's cells keep its camera's aspect: a zero3's is portrait, a beta's landscape.
    cells = []
    for folder, _, rows in sources:
        with Image.open(folder / rows[len(rows) // 2]["file"]) as sample:
            portrait = sample.height > sample.width
        cells.append(CELL if portrait else CELL[::-1])
    seconds = range(int(first), int(last) + 1)
    column_width = max(cell[0] for cell in cells)
    tops = [24 + sum(cell[1] for cell in cells[:row]) for row in range(len(cells))]
    sheet = Image.new(
        "RGB", (60 + column_width * len(seconds), 24 + sum(c[1] for c in cells)), "black"
    )
    draw = ImageDraw.Draw(sheet)
    for column, second in enumerate(seconds):
        draw.text((60 + column * column_width + 4, 4), f"{second}s", fill="white")
    for row, (folder, start, rows) in enumerate(sources):
        draw.text((4, tops[row] + 4), Path(runs[row]).name[:9], fill="white")
        for column, second in enumerate(seconds):
            shown = [r for r in rows if r["t"] - start <= second]
            if not shown:
                continue
            image = Image.open(folder / shown[-1]["file"])
            image.thumbnail(cells[row])
            sheet.paste(image, (60 + column * column_width, tops[row]))
    sheet.save(out, quality=88)


if __name__ == "__main__":
    main()
