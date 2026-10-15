"""Make labelled motion comparisons only where both live replays drew the same source view.

.venv/bin/python e2e/estimator_sheets.py RECORDING --main OLD_RUN --branch NEW_RUN \
  --out OUT --labels main,estimator --at 1 10 15
Clocks retain each run's real display time. No image/pose best fitting or quality scores.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def index(root):
    entries = [
        json.loads(row) for row in (root / "motion/pilot/frames.jsonl").read_text().splitlines()
    ]
    return {row["source_frame_id"]: row for row in entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--branch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--labels", default="main,estimator")
    parser.add_argument("--commits", default="unspecified,unspecified")
    parser.add_argument("--at", type=float, nargs="+", required=True)
    args = parser.parse_args()
    before, after = index(args.main), index(args.branch)
    common = sorted(before.keys() & after.keys())
    if len(common) < 2:
        raise RuntimeError("No two exactly shared source views; record both replays again")
    sources = [
        json.loads(s) for s in (args.recording / "source/frames.jsonl").read_text().splitlines()
    ]
    source = {row["source_frame_id"]: row for row in sources}
    args.out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("DejaVuSans.ttf", 12)
    labels = args.labels.split(",")
    commits = args.commits.split(",")
    displays = [
        json.loads((root / "display.json").read_text()) for root in (args.main, args.branch)
    ]
    k = json.loads((args.recording / "source/camera.json").read_text())
    width = 400 if k["width"] > k["height"] else 336
    image_height = round(width * k["height"] / k["width"])
    row_height = image_height + 125
    metadata = []
    for moment in args.at:
        candidates = sorted(common, key=lambda i: abs(after[i]["t"] - moment))
        first = candidates[0]
        later = [i for i in common if source[i]["captured"] > source[first]["captured"] + 0.05]
        if not later:
            continue
        second = min(
            later, key=lambda i: abs(source[i]["captured"] - source[first]["captured"] - 0.1)
        )
        sheet = Image.new("RGB", (3 * width, 2 * row_height + 32), (24, 24, 24))
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (5, 5),
            "Exact source views: RGB | " + labels[0] + " | " + labels[1],
            font=font,
            fill="white",
        )
        cells = []
        for row, i in enumerate((first, second)):
            stamp = source[i]["captured"] - sources[0]["captured"]
            rgb = np.load(args.recording / "source" / source[i]["file"])["rgb"]
            values = [
                (
                    Image.fromarray(rgb),
                    (
                        f"RGB / source {i} / exposure +{stamp:.3f}s\n"
                        f"{k['width']}x{k['height']} source / sim lens / v8 policy"
                    ),
                )
            ]
            for root, entries, label, commit, display in (
                (args.main, before, labels[0], commits[0], displays[0]),
                (args.branch, after, labels[1], commits[1], displays[1]),
            ):
                entry = entries[i]
                settings = entry["settings"]
                budget = display.get("render_budget")
                render_cap = "uncapped" if budget is None else f"{budget:,}"
                camera_mode = entry.get("camera_mode", "legacy camera: see report")
                if camera_mode == "own published pose":
                    camera_mode = (
                        "true source pose"
                        if entry["arm"].removeprefix("main-")[0] == "T"
                        else "last visual pose (legacy replay)"
                    )
                if camera_mode == "last visual pose (main)":
                    camera_mode = "last available visual pose"
                caption = (
                    f"{label} / {entry['arm']} / source {i}\n"
                    f"display +{entry['t']:.3f}s / exposure +{stamp:.3f}s\n"
                    f"{settings['precision']} / Input {settings['input_resolution']}% / "
                    f"Max {settings['max_splats']:,}\n"
                    f"{camera_mode}\n"
                    f"code {commit} / render cap {render_cap}"
                )
                if settings.get("ignore_depth_confidence"):
                    caption += "\nconfidence IGNORED (ablation)"
                values.append(
                    (Image.open(root / "motion/pilot" / entry["file"]).convert("RGB"), caption)
                )
                cells.append(
                    dict(
                        row=row + 1,
                        column=len(values),
                        label=label,
                        source_frame_id=i,
                        display=entry["t"],
                        settings=settings,
                        code=commit,
                        camera_mode=camera_mode,
                        render_budget=display.get("render_budget"),
                    )
                )
            for column, (image, caption) in enumerate(values):
                y = 32 + row * row_height
                sheet.paste(image.resize((width, image_height)), (column * width, y))
                draw.multiline_text(
                    (column * width + 5, y + image_height + 3),
                    caption,
                    font=font,
                    fill="white",
                    spacing=3,
                )
        name = f"motion-{moment:g}.jpg"
        sheet.save(args.out / name, quality=92)
        metadata.append(dict(file=name, requested_branch_time=moment, cells=cells))
    (args.out / "sheets.json").write_text(json.dumps(metadata, indent=2))
    print(
        json.dumps(dict(exact_common_views=len(common), sheets=len(metadata), output=str(args.out)))
    )


if __name__ == "__main__":
    main()
