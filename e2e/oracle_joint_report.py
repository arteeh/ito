"""Compare recorded own-pose views at common source time; no new pixels from the future.

PYTHONPATH=.:e2e python e2e/oracle_joint_report.py --out OUT --stage before|after
"""

import argparse
import json
from pathlib import Path

import numpy as np
from oracle_report import View, at
from PIL import Image, ImageDraw, ImageFont

from ito.protocol import Intrinsics


def sheet(cells, target, title):
    width, height, label_height = 300, 533, 76
    font = ImageFont.truetype("DejaVuSans.ttf", 11)
    pic = Image.new(
        "RGB", (width * len(cells[0]), 28 + len(cells) * (height + label_height)), (24, 24, 24)
    )
    draw = ImageDraw.Draw(pic)
    draw.text((4, 5), title, font=font, fill="white")
    for y, row in enumerate(cells):
        top = 28 + y * (height + label_height)
        for x, (image, label) in enumerate(row):
            pic.paste(image.resize((width, height)), (x * width, top))
            draw.multiline_text(
                (x * width + 4, top + height + 2), label, font=font, spacing=2, fill="white"
            )
    target.parent.mkdir(parents=True, exist_ok=True)
    pic.save(target, quality=90)


def metadata(root, seq, arm):
    return json.loads((root / "visual" / f"{seq}-{arm}" / "display.json").read_text())


def label(meta, name, arm, age, source_time, shown):
    s = meta["settings"]
    code = s.get("code_commit", "0634bbd")[:7]
    mode = meta.get("camera_mode", "")
    mode = (
        "true source"
        if arm.removeprefix("main-")[0] == "T"
        else "motion+vision"
        if "motion propagated" in mode
        else "held visual"
    )
    return (
        f"{name} {arm} / age {age:.3f}s / GL {shown:.3f}s\n"
        f"source time {source_time:.3f}s / own {arm.removeprefix('main-')[0]} pose\n"
        f"{s['precision'].title()} / Input{s['input_resolution']}% / Max{s['max_splats']:,}\n"
        f"360p30 raw360x640 / fit288x512 / code {code}\n"
        f"map {s.get('mapper_commit', code)[:7]} / cap {meta.get('render_budget') or 'none'}"
    )


def motion(roots, seq, arm, moment, target, *, rgb=False):
    current = metadata(roots[-1][0], seq, arm)
    source_zero = current["settings"]["view_start"]
    baseline = f"main-{arm}" if arm != "TF" else "main-TT"
    cells = []
    for age in (moment, moment + 0.1):
        source_time = source_zero + age
        row = []
        if rgb:
            picture, capture = at(
                roots[-1][0] / "visual" / f"{seq}-{baseline}" / "motion", "camera", age
            )
            s = current["settings"]
            row.append(
                (
                    picture,
                    f"RGB source / {seq} / t={age:.3f}s\n"
                    f"source {source_time:.3f}s / frame "
                    f"{capture.get('captured') if capture else 'none'}\n"
                    f"{s['precision'].title()} / Input{s['input_resolution']}% / "
                    f"Max{s['max_splats']:,}\n"
                    f"360p30 RAW360x640 / code {s.get('code_commit', 'unknown')[:7]}\n"
                    "true physical camera / map cap N/A",
                )
            )
        for root, name, override in [
            (roots[-1][0], "main placer", baseline),
            *[(r, n, arm) for r, n in roots],
        ]:
            meta = metadata(root, seq, override)
            t = source_time - meta["settings"].get("view_start", meta["settings"].get("zero", 0))
            # Round2 recorded view_start is implicit: recover from worker's causal clock.
            if "view_start" not in meta["settings"]:
                w = json.loads((root / "visual" / f"{seq}-{override}" / "worker.json").read_text())
                t = source_time - w["zero"]
            image, shown = at(root / "visual" / f"{seq}-{override}" / "motion", "pilot", t)
            row.append(
                (image, label(meta, name, override, t, source_time, shown["t"] if shown else -1))
            )
        cells.append(row)
    sheet(
        cells,
        target,
        f"{seq} {arm}: common source time / own poses / main uses final input schedule",
    )


def prefixes(root, seq, arm, target):
    meta = metadata(root, seq, arm)
    zero = meta["settings"]["view_start"]
    k = Intrinsics.model_validate_json((root / seq / "source/camera.json").read_text())
    view = View(k, render_budget=meta.get("render_budget"))
    cells = []
    try:
        for ages in ((0.1, 0.5), (1.0, 2.0)):
            row = []
            for age in ages:
                data = np.load(root / "visual" / f"{seq}-{arm}" / f"snapshot-{age:.1f}.npz")
                assert float(data["actual"]) <= age
                view.upload(data["records"], stable_slots=True)
                rgb, _ = view.draw(data["camera"])
                row.append(
                    (
                        Image.fromarray((rgb * 255).astype(np.uint8)),
                        label(meta, "round3", arm, age, zero + age, float(data["actual"])),
                    )
                )
            cells.append(row)
    finally:
        view.close()
    sheet(cells, target, f"{seq} {arm}: causal 100ms | 500ms / 1s | 2s; last completed map")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stage", choices=("before", "after"), required=True)
    p.add_argument("--images", type=Path)
    p.add_argument("--sequence", choices=("sweep180", "tour"))
    args = p.parse_args()
    image_root = args.images or args.out / "review" / args.stage
    roots = [
        (Path.home() / "Projects/ito/out/oracle-40/round2", "round2 oldE"),
        (args.out / "before", "round3 newE before"),
    ]
    if args.stage == "after":
        roots = [
            (args.out / "paired", "R2 mapper latestE"),
            (args.out / "after", "R3 mapper latestE"),
        ]
    for seq in [args.sequence] if args.sequence else ("sweep180", "tour"):
        for arm in ("TT", "TE", "ET", "EE", "TF"):
            for moment in (
                0.9,
                10.0 if seq == "sweep180" else 76.0 if arm in ("TT", "TF") else 50.0,
            ):
                motion(roots, seq, arm, moment, image_root / f"{seq}-{arm}-motion-{moment:.1f}.jpg")
            if args.stage == "after":
                prefixes(roots[-1][0], seq, arm, image_root / f"{seq}-{arm}-prefix.jpg")


if __name__ == "__main__":
    main()
