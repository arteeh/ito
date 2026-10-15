"""Label causal motion cells and render near-eye comparisons with production GL.

PYTHONPATH=. .venv/bin/python e2e/oracle_report.py --out ~/Projects/ito/out/oracle-40
Uses software EGL for snapshot rendering; no timing conclusions come from this pass.
"""

import argparse
import bisect
import json
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ["GALLIUM_DRIVER"] = "llvmpipe"

import numpy as np
from measure_gl import View
from PIL import Image, ImageDraw, ImageFont

from ito.protocol import Intrinsics
from ito.reconstruction.slam_image import SLAMImage
from ito.render.renderer import LIVE_RENDER_SPLATS

ARMS = ("TT", "TE", "ET", "EE", "TF")


def settings_label(out, seq, arm):
    settings = json.loads((out / "visual" / f"{seq}-{arm}" / "display.json").read_text())[
        "settings"
    ]
    return (
        f"{settings['precision'].title()} / Input {settings['input_resolution']}% / "
        f"Max {settings['max_splats']:,}\n"
        f"code {settings.get('code_commit', '0634bbd')[:7]} / 360p30 raw360x640"
    )


def view_label(out, seq, arm):
    folder = out / "visual" / f"{seq}-{arm}"
    display = json.loads((folder / "display.json").read_text())
    k = Intrinsics.model_validate_json((out / seq / "source/camera.json").read_text())
    worker = (
        json.loads((folder / "worker.json").read_text())
        if (folder / "worker.json").exists()
        else {}
    )
    size = worker.get("training_size") or (
        SLAMImage(k, display["settings"]["input_resolution"]).scaled
        if arm.startswith("main-")
        else (k.width, k.height)
    )
    cap = display.get("render_budget") or "none"
    return f"own {arm.removeprefix('main-')[0]} / grid {size[0]}x{size[1]} / cap {cap}"


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def at(recording, kind, t):
    index = rows(recording / kind / "frames.jsonl")
    pos = bisect.bisect_right([r["t"] for r in index], t) - 1
    if pos < 0:
        return Image.new("RGB", (360, 640)), None
    r = index[pos]
    return Image.open(recording / kind / r["file"]).convert("RGB"), r


def grid(cells, path, title):
    columns = len(cells[0])
    w, h = 360, 640
    font = ImageFont.truetype("DejaVuSans.ttf", 13)
    picture = Image.new("RGB", (columns * w, len(cells) * (h + 76) + 28), (24, 24, 24))
    draw = ImageDraw.Draw(picture)
    draw.text((5, 5), title, font=font, fill="white")
    for row, values in enumerate(cells):
        for col, (image, label) in enumerate(values):
            top = 28 + row * (h + 76)
            picture.paste(image.resize((w, h)), (col * w, top))
            draw.multiline_text(
                (col * w + 5, top + h + 3), label, font=font, fill="white", spacing=3
            )
    picture.save(path, quality=92)


def motion(out, seq, arm, start, output, baseline_arm="main-TT"):
    baseline = out / "visual" / f"{seq}-{baseline_arm}" / "motion"
    candidate = out / "visual" / f"{seq}-{arm}" / "motion"
    cells = []
    for t in (start, start + 0.1):
        camera, cr = at(baseline, "camera", t)
        main, mr = at(baseline, "pilot", t)
        mapped, rr = at(candidate, "pilot", t)
        cells.append(
            [
                (
                    camera,
                    f"RGB reference / {seq} / t={t:.2f}s\n"
                    f"source frame {cr.get('captured') if cr else 'none'}\n"
                    "360p30, instantaneous sim exposure",
                ),
                (
                    main,
                    f"{baseline_arm} / t={t:.2f}s / shown {mr['t']:.3f}s\n"
                    f"{settings_label(out, seq, baseline_arm)}\n"
                    f"{view_label(out, seq, baseline_arm)}",
                ),
                (
                    mapped,
                    f"{arm} / t={t:.2f}s / shown {rr['t']:.3f}s\n"
                    f"{settings_label(out, seq, arm)}\n{view_label(out, seq, arm)}",
                ),
            ]
        )
    grid(cells, output, f"{seq} causal motion: RGB | unoptimized {baseline_arm} | optimized {arm}")


def prefixes(out, seq, arm, output, baseline_arm="main-TT"):
    k = Intrinsics.model_validate_json((out / seq / "source/camera.json").read_text())
    view = View(k, render_budget=LIVE_RENDER_SPLATS)
    cells = []
    try:
        for t in (0.1, 0.5, 1.0, 2.0):
            row = []
            for name in (baseline_arm, arm):
                snap = np.load(out / "visual" / f"{seq}-{name}" / f"snapshot-{t:.1f}.npz")
                assert float(snap["actual"]) <= t
                r = snap["records"]
                view.renderer.render_budget = (
                    int(snap["render_budget"]) or None if "render_budget" in snap else None
                )
                view.upload(r, stable_slots=True)
                rgb, _ = view.draw(snap["camera"])
                label = (
                    f"{name} / prefix {t:.1f}s / GL {float(snap['actual']):.3f}s\n"
                    f"{settings_label(out, seq, name)}\n{view_label(out, seq, name)}"
                )
                row.append((Image.fromarray((rgb * 255).astype(np.uint8)), label))
            cells.append(row)
    finally:
        view.close()
    # Two pages keep the original pixel size and legible captions.
    for n in range(2):
        grid(
            cells[n * 2 : n * 2 + 2],
            output.with_name(output.stem + f"-{n + 1}.jpg"),
            f"{seq} causal prefixes: {baseline_arm} | {arm}; first-input clock",
        )


def stereo(out, seq, arm, moment, output, baseline_arm="main-TT"):
    k = Intrinsics.model_validate_json((out / seq / "source/camera.json").read_text())
    view = View(k, render_budget=LIVE_RENDER_SPLATS)
    cells = []
    try:
        for name in (baseline_arm, arm):
            snap = np.load(out / "visual" / f"{seq}-{name}" / f"snapshot-{moment:.1f}.npz")
            r = snap["records"]
            view.renderer.render_budget = (
                int(snap["render_budget"]) or None if "render_budget" in snap else None
            )
            view.upload(r, stable_slots=True)
            row = []
            for offset in (-0.032, 0.032):
                camera = snap["camera"].copy()
                camera[:3, 3] += camera[:3, 0] * offset
                rgb, _ = view.draw(camera)
                row.append(
                    (
                        Image.fromarray((rgb * 255).astype(np.uint8)),
                        f"{name} / prefix {moment:.1f}s / {'LEFT' if offset < 0 else 'RIGHT'}\n"
                        f"{settings_label(out, seq, name)}\n"
                        f"{view_label(out, seq, name)}",
                    )
                )
            cells.append(row)
    finally:
        view.close()
    grid(
        cells,
        output,
        f"{seq}: left | right ({baseline_arm} above, {arm} below); 64mm, camera FOV, no headset",
    )


def interventions(out, seq, moment, output):
    k = Intrinsics.model_validate_json((out / seq / "source/camera.json").read_text())
    view = View(k, render_budget=LIVE_RENDER_SPLATS)
    cells = []
    try:
        for names in (("TT", "TE", "ET"), ("EE", "TF", "main-EE")):
            row = []
            for name in names:
                snap = np.load(out / "visual" / f"{seq}-{name}" / f"snapshot-{moment:.1f}.npz")
                assert float(snap["actual"]) <= moment
                records = snap["records"]
                view.renderer.render_budget = (
                    int(snap["render_budget"]) or None if "render_budget" in snap else None
                )
                view.upload(records, stable_slots=True)
                rgb, _ = view.draw(snap["camera"])
                row.append(
                    (
                        Image.fromarray((rgb * 255).astype(np.uint8)),
                        f"{name} / {seq} / prefix {moment:.1f}s\n{settings_label(out, seq, name)}\n"
                        f"latest GL completion {float(snap['actual']):.4f}s",
                    )
                )
            cells.append(row)
    finally:
        view.close()
    grid(cells, output, f"{seq}: identical input prefix; pose/depth TT,TE,ET | EE,ToF,main-EE")


def before_after(out, previous, seq, arm, output, baseline_arm="main-TT"):
    """Same own camera for both saved maps, including the round-one estimated arms."""
    k = Intrinsics.model_validate_json((out / seq / "source/camera.json").read_text())
    view = View(k, render_budget=LIVE_RENDER_SPLATS)
    cells = []
    try:
        for moment in (0.5, 1.0):
            current = np.load(out / "visual" / f"{seq}-{arm}" / f"snapshot-{moment:.1f}.npz")
            camera = current["camera"]
            row = []
            for root, name, label in (
                (out, baseline_arm, baseline_arm),
                (previous, arm, f"round 1 {arm}"),
                (out, arm, f"round 2 {arm}"),
            ):
                snap = np.load(root / "visual" / f"{seq}-{name}" / f"snapshot-{moment:.1f}.npz")
                assert float(snap["actual"]) <= moment
                view.renderer.render_budget = (
                    int(snap["render_budget"]) or None if "render_budget" in snap else None
                )
                records = snap["records"]
                view.upload(records, stable_slots=True)
                rgb, _ = view.draw(camera)
                row.append(
                    (
                        Image.fromarray((rgb * 255).astype(np.uint8)),
                        f"{label} / prefix {moment:.1f}s / own {arm[0]} pose\n"
                        f"{settings_label(root, seq, name)}\n"
                        f"{view_label(root, seq, name)}",
                    )
                )
            cells.append(row)
    finally:
        view.close()
    grid(cells, output, f"{seq}: matched main | round 1 | round 2; SAME own camera")


def native_views(out, seq, moment, output):
    """Show native 90-degree selection, both whole eyes and explicit camera-FOV crops."""
    for crop, suffix in ((True, "camera-crop"), (False, "whole")):
        cells = []
        for arm in ("main-TT", "TT", "TF"):
            folder = out / "native-views" / f"{seq}-{arm}-{moment:.1f}"
            info = json.loads((folder / "capture.json").read_text())
            row = []
            for eye in ("left", "right"):
                picture = Image.open(folder / f"{eye}-{suffix}.png").convert("RGB")
                if not crop:
                    square = picture.resize((360, 360))
                    picture = Image.new("RGB", (360, 640))
                    picture.paste(square, (0, 140))
                row.append(
                    (
                        picture,
                        f"native {arm} / {moment:.1f}s / {eye.upper()}\n"
                        f"{settings_label(out, seq, arm)}\n"
                        f"90deg {'CAMERA CROP' if crop else 'WHOLE EYE'} / "
                        f"cap {info['render_budget'] or 'none'}",
                    )
                )
            cells.append(row)
        grid(
            cells,
            output.with_name(output.stem + f"-{suffix}.jpg"),
            f"{seq}: native GL / 64mm IPD / head centered / frozen causal map; main | TT | ToF",
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--images", type=Path, default=Path("docs/gsplat-oracle-img"))
    p.add_argument("--sequence", choices=("sweep180", "tour"), required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--matched-controls", action="store_true")
    p.add_argument("--previous", type=Path)
    args = p.parse_args()
    args.images.mkdir(parents=True, exist_ok=True)
    seq, arm = args.sequence, args.arm
    baseline = f"main-{arm}" if args.matched_controls and arm != "TF" else "main-TT"
    if args.previous:
        before_after(
            args.out,
            args.previous,
            seq,
            arm,
            args.images / f"{seq}-{arm}-before-after.jpg",
            baseline,
        )
    for t in (0.9, 1.9, 10.0 if seq == "sweep180" else 50.0, 15.0 if seq == "sweep180" else 76.0):
        motion(args.out, seq, arm, t, args.images / f"{seq}-{arm}-motion-{t:.1f}.jpg", baseline)
    prefixes(args.out, seq, arm, args.images / f"{seq}-{arm}-prefix.jpg", baseline)
    stereo(
        args.out,
        seq,
        arm,
        78.0 if seq == "tour" else 10.5,
        args.images / f"{seq}-{arm}-stereo.jpg",
        baseline,
    )


if __name__ == "__main__":
    main()
