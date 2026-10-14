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

ARMS = ("TT", "TE", "ET", "EE", "TF")
SETTINGS = "Full / Input 100% / Max 1,000,000"


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
    picture = Image.new("RGB", (columns * w, len(cells) * (h + 58) + 28), (24, 24, 24))
    draw = ImageDraw.Draw(picture)
    draw.text((5, 5), title, font=font, fill="white")
    for row, values in enumerate(cells):
        for col, (image, label) in enumerate(values):
            top = 28 + row * (h + 58)
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
                    f"{SETTINGS}\nfixed exact input prefix / production GL",
                ),
                (
                    mapped,
                    f"{arm} / t={t:.2f}s / shown {rr['t']:.3f}s\n"
                    f"{SETTINGS}\nfixed exact input prefix / production GL",
                ),
            ]
        )
    grid(cells, output, f"{seq} causal motion: RGB | unoptimized {baseline_arm} | optimized {arm}")


def prefixes(out, seq, arm, output):
    k = Intrinsics.model_validate_json((out / seq / "source/camera.json").read_text())
    view = View(k)
    cells = []
    try:
        for t in (0.1, 0.5, 1.0, 2.0):
            row = []
            for name in ("main-TT", arm):
                snap = np.load(out / "visual" / f"{seq}-{name}" / f"snapshot-{t:.1f}.npz")
                assert float(snap["actual"]) <= t
                r = snap["records"]
                r = r[r[:, 0, 3] > 0]
                view.upload(r)
                rgb, _ = view.draw(snap["camera"])
                label = (
                    f"{name} / prefix {t:.1f}s\n{SETTINGS}\n"
                    f"last GL completion {float(snap['actual']):.4f}s"
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
            f"{seq} causal prefixes: main-TT | {arm} (clock starts at first input availability)",
        )


def stereo(out, seq, arm, moment, output):
    k = Intrinsics.model_validate_json((out / seq / "source/camera.json").read_text())
    view = View(k)
    cells = []
    try:
        for name in ("main-TT", arm):
            snap = np.load(out / "visual" / f"{seq}-{name}" / f"snapshot-{moment:.1f}.npz")
            r = snap["records"]
            r = r[r[:, 0, 3] > 0]
            view.upload(r)
            row = []
            for offset in (-0.032, 0.032):
                camera = snap["camera"].copy()
                camera[:3, 3] += camera[:3, 0] * offset
                rgb, _ = view.draw(camera)
                row.append(
                    (
                        Image.fromarray((rgb * 255).astype(np.uint8)),
                        f"{name} / prefix {moment:.1f}s / {'LEFT' if offset < 0 else 'RIGHT'}\n"
                        f"{SETTINGS}\n64mm IPD / calibrated camera FOV / frozen map",
                    )
                )
            cells.append(row)
    finally:
        view.close()
    grid(cells, output, f"{seq}: left | right (main-TT above, {arm} below); not a headset capture")


def interventions(out, seq, moment, output):
    k = Intrinsics.model_validate_json((out / seq / "source/camera.json").read_text())
    view = View(k)
    cells = []
    try:
        for names in (("TT", "TE", "ET"), ("EE", "TF", "main-EE")):
            row = []
            for name in names:
                snap = np.load(out / "visual" / f"{seq}-{name}" / f"snapshot-{moment:.1f}.npz")
                assert float(snap["actual"]) <= moment
                records = snap["records"]
                view.upload(records[records[:, 0, 3] > 0])
                rgb, _ = view.draw(snap["camera"])
                row.append(
                    (
                        Image.fromarray((rgb * 255).astype(np.uint8)),
                        f"{name} / {seq} / prefix {moment:.1f}s\n{SETTINGS}\n"
                        f"latest GL completion {float(snap['actual']):.4f}s",
                    )
                )
            cells.append(row)
    finally:
        view.close()
    grid(cells, output, f"{seq}: identical input prefix; pose/depth TT,TE,ET | EE,ToF,main-EE")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--images", type=Path, default=Path("docs/gsplat-oracle-img"))
    p.add_argument("--sequence", choices=("sweep180", "tour"), required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    args = p.parse_args()
    args.images.mkdir(parents=True, exist_ok=True)
    seq, arm = args.sequence, args.arm
    for t in (0.9, 1.9, 10.0, 15.0):
        motion(args.out, seq, arm, t, args.images / f"{seq}-{arm}-motion-{t:.1f}.jpg")
    prefixes(args.out, seq, arm, args.images / f"{seq}-{arm}-prefix.jpg")
    stereo(
        args.out, seq, arm, 78.0 if seq == "tour" else 10.5, args.images / f"{seq}-{arm}-stereo.jpg"
    )


if __name__ == "__main__":
    main()
