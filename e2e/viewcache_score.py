"""Score 1:1 truth/main/#47/cache sheets with the unchanged baseline metrics."""

import argparse
import json
from pathlib import Path

import numpy as np
from measure_images import Metrics
from PIL import Image, ImageDraw


def sheet(images, captions, path):
    h, w = images[0].shape[:2]
    result = Image.new("RGB", (w, (h + 36) * len(images)), (24, 24, 24))
    draw = ImageDraw.Draw(result)
    for k, (im, caption) in enumerate(zip(images, captions, strict=True)):
        draw.text((8, k * (h + 36) + 10), caption, fill="white")
        result.paste(Image.fromarray(im), (0, k * (h + 36) + 36))
    for q in (90, 80, 70, 60):
        result.save(path, quality=q)
        if path.stat().st_size < 1_000_000:
            break


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    args = p.parse_args()
    report = json.loads((args.folder / "render.json").read_text())
    metric = Metrics()
    panels = {}
    for row in report["moments"]:
        name = f"{row['run']}-{row['frame']:04d}-{row['view']}"
        im = np.asarray(Image.open(args.folder / f"{name}.png").convert("RGB"))
        w = im.shape[1] // 4
        masks = np.load(args.folder / f"{name}-mask.npz")
        row.update(
            metric.score(
                im[:, :w].astype("f4") / 255,
                im[:, -w:].astype("f4") / 255,
                masks["covered"],
                masks["seen"],
            )
        )
        panels[(row["run"], row["frame"], row["view"])] = im
        print(name, f"{row['psnr_db']:.2f} dB LPIPS {row['lpips']:.3f}", flush=True)
    report["means"] = {}
    for run, frames in [("sweep180", [215, 453, 693]), ("tour", [363, 1501, 2284])]:
        eyes = [
            r
            for r in report["moments"]
            if r["run"] == run and r["frame"] in frames and r["view"] in ("left", "right")
        ]
        report["means"][run] = {
            k: float(np.mean([r[k] for r in eyes]))
            for k in ("psnr_db", "ssim", "lpips", "coverage", "previously_seen_holes")
        }
        for label, keys in [
            ("camera", [(run, f, "camera") for f in frames]),
            ("eyes", [(run, f, e) for f in frames for e in ("left", "right")]),
        ]:
            sheet(
                [panels[k] for k in keys],
                [f"{run} #{f} {v} | truth / main / #47 / view-cache" for _, f, v in keys],
                args.folder / f"{run}-{label}.jpg",
            )
    sheet(
        [panels[("sweep180", 526, v)] for v in ("camera", "left", "right")],
        [f"hard #526 {v} | truth / main / #47 / view-cache" for v in ("camera", "left", "right")],
        args.folder / "hard-526.jpg",
    )
    (args.folder / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["means"], indent=2))


if __name__ == "__main__":
    main()
