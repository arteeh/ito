"""Baseline residual-change and edge diagnostics on matched #47/cache temporal frames."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import numpy as np
from measure_images import Room, edges, percentiles
from PIL import Image, ImageDraw
from viewcache_score import sheet

from ito.protocol import Intrinsics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("--work", type=Path, default=Path("..").resolve())
    args = p.parse_args()
    report = json.loads((args.folder / "render.json").read_text())
    for row in report["clips"]:
        run, basename = row["run"], row["clip"]
        folder = args.folder / run / basename
        data = np.load(folder / "frames.npz")
        baseline, room = None, None
        if run == "sweep180":
            baseline = imageio.get_reader(args.work / f"m1-out/verified/repeat-1/{basename}.mp4")
        else:
            i = Intrinsics.model_validate_json(
                (args.work / f"p47-out/prepared/{run}/camera.json").read_text()
            )
            room = Room(args.work / "truth/data/scan/scene.mjb", i)
        measures = {
            name: dict(residual=[], edges=[], coverage=[]) for name in ("main", "mesh", "cache")
        }
        previous = {}
        panels, captions = [], []
        writer = imageio.get_writer(folder / "comparison.mp4", fps=30, macro_block_size=1)
        slow = imageio.get_writer(folder / "comparison-slow.mp4", fps=10, macro_block_size=1)
        for n, camera in enumerate(data["cameras"]):
            h, w = data["cache"][n].shape[:2]
            if baseline:
                pair = baseline.get_data(n)[-h:, : 2 * w]
                truth = pair[:, :w].astype("f4") / 255
                main_image = pair[:, w:]
            else:
                truth = room.draw(camera)
                main_image = None
            im = {}
            if main_image is not None:
                im["main"] = main_image
            for name in ("mesh", "cache"):
                im[name] = data[name][n, :, :, :3]
                measures[name]["coverage"].append(float((data[name][n, :, :, 3] > 127).mean()))
            for name, image in im.items():
                rgb = image.astype("f4") / 255
                residual = rgb - truth
                edge = edges(rgb)
                if name in previous:
                    measures[name]["residual"].append(
                        float(np.abs(residual - previous[name][0]).mean())
                    )
                    measures[name]["edges"].append(float(np.mean(edge != previous[name][1])))
                previous[name] = residual, edge
            truth_rgb = (truth * 255).round().clip(0, 255).astype("u1")
            images = (
                [truth_rgb]
                + ([main_image] if main_image is not None else [])
                + [im["mesh"], im["cache"]]
            )
            panel = np.concatenate(images, axis=1)
            caption = (
                f"{run} {basename} t={n / 30:.2f}s | truth / "
                + ("main / " if main_image is not None else "")
                + "#47 / cache"
            )
            margin = np.zeros((32, panel.shape[1], 3), "u1")
            labelled = Image.fromarray(np.concatenate((margin, panel)))
            ImageDraw.Draw(labelled).text((8, 8), caption, fill="white")
            writer.append_data(np.asarray(labelled))
            slow.append_data(np.asarray(labelled))
            if n in (0, 15, 30, 45, 60):
                panels.append(panel)
                captions.append(caption)
        writer.close()
        slow.close()
        if baseline:
            baseline.close()
        if room:
            room.renderer.close()
        sheet(panels, captions, folder / "filmstrip.jpg")
        row["metrics"] = {
            name: dict(
                residual_change=percentiles(v["residual"]),
                edge_changed_fraction=percentiles(v["edges"]),
                minimum_coverage=min(v["coverage"]) if v["coverage"] else None,
            )
            for name, v in measures.items()
        }
        for key in ("residual_change", "edge_changed_fraction"):
            a, b = [row["metrics"][name][key]["p95"] for name in ("mesh", "cache")]
            row[key + "_improvement"] = 1 - b / a if a else None
        row["main_note"] = (
            "Frozen baseline compressed clip, same truth for paired candidates"
            if baseline
            else "No frozen main tour temporal clip: not fabricated"
        )
        print(run, basename, json.dumps(row["metrics"]), flush=True)
    (args.folder / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
