"""Audit all candidate prefixes and assemble the small, committed evidence bundle."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from depthmesh_evaluate import sheet
from PIL import Image

VARIANTS = (
    "truth",
    "da2-small",
    "da2-large",
    "metric-small",
    "metric-large",
    "video-small",
    "prompt-small",
)


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def aggregate(rows):
    return {
        key: float(np.mean([r[key] for r in rows]))
        for key in ("psnr_db", "ssim", "lpips", "coverage", "previously_seen_holes")
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("../p47-out").resolve())
    p.add_argument("--work", type=Path, default=Path("..").resolve())
    p.add_argument("--out", type=Path, default=Path("docs/proto-depth-mesh").resolve())
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result = dict(
        scope="Static legacy recordings; held-out prefixes and novel eyes, not physical VR",
        inputs=json.loads((args.root / "prepared/inputs.json").read_text()),
        summary=[],
        variants={},
        inference={},
        gpu={},
        audit={},
        code_sha256={},
    )
    baseline = json.loads((args.work / "m1-out/verified/repeat-1/metrics.json").read_text())

    def summaries(name, rows):
        for run in ("sweep180", "tour"):
            for views in (("camera",), ("left", "right")):
                chosen = [
                    r for r in rows if r["run"] == run and r["view"] in views and r["frame"] != 526
                ]
                result["summary"].append(
                    dict(variant=name, run=run, views=list(views), **aggregate(chosen))
                )

    summaries("main", baseline["moments"])
    result["main_moments"] = baseline["moments"]
    audited = held_out = 0
    for variant in VARIANTS:
        for pose in ("oracle", "robot"):
            name = f"{variant}-{pose}-cap64"
            path = args.root / f"rendered/{name}/metrics.json"
            data = json.loads(path.read_text())
            if len(data["moments"]) != 35:
                raise AssertionError(f"Incomplete comparison: {name}")
            for row in data["moments"]:
                publications = [
                    r
                    for r in data["publications"]
                    if r["run"] == row["run"] and r["published"] <= row["deadline"]
                ]
                latest = publications[-1] if publications else None
                if latest is None or latest["frame"] + 1 != row["prefix_frame"]:
                    raise AssertionError(f"Incorrect availability at {name}: {row}")
                if row["prefix_frame"] >= row["frame"]:
                    raise AssertionError("Held-out camera RGB entered its own snapshot")
                if latest["capture"] > row["deadline"] or latest["arrival"] > row["deadline"]:
                    raise AssertionError("Future observation")
                audited += 1
                held_out += 1
            data.pop("publications")
            result["variants"][name] = data
            summaries(name, data["moments"])
        if variant != "truth":
            result["inference"][variant] = {}
            for run in ("sweep180", "tour"):
                data = json.loads((args.root / f"depth/{variant}/{run}/inference.json").read_text())
                frames = np.load(args.root / f"prepared/{run}/trajectory.npz")["frames"].tolist()
                if [r["frame"] for r in data["timings"]] != frames:
                    raise AssertionError("Inference admissions differ from frozen inputs")
                data.pop("timings")
                result["inference"][variant][run] = data
    clip_frames = 0
    for trace in (args.root / "clips/metric-small-robot").glob("*.jsonl"):
        for line in trace.read_text().splitlines():
            row = json.loads(line)
            if row["publication"] is not None and row["publication"] > row["deadline"]:
                raise AssertionError("Clip uses unpublished patch")
            clip_frames += 1
    result["clips"] = json.loads((args.root / "clips/metric-small-robot/metrics.json").read_text())
    result["offsets"] = json.loads((args.root / "offsets/metrics.json").read_text())
    for path in sorted(args.root.glob("display-*-*.json")):
        result["gpu"][path.stem] = json.loads(path.read_text())
    result["audit"] = dict(
        prefix_views=audited,
        held_out_camera_views=held_out,
        clip_frames=clip_frames,
        future_observations=0,
        future_publications=0,
    )
    for path in sorted(Path("ito/proto").rglob("*.py")) + sorted(Path("e2e").glob("depthmesh_*")):
        if path.is_file():
            result["code_sha256"][str(path)] = digest(path)
    selected = {
        "mono-sweep.jpg": ("metric-small-robot-cap64", "sweep180-camera.jpg"),
        "mono-tour.jpg": ("metric-small-robot-cap64", "tour-camera.jpg"),
        "oracle-sweep.jpg": ("metric-small-oracle-cap64", "sweep180-camera.jpg"),
        "oracle-tour.jpg": ("metric-small-oracle-cap64", "tour-camera.jpg"),
        "tof-oracle-tour.jpg": ("prompt-small-oracle-cap64", "tour-camera.jpg"),
        "mono-wall-eyes.jpg": ("metric-small-robot-cap64", "tour-eyes.jpg"),
        "hard-turn.jpg": ("metric-small-robot-cap64", "hard-526.jpg"),
    }
    image_folder = args.out / "img"
    image_folder.mkdir(exist_ok=True)
    for name, (variant, image) in selected.items():
        shutil.copy2(args.root / f"rendered/{variant}/{image}", image_folder / name)
    shutil.copy2(args.root / "offsets/wide-offsets.jpg", image_folder / "wide-offsets.jpg")
    pot_folder = args.root / "rendered/metric-small-oracle-cap64"
    sheet(
        [np.asarray(Image.open(pot_folder / f"tour-0363-{eye}.png")) for eye in ("left", "right")],
        [
            f"Near pot {eye} | truth / main / Metric DA Small, oracle pose"
            for eye in ("left", "right")
        ],
        image_folder / "pot-oracle-eyes.jpg",
    )
    for path in image_folder.glob("*.jpg"):
        if path.stat().st_size >= 1_000_000:
            raise AssertionError(f"Oversize sheet: {path}")
    result["sheet_sha256"] = {
        path.name: digest(path) for path in sorted(image_folder.glob("*.jpg"))
    }
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["audit"], indent=2))
    for row in result["summary"]:
        if row["views"] == ["left", "right"]:
            print(
                row["variant"],
                row["run"],
                f"{row['psnr_db']:.2f} / {row['ssim']:.3f} / {row['lpips']:.3f}",
                f"coverage {row['coverage'] * 100:.2f}% "
                f"seen holes {row['previously_seen_holes'] * 100:.2f}%",
            )


if __name__ == "__main__":
    main()
