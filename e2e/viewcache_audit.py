"""Audit the actual cached replay artifacts and build the staged decision record."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def average(report, run):
    rows = [
        r
        for r in report["moments"]
        if r["run"] == run and r["frame"] != 526 and r["view"] in ("left", "right")
    ]
    return {
        key: float(np.mean([r[key] for r in rows]))
        for key in ("psnr_db", "ssim", "lpips", "coverage", "previously_seen_holes")
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, default=Path("..").resolve())
    p.add_argument("--root", type=Path, default=Path("../p48-out").resolve())
    args = p.parse_args()
    oracle = json.loads((args.root / "rendered/truth-oracle/metrics.json").read_text())
    clips = json.loads((args.root / "clips/truth-oracle/metrics.json").read_text())
    baseline = json.loads((args.work / "m1-out/verified/repeat-1/metrics.json").read_text())
    mesh = json.loads((args.work / "p47-out/rendered/truth-oracle-cap64/metrics.json").read_text())
    for path, expected in oracle["inputs"].items():
        with (args.work / path).open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != expected:
                raise AssertionError(f"Frozen input changed: {path}")
    snapshots, deadlines, observations = 0, 0, 0
    for row in oracle["moments"]:
        if row["publication"] > row["deadline"] or row["prefix_frame"] >= row["frame"]:
            raise AssertionError("Future snapshot")
        snapshots += 1
    for row in clips["clips"]:
        trace = json.loads(
            (args.root / f"clips/truth-oracle/{row['run']}/{row['clip']}/trace.json").read_text()
        )
        source = np.load(args.work / f"p47-out/prepared/{row['run']}/trajectory.npz")
        captured = dict(zip(source["frames"].tolist(), source["captured"].tolist(), strict=True))
        for r in trace:
            if r["publication"] > r["map_deadline"]:
                raise AssertionError("Future publication")
            if len(r["retained_frames"]) > 64:
                raise AssertionError("Cache cap exceeded")
            if not set(r["selected_frames"]).issubset(r["retained_frames"]):
                raise AssertionError("Selected evicted source")
            for frame in r["retained_frames"]:
                if captured[frame] > r["map_deadline"] or frame + 1 > r["observation_frame"]:
                    raise AssertionError("Future source in clip cache")
                observations += 1
            deadlines += 1
    comparisons = {}
    stage1 = True
    for run in ("sweep180", "tour"):
        main, reference, cache = [average(rep, run) for rep in (baseline, mesh, oracle)]
        gain = cache["psnr_db"] - main["psnr_db"]
        perceptual = 1 - cache["lpips"] / main["lpips"]
        stage1 &= gain >= 2 or perceptual >= 0.2
        comparisons[run] = dict(
            main=main,
            mesh=reference,
            cache=cache,
            psnr_gain_over_main_db=gain,
            lpips_improvement_over_main=perceptual,
            psnr_gain_over_mesh_db=cache["psnr_db"] - reference["psnr_db"],
        )
    # Pool the synchronized per-frame changes across the three already-observed
    # modes. The all-black frozen hard-turn clip cannot earn a stability gain.
    temporal = {}
    for run in ("sweep180", "tour"):
        rows = [
            r
            for r in clips["clips"]
            if r["run"] == run and (run != "sweep180" or r["clip"].startswith("bookshelf-"))
        ]
        temporal[run] = {
            r["clip"]: dict(
                residual_improvement=r["residual_change_improvement"],
                edge_improvement=r["edge_changed_fraction_improvement"],
            )
            for r in rows
        }
    report = dict(
        stage1="PASS" if stage1 else "FAIL",
        comparisons=comparisons,
        temporal=temporal,
        audit=dict(
            snapshot_views=snapshots,
            clip_deadlines=deadlines,
            retained_source_checks=observations,
            hashed_inputs=len(oracle["inputs"]),
            future_observations=0,
            future_publications=0,
        ),
        oracle=oracle,
        clips=clips,
        later_stages="Unassessed until the stage-2 decision; no mono/timing claim",
    )
    (args.root / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: report[k] for k in ("stage1", "comparisons", "temporal", "audit")}, indent=2)
    )


if __name__ == "__main__":
    main()
