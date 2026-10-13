"""Audit actual replay artifacts, then summarize repeated baseline measurements."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    reports, prefixes, deadlines, clip_frames = [], 0, 0, 0
    source = json.loads((args.output / "source/summary.json").read_text())
    assert source["capture"]["exact_fraction"] == 1
    assert source["calibration"]["pass"]
    for repeat in range(1, 4):
        folder = args.output / f"repeat-{repeat}"
        metrics = json.loads((folder / "metrics.json").read_text())
        for clip in metrics["clips"]:
            trace = folder / Path(clip["file"]).with_suffix(".jsonl")
            rows = [json.loads(s) for s in trace.read_text().splitlines()]
            assert len(rows) == 61
            for row in rows:
                assert row["publication"] <= row["deadline"]
                assert row["newest_map_observation_age_s"] >= 0
                assert row["head_sample"] <= row["display_complete"]
                clip_frames += 1
        display = json.loads((folder / "display.json").read_text())
        summary = {"repeat": repeat, "sequences": {}}
        for run in ("sweep180", "tour"):
            durations = np.load(folder / f"{run}-stereo-ms.npy")
            assert len(durations) == display[run]["frames"] == 1000
            assert np.all(np.isfinite(durations)) and np.all(durations > 0)
            plan = json.loads((folder / f"{run}-schedule.json").read_text())
            publications = [
                json.loads(s)
                for s in (folder / run / "publications.jsonl").read_text().splitlines()
            ]
            assert [p["frame"] for p in publications] == [a["frame"] for a in plan["admissions"]]
            for p in publications:
                assert p["arrival"] <= p["started"] <= p["publication"]
                assert p["max_motion_arrival"] <= p["started"] + 1e-7
            for target in plan["snapshots"]:
                with np.load(folder / run / f"snapshot-{target['frame']:06d}.npz") as snap:
                    assert float(snap["published"]) <= target["deadline"]
                    assert np.all(snap["observation_indices"] <= target["frame"])
                    assert float(snap["max_observation_arrival"]) <= target["deadline"]
                    prefixes += 1
            for line in (folder / run / "poses.jsonl").read_text().splitlines():
                row = json.loads(line)
                if row["published"] is not None:
                    assert row["published"] <= row["deadline"]
                deadlines += 1
            rows = [r for r in metrics["moments"] if r["run"] == run and r["view"] == "camera"]
            summary["sequences"][run] = dict(
                rates=json.loads((folder / run / "summary.json").read_text()),
                moments=[
                    {
                        k: r[k]
                        for k in (
                            "frame",
                            "prefix_frame",
                            "psnr_db",
                            "ssim",
                            "lpips",
                            "coverage",
                            "previously_seen_holes",
                        )
                    }
                    for r in rows
                ],
                draw=metrics[f"{run}_draw"],
            )
        reports.append(summary)
    report = dict(
        prefixes_checked=prefixes,
        pose_deadlines_checked=deadlines,
        clip_frames_checked=clip_frames,
        repeats=reports,
        evaluation_scripts_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("measure*"))
            if p.is_file()
        },
    )
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"PASS: {prefixes} causal prefixes, {deadlines} pose deadlines, "
        f"{clip_frames} clip frames, identical admissions, exact source IDs and calibration"
    )


if __name__ == "__main__":
    main()
