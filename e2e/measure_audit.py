"""Audit actual replay artifacts, then summarize repeated baseline measurements."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    reports, prefixes, deadlines = [], 0, 0
    for repeat in range(1, 4):
        folder = args.output / f"repeat-{repeat}"
        metrics = json.loads((folder / "metrics.json").read_text())
        summary = {"repeat": repeat, "sequences": {}}
        for run in ("sweep180", "tour"):
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
    report = dict(prefixes_checked=prefixes, pose_deadlines_checked=deadlines, repeats=reports)
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {prefixes} causal prefixes, {deadlines} pose deadlines, identical admissions")


if __name__ == "__main__":
    main()
