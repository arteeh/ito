"""Reuse a completed capture only when its code, schedule and causal artifacts agree."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder, schedule, recording = map(Path, sys.argv[1:])
    try:
        summary = json.loads((folder / "summary.json").read_text())
        assert summary["capture_script_sha256"] == digest(
            Path(__file__).with_name("measure_capture.py")
        )
        assert summary["schedule_sha256"] == digest(schedule)
        plan = json.loads(schedule.read_text())
        publications = [
            json.loads(s) for s in (folder / "publications.jsonl").read_text().splitlines()
        ]
        assert [p["frame"] for p in publications] == [a["frame"] for a in plan["admissions"]]
        for p in publications:
            assert p["arrival"] <= p["started"] <= p["publication"]
            assert p["max_motion_arrival"] <= p["started"] + 1e-7
        poses = [json.loads(s) for s in (folder / "poses.jsonl").read_text().splitlines()]
        assert [p["frame"] for p in poses] == list(
            range(len((recording / "frames.jsonl").read_text().splitlines()))
        )
        for p in poses:
            assert p["published"] is None or p["published"] <= p["deadline"]
        for target in plan["snapshots"]:
            with np.load(folder / f"snapshot-{target['frame']:06d}.npz") as snap:
                assert float(snap["published"]) <= target["deadline"]
                assert np.all(snap["observation_indices"] <= target["frame"])
                assert float(snap["max_observation_arrival"]) <= target["deadline"]
                assert len(snap["records"]) == len(snap["last_observed_capture"])
    except (OSError, ValueError, KeyError, AssertionError):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
