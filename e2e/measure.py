"""Freeze inputs and run the measurement baseline without a live robot or visible window.

python e2e/measure.py --work ../ --out ../m1-out/baseline --prepare
See docs/measure.md for the complete one-command run on this machine.
"""

import argparse
import json
import sys
from pathlib import Path

from measure_capture import digest

BASELINE = "316a556"
MOMENTS = {"sweep180": [214, 452, 692, 525], "tour": [362, 1500, 2283]}


def prepare(work, out):
    manifest = dict(
        baseline=BASELINE,
        files={},
        moments=MOMENTS,
        hard_moment="sweep180 frame 526: newly exposed edge during fast turn",
    )
    for run, moments in MOMENTS.items():
        recording = work / f"rf-run/out/{run}/rec"
        rows = [json.loads(s) for s in (recording / "frames.jsonl").read_text().splitlines()]
        start = rows[0]["captured"]
        # Identical admissions for every candidate, chosen without candidate runtimes.
        admissions = [
            dict(frame=i, arrival=r.get("arrival", r["captured"]) - start)
            for i, r in enumerate(rows)
            if i % 6 == 0
        ]
        targets = set(moments)
        if run == "sweep180":
            targets.update(range(495, 556, 6))  # Two seconds around the preselected hard turn.
        snapshots = [dict(frame=i, deadline=rows[i]["captured"] - start) for i in sorted(targets)]
        plan = dict(
            frames_sha256=digest(recording / "frames.jsonl"),
            admissions=admissions,
            snapshots=snapshots,
            arrival_status="legacy: capture used as availability",
            motion_arrival_status="legacy: source-local timestamp used as availability",
        )
        path = out / f"{run}-schedule.json"
        encoded = json.dumps(plan, indent=2) + "\n"
        if path.exists() and path.read_text() != encoded:
            raise ValueError(f"Frozen schedule changed: {path}")
        path.write_text(encoded)
        for path in sorted(recording.iterdir()):
            if path.is_file():
                manifest["files"][str(path.relative_to(work))] = digest(path)
        for path in (work / f"gs/out/{run}/data.npz", work / f"rf-run/out/{run}/truth.jsonl"):
            manifest["files"][str(path.relative_to(work))] = digest(path)
    scene = work / "truth/data/scan/scene.mjb"
    manifest["files"][str(scene.relative_to(work))] = digest(scene)
    path = out / "inputs.json"
    encoded = json.dumps(manifest, indent=2) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise ValueError("Frozen reference input changed")
    path.write_text(encoded)
    print(f"Verified {len(manifest['files'])} frozen input hashes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--ipd", type=float, default=0.064)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    prepare(args.work, args.out)
    if not args.prepare:
        from measure_images import evaluate

        evaluate(args.work, args.out, args.ipd)


if __name__ == "__main__":
    sys.exit(main())
