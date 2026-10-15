"""Materialize a labelled frozen DPVO rotation intervention for oracle EE/TE replay.

.venv/bin/python e2e/estimator_dpvo_arms.py BASE DPVO --out OUT --mode fused
Independent compute timings are not concurrent live tracker/mapper measurements.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("dpvo", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("raw", "fused"), default="fused")
    args = parser.parse_args()
    rows = json.loads((args.base / "frames.json").read_text())
    rotations = json.loads((args.dpvo / "frames.json").read_text())
    availability = np.array([r["available"] for r in rotations])
    if np.any(np.diff(availability) < 0):
        raise ValueError("DPVO snapshots must be ordered by availability")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for row in rows:
        with np.load(args.base / row["file"]) as archive:
            data = dict(archive)
        # The correction must exist before this depth frame starts processing.
        index = int(np.searchsorted(availability, row["started"], side="right")) - 1
        used = index >= 0 and rotations[index]["used_native_rotation"]
        if used:
            observed = rotations[index]
            with np.load(args.dpvo / observed["file"]) as archive:
                pose = archive["pose" if args.mode == "raw" else "fused_pose"]
                prior = archive["motion_pose"]
            current_prior = np.asarray(row["motion_pose"])
            data["pose"] = data["pose"].copy()
            data["pose"][:3, :3] = pose[:3, :3] @ prior[:3, :3].T @ current_prior[:3, :3]
            if not np.isfinite(data["pose"]).all():
                raise ValueError("DPVO produced a nonfinite rotation")
        np.savez(args.out / row["file"], **data)
        manifest.append(
            dict(
                row,
                dpvo_mode=args.mode,
                dpvo_used=bool(used),
                dpvo_available=None if not used else rotations[index]["available"],
                dpvo_captured=None if not used else rotations[index]["captured"],
                graph_pose=data["pose"].tolist(),
            )
        )
    settings = json.loads((args.base / "settings.json").read_text())
    settings.update(
        estimator=f"frozen DPVO {args.mode} rotation; MASt3R metric translation and depth",
        intervention="causal snapshots with isolated timings; NOT co-scheduled live throughput",
        dpvo_summary=json.loads((args.dpvo / "summary.json").read_text()),
        base_schedule_sha256=hashlib.sha256((args.base / "frames.json").read_bytes()).hexdigest(),
        dpvo_schedule_sha256=hashlib.sha256((args.dpvo / "frames.json").read_bytes()).hexdigest(),
        intervention_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        depth_unchanged=True,
        translation_unchanged=True,
    )
    (args.out / "frames.json").write_text(json.dumps(manifest, indent=2))
    (args.out / "settings.json").write_text(json.dumps(settings, indent=2))
    print(f"{sum(r['dpvo_used'] for r in manifest)}/{len(manifest)} frames use DPVO {args.mode}")


if __name__ == "__main__":
    main()
