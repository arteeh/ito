"""Summarize audited joint replays, without turning image quality into a number.

PYTHONPATH=. python e2e/oracle_joint_manifest.py OUT --estimator COMMIT \
    --extraction-estimator COMMIT --images DIR --output JSON
OUT contains before/, final/{after,paired}/; run oracle_audit.py on final/after first.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def read(path):
    return json.loads(path.read_text())


def workers(root):
    result = {}
    for path in sorted((root / "visual").glob("*/worker.json")):
        work = read(path)
        display = read(path.parent / "display.json")
        iterations = [work["view_iterations"].get(str(n), 0) for n in work.get("keyframes", [])]
        result[path.parent.name] = dict(
            settings=display["settings"],
            camera_mode=display["camera_mode"],
            training_size=work["training_size"],
            observations=len(work["admitted"]),
            steps=work["steps"],
            elapsed=work["elapsed"],
            steps_per_second=work["steps"] / work["elapsed"],
            image_iterations_per_second=work.get("image_iterations", 0) / work["elapsed"],
            keyframe_iterations=iterations,
            keyframes_at_least_100=sum(n >= 100 for n in iterations),
            keyframe_median=float(np.median(iterations)) if iterations else None,
            oom=work["oom"],
            torch_peak_mib=work["torch_peak_reserved"] / 2**20,
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--estimator", required=True)
    p.add_argument("--extraction-estimator", required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    final = args.out / "final"
    audit = read(final / "after/audit.json")
    assert not subprocess.check_output(
        [
            "git",
            "diff",
            "--name-only",
            args.extraction_estimator,
            args.estimator,
            "--",
            "ito/reconstruction",
            "e2e/oracle_estimate.py",
        ],
        text=True,
    ), "Estimator inference changed; freeze fresh inputs before the final comparison"
    main_native = {}
    for path in sorted((final / "after/native-main9a").glob("*/display.json")):
        row = read(path)
        assert "NVIDIA" in row["renderer"]
        assert not row["optimizer"] and row["render_budget"] is None
        assert row["settings"] == dict(precision="full", input_resolution=100, max_splats=1_000_000)
        main_native[path.parent.name] = row
    images = {}
    for path in sorted(args.images.iterdir()):
        with Image.open(path) as pic:
            images[path.name] = dict(
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bytes=path.stat().st_size,
                size=list(pic.size),
            )
    result = dict(
        estimator_commit=args.estimator,
        frozen_extraction_estimator=args.extraction_estimator,
        estimator_equivalence="No changes to reconstruction or oracle_estimate.py between "
        "frozen extraction and latest merged estimator commits (git diff verified)",
        settings=dict(precision="full", input_resolution=100, max_splats=1_000_000),
        camera="velstand v8 / real62deg long-side / 360p30 upright360x640; fit288x512",
        render_cap=98_304,
        quality="By-eye motion-sheet verdicts in gsplat-oracle.md; no numerical quality gate",
        native_scope="Frozen mature map, two2160x2160 eyes, 90deg, 64mm IPD, moving head, "
        "two4096-record uploads/frame. Real cold-prefix fitter load; publication drained. "
        "No estimator, live changing map, OpenXR/compositor or scanout.",
        inputs=audit["inputs"],
        before=workers(args.out / "before"),
        paired_round2_same_inputs=workers(final / "paired"),
        round3=workers(final / "after"),
        memory={
            name: {key: value for key, value in row.items() if key.endswith("mib")}
            for name, row in audit["visual"].items()
        },
        native=audit["native"],
        actual_main_archived_map_native=main_native,
        half_fault=audit.get("half_fault"),
        half_refinement_fault=audit.get("half_refinement_fault"),
        causal_format_audit="passed; exact payload/schedule hashes, completed prefixes, "
        "no future observation/motion/visual correction, finite native records",
        images=images,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
