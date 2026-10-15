"""Compare causal estimator outputs to sim truth; numbers locate cells to review.

uv run python e2e/estimator_report.py ESTIMATES [ESTIMATES ...] --out report.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def report(root):
    rows = json.loads((root / "frames.json").read_text())
    ratios, absolute, spread, rotation, translation = [], [], [], [], []
    alignment = None
    for row in rows:
        data = np.load(root / row["file"])
        estimated, truth = data["pose"].astype(float), data["true_pose"].astype(float)
        if alignment is None:
            alignment = estimated @ np.linalg.inv(truth)
        truth = alignment @ truth
        rotation.append(
            float(np.degrees(Rotation.from_matrix(truth[:3, :3].T @ estimated[:3, :3]).magnitude()))
        )
        translation.append(float(np.linalg.norm(estimated[:3, 3] - truth[:3, 3])))
        z, t = data["depth"], data["true_depth"]
        valid = (z > 0) & (t > 0) & np.isfinite(z) & np.isfinite(t)
        r = z[valid] / t[valid]
        ratios.append(float(np.median(r)))
        absolute.append(float(np.median(np.abs(z[valid] - t[valid]))))
        spread.append(np.percentile(r, [10, 90]))

    def stats(values):
        return dict(
            zip(
                ("min", "median", "p95", "max"),
                np.percentile(values, [0, 50, 95, 100]).tolist(),
                strict=True,
            )
        )

    result = dict(
        root=str(root),
        frames=len(rows),
        rotation_deg=stats(rotation),
        translation_m=stats(translation),
        depth_ratio=stats(ratios),
        depth_absolute_m=stats(absolute),
        within_frame_ratio_p10_p90=np.median(spread, axis=0).tolist(),
        surface_hz=len(rows) / (rows[-1]["available"] - rows[0]["started"]),
        surface_delay_ms=stats([(r["available"] - r["captured"]) * 1000 for r in rows]),
    )
    if "scale" in rows[0]:
        result.update(
            scale=stats([r["scale"] for r in rows]),
            scale_observations=rows[-1]["scale_observations"],
            rejected_visual=rows[-1]["rejected_visual"],
            delay_ms=stats([r["delay"] * 1000 for r in rows]),
        )
        for i, name in [(0, "propagated_pose_ms"), (-1, "corrected_pose_ms")]:
            values = [
                r["pose_publications"][i]["seconds"] * 1000
                for r in rows
                if r.get("pose_publications")
            ]
            if values:
                result[name] = stats(values)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("estimates", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = [report(root) for root in args.estimates]
    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
