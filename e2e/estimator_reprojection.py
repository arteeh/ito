"""Cross-view pixel error on bidirectional feature tracks, with exclusions recorded.

.venv/bin/python e2e/estimator_reprojection.py ESTIMATES --camera camera.json --out report.json
No residual-based pruning: every bidirectional, in-bounds, positive-depth track is scored.
Feature support is disclosed separately from dense image coverage.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("estimates", type=Path)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    k = json.loads(args.camera.read_text())
    rows = json.loads((args.estimates / "frames.json").read_text())
    previous = None
    errors, pairs = [], []
    attempted = supported = 0
    coverage = []
    for row in rows:
        frame = np.load(args.estimates / row["file"])
        rgb = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2GRAY)
        coverage.append(float((frame["depth"] > 0).mean()))
        if previous is not None:
            before, image, stamp = previous
            corners = cv2.goodFeaturesToTrack(image, 1500, 0.01, 7)
            if corners is not None:
                attempted += len(corners)
                target, good, _ = cv2.calcOpticalFlowPyrLK(image, rgb, corners, None)
                if target is not None:
                    reverse, back, _ = cv2.calcOpticalFlowPyrLK(rgb, image, target, None)
                    if reverse is not None:
                        keep = good[:, 0].astype(bool) & back[:, 0].astype(bool)
                        keep &= np.linalg.norm(reverse[:, 0] - corners[:, 0], axis=1) < 0.5
                        uv = corners[:, 0]
                        xy = np.rint(uv).astype(int)
                        depth = before["depth"][xy[:, 1], xy[:, 0]]
                        xyz = (
                            np.column_stack(
                                (
                                    (uv[:, 0] - k["cx"]) / k["fx"],
                                    -(uv[:, 1] - k["cy"]) / k["fy"],
                                    -np.ones(len(uv)),
                                )
                            )
                            * depth[:, None]
                        )
                        transform = np.linalg.inv(frame["pose"].astype(float)) @ before["pose"]
                        current = xyz @ transform[:3, :3].T + transform[:3, 3]
                        axial = -current[:, 2]
                        projected = np.column_stack(
                            (
                                k["fx"] * current[:, 0] / np.maximum(axial, 1e-8) + k["cx"],
                                k["cy"] - k["fy"] * current[:, 1] / np.maximum(axial, 1e-8),
                            )
                        )
                        keep &= (depth > 0) & (axial > 0) & np.isfinite(projected).all(axis=1)
                        keep &= (
                            (target[:, 0, 0] >= 0)
                            & (target[:, 0, 0] < k["width"])
                            & (target[:, 0, 1] >= 0)
                            & (target[:, 0, 1] < k["height"])
                        )
                        residual = np.linalg.norm(projected[keep] - target[keep, 0], axis=1)
                        errors.extend(residual.tolist())
                        supported += int(keep.sum())
                        if len(residual):
                            pairs.append(
                                dict(
                                    before=stamp,
                                    after=row["captured"],
                                    tracks=len(residual),
                                    pixel_p95=float(np.percentile(residual, 95)),
                                )
                            )
        # The arrays outlive the zip reader explicitly, rather than retaining an open file.
        previous = ({key: frame[key] for key in ("pose", "depth")}, rgb, row["captured"])
        frame.close()
    report = dict(
        estimates=str(args.estimates),
        attempted_features=attempted,
        supported_tracks=supported,
        supported_fraction=supported / max(attempted, 1),
        exclusions=(
            "KLT status, forward-backward >0.5px, outside image, invalid depth; "
            "no reprojection-error cut"
        ),
        dense_depth_coverage_median=float(np.median(coverage)),
        pixel_error=dict(
            zip(
                ("median", "p95", "max"), np.percentile(errors, [50, 95, 100]).tolist(), strict=True
            )
        ),
        pairs=pairs,
    )
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
