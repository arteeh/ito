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


def tracks(before, after, matcher):
    if matcher == "klt":
        corners = cv2.goodFeaturesToTrack(before, 1500, 0.01, 7)
        if corners is None:
            return None
        target, status, _ = cv2.calcOpticalFlowPyrLK(before, after, corners, None)
        if target is None:
            return None
        reverse, back, _ = cv2.calcOpticalFlowPyrLK(after, before, target, None)
        initial = target
        attempted = len(corners)
    else:
        sift = cv2.SIFT_create(nfeatures=1500, contrastThreshold=0.01)
        a, da = sift.detectAndCompute(before, None)
        b, db = sift.detectAndCompute(after, None)
        if da is None or db is None or min(len(a), len(b)) < 2:
            return None
        bf = cv2.BFMatcher()
        forward = {
            m.queryIdx: m.trainIdx
            for m, n in bf.knnMatch(da, db, k=2)
            if m.distance < 0.7 * n.distance
        }
        backward = {
            m.queryIdx: m.trainIdx
            for m, n in bf.knnMatch(db, da, k=2)
            if m.distance < 0.7 * n.distance
        }
        pairs = [(i, j) for i, j in forward.items() if backward.get(j) == i]
        if not pairs:
            return (
                np.empty((0, 1, 2), np.float32),
                np.empty((0, 1, 2), np.float32),
                np.empty(0, bool),
                len(a),
            )
        corners = np.array([a[i].pt for i, _ in pairs], np.float32)[:, None]
        initial = np.array([b[j].pt for _, j in pairs], np.float32)[:, None]
        options = dict(flags=cv2.OPTFLOW_USE_INITIAL_FLOW, maxLevel=1)
        target, status, _ = cv2.calcOpticalFlowPyrLK(
            before, after, corners, initial.copy(), **options
        )
        reverse, back, _ = cv2.calcOpticalFlowPyrLK(
            after, before, target, corners.copy(), **options
        )
        attempted = len(a)
    if reverse is None:
        return None
    keep = status[:, 0].astype(bool) & back[:, 0].astype(bool)
    keep &= np.linalg.norm(reverse[:, 0] - corners[:, 0], axis=1) < 0.5
    keep &= np.linalg.norm(target[:, 0] - initial[:, 0], axis=1) < 3
    return corners, target, keep, attempted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("estimates", type=Path)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--truth", action="store_true", help="Diagnose feature tracks against sim truth"
    )
    parser.add_argument("--matcher", choices=("klt", "sift"), default="klt")
    args = parser.parse_args()
    k = json.loads(args.camera.read_text())
    rows = json.loads((args.estimates / "frames.json").read_text())
    previous = None
    errors, pairs = [], []
    attempted = supported = 0
    coverage = []
    for row in rows:
        with np.load(args.estimates / row["file"]) as archive:
            frame = {key: archive[key] for key in ("rgb", "pose", "depth")}
            if args.truth:
                frame.update(pose=archive["true_pose"], depth=archive["true_depth"])
        rgb = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2GRAY)
        coverage.append(float((frame["depth"] > 0).mean()))
        if previous is not None:
            before, image, stamp = previous
            found = tracks(image, rgb, args.matcher)
            if found is not None:
                corners, target, keep, count = found
                attempted += count
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
    report = dict(
        estimates=str(args.estimates),
        geometry="sim truth (diagnostic only)" if args.truth else "estimated",
        matcher=args.matcher,
        attempted_features=attempted,
        supported_tracks=supported,
        supported_fraction=supported / max(attempted, 1),
        exclusions=(
            ("mutual SIFT ratio >=0.7, LK refinement >3px; " if args.matcher == "sift" else "")
            + "KLT status, forward-backward >0.5px, outside image, invalid depth; "
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
