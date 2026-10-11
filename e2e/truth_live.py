"""Approximate transport lag by matching live thumbnails to source RGB on one host clock.

python e2e/truth_live.py DATASET
This measures both video hops, not mediad alone. Ambiguous near-static matches are excluded.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def stats(values):
    return (
        dict(
            median=float(np.median(values)),
            p95=float(np.percentile(values, 95)),
            min=float(np.min(values)),
            max=float(np.max(values)),
        )
        if len(values)
        else None
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    root = args.dataset
    rows = [json.loads(s) for s in (root / "frames.jsonl").read_text().splitlines()]
    live = [json.loads(s) for s in (root / "live.jsonl").read_text().splitlines()]
    times = np.array([r["monotonic_ns"] for r in rows], dtype=np.int64) / 1e9
    received = np.array([r["monotonic_ns"] for r in live], dtype=np.int64) / 1e9
    size = Image.open(root / live[0]["rgb"]).size
    # Determine the transport's quarter-turn, rather than assuming its configured mount.
    scores, rotated = [], []
    images = [np.asarray(Image.open(root / r["rgb"])) for r in rows]
    for turns in range(4):
        sources = np.array(
            [
                np.asarray(
                    Image.fromarray(np.rot90(image, turns)).resize(size), dtype=np.float32
                ).ravel()
                for image in images
            ]
        )
        errors = []
        for i in np.linspace(0, len(live) - 1, 20, dtype=int):
            candidates = np.flatnonzero((times <= received[i]) & (times >= received[i] - 2))
            if len(candidates):
                image = np.asarray(Image.open(root / live[i]["rgb"]), dtype=np.float32).ravel()
                errors.append(float(np.min(np.mean((sources[candidates] - image) ** 2, axis=1))))
        scores.append(float(np.median(errors)))
        rotated.append(sources)
    turns = int(np.argmin(scores))
    sources = rotated[turns]
    del images, rotated
    cameras = np.array([r["camera"] for r in rows])
    matches = []
    for i, row in enumerate(live):
        candidates = np.flatnonzero((times <= received[i]) & (times >= received[i] - 2))
        if len(candidates) < 2:
            continue
        image = np.asarray(Image.open(root / row["rgb"]), dtype=np.float32).ravel()
        errors = np.sqrt(np.mean((sources[candidates] - image) ** 2, axis=1))
        order = np.argsort(errors)
        matched = candidates[order[0]]
        joined = int(np.argmin(abs(times - received[i])))
        angle = Rotation.from_matrix(
            cameras[matched, :3, :3].T @ cameras[joined, :3, :3]
        ).magnitude()
        confidence = bool(errors[order[0]] < 12 and errors[order[1]] - errors[order[0]] > 0.15)
        matches.append(
            dict(
                live=i,
                source=int(matched),
                receive_join=int(joined),
                lag_ms=float((received[i] - times[matched]) * 1000),
                rgb_rmse=float(errors[order[0]]),
                runnerup_gap=float(errors[order[1]] - errors[order[0]]),
                confident=confidence,
                wrong_frame=bool(matched != joined),
                position_error_m=float(
                    np.linalg.norm(cameras[matched, :3, 3] - cameras[joined, :3, 3])
                ),
                rotation_error_deg=float(np.degrees(angle)),
            )
        )
    confident = [m for m in matches if m["confident"]]
    report = dict(
        scope="APPROXIMATE image-matched source to pilot (mediad plus Ito WebRTC hops)",
        method="Closest source thumbnail in preceding 2 seconds; same distro monotonic clock. "
        "Confident requires RGB RMSE <12/255 and next-best RMSE gap >0.15/255. "
        "Receive-time join uses nearest source pose; does not include robot FK/IMU error.",
        rotation_ccw_quarters=turns,
        rotation_match_mse=scores,
        live_frames=len(live),
        source_frames=len(rows),
        matched=len(matches),
        confident=len(confident),
        ambiguous_fraction=1 - len(confident) / max(1, len(matches)),
        lag_ms=stats([m["lag_ms"] for m in confident]),
        receive_join_position_m=stats([m["position_error_m"] for m in confident]),
        receive_join_rotation_deg=stats([m["rotation_error_deg"] for m in confident]),
        wrong_frame_fraction=float(np.mean([m["wrong_frame"] for m in confident]))
        if confident
        else None,
        materially_wrong_pose_fraction=float(
            np.mean(
                [m["position_error_m"] > 0.01 or m["rotation_error_deg"] > 1 for m in confident]
            )
        )
        if confident
        else None,
    )
    (root / "live-matches.json").write_text(json.dumps(matches))
    (root / "live-summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
