"""Measure sim exposure-to-receipt delay by matching decoded RGB to exact source RGB.

.venv/bin/python e2e/camera_timing.py RECORDING --out timing.json
Ambiguous/static images are excluded explicitly; no guessed source identity is assigned.
This calibrates the simulated mediad pipeline, not the physical sensor or rolling shutter.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    def rows(path):
        return [json.loads(row) for row in path.read_text().splitlines()]

    sources = rows(args.recording / "source/frames.jsonl")
    received = rows(args.recording / "received/frames.jsonl")
    times = np.array([row["captured"] for row in sources])
    images = np.stack(
        [
            cv2.resize(np.load(args.recording / "source" / row["file"])["rgb"], (90, 160)).astype(
                np.float32
            )
            for row in sources
        ]
    )
    matches = []
    for row in received:
        image = cv2.cvtColor(
            cv2.imread(str(args.recording / "received" / row["name"])), cv2.COLOR_BGR2RGB
        )
        image = cv2.resize(image, (90, 160)).astype(np.float32)
        lo, hi = np.searchsorted(times, [row["captured"] - 0.5, row["captured"] + 0.05])
        if hi - lo < 2:
            continue
        costs = np.mean(np.abs(images[lo:hi] - image), axis=(1, 2, 3))
        best, runner = np.argsort(costs)[:2]
        # Only a distinct best RGB match supports a timing observation. Static frames
        # provide no delay information even when their absolute appearance matches.
        if costs[best] > 0.8 * costs[runner]:
            continue
        source = sources[lo + best]
        matches.append(
            dict(
                source_frame_id=source["source_frame_id"],
                transport_sequence=row["transport_sequence"],
                captured=source["captured"],
                receipt_delay_ms=1000 * (row["captured"] - source["captured"]),
                delivered_delay_ms=1000 * (row["arrival"] - source["captured"]),
                match_mean_absolute_rgb=float(costs[best]),
            )
        )

    if not matches:
        raise SystemExit("No distinct RGB identities: delay is unobservable in these views")

    def stats(key):
        values = np.array([r[key] for r in matches])
        return dict(
            zip(
                ("p5", "median", "p95", "max"),
                np.percentile(values, [5, 50, 95, 100]).tolist(),
                strict=True,
            )
        ) | dict(jitter_mad_ms=float(1.4826 * np.median(np.abs(values - np.median(values)))))

    result = dict(
        recording=str(args.recording),
        received=len(received),
        supported=len(matches),
        excluded=len(received) - len(matches),
        exclusion="ambiguous RGB identity (best not 20% better than runner-up) or no candidates",
        receipt_delay_ms=stats("receipt_delay_ms"),
        delivered_delay_ms=stats("delivered_delay_ms"),
        matches=matches,
    )
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "matches"}, indent=2))


if __name__ == "__main__":
    main()
