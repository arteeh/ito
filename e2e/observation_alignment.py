"""Inspect recorded real daemon observations against sim truth (evaluation only).

PYTHONPATH=. .venv/bin/python e2e/observation_alignment.py RECORDING --out report.json
No truth enters the estimator. The first inertial camera pose fixes one arbitrary world yaw.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ito.protocol import SensorObservations
from ito.reconstruction.sensors import Sensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.recording / "source"
    frames = [json.loads(row) for row in (source / "frames.jsonl").read_text().splitlines()]
    sensors = Sensors()
    for row in (args.recording / "received/sensors.jsonl").read_text().splitlines():
        sensors.extend(
            SensorObservations.model_validate_json(json.dumps(json.loads(row)["observation"]))
        )
    k = json.loads((source / "camera.json").read_text())
    origin = frames[0]["captured"]
    alignment = None
    rotations, times, range_errors, sample_periods = [], [], [], []
    for a, b in zip(sensors.imu, list(sensors.imu)[1:]):
        sample_periods.append(b[0] - a[0])
    for row in frames:
        stamp = row["captured"]
        if not sensors.covers(stamp, slack=0):
            continue
        data = np.load(source / row["file"])
        rotation = sensors.at(stamp)[0]
        truth = Rotation.from_matrix(data["camera"][:3, :3])
        if alignment is None:
            alignment = rotation * truth.inv()
        rotations.append(float(np.degrees(((alignment * truth).inv() * rotation).magnitude())))
        times.append(stamp - origin)
        if not sensors.ranges:
            continue
        sample = min(sensors.ranges, key=lambda s: abs(s.time - stamp))
        if abs(sample.time - stamp) > 0.04:
            continue
        mount = Rotation.from_quat(sample.mount.orientation)
        for zone in sample.zones:
            if zone.quality != "valid":
                continue
            point = mount.apply(np.array(zone.direction) * zone.distance) + sample.mount.position
            if point[2] >= -0.02:
                continue
            u = round(k["cx"] + k["fx"] * point[0] / -point[2])
            v = round(k["cy"] - k["fy"] * point[1] / -point[2])
            if not (3 <= u < k["width"] - 3 and 3 <= v < k["height"] - 3):
                continue
            patch = data["depth"][v - 3 : v + 4, u - 3 : u + 4]
            if np.isfinite(patch).all() and patch.min() > 0 and patch.max() - patch.min() < 0.05:
                range_errors.append(abs(float(np.median(patch)) + point[2]))

    def stats(values):
        return (
            dict(
                zip(
                    ("median", "p95", "max"),
                    np.percentile(values, [50, 95, 100]).tolist(),
                    strict=True,
                )
            )
            if values
            else None
        )

    report = dict(
        recording=str(args.recording),
        samples=len(sensors.imu),
        views=len(rotations),
        sample_period_s=stats(sample_periods),
        raw_inertial_rotation_deg=stats(rotations),
        range_center_axial_error_m=stats(range_errors),
        range_comparisons=len(range_errors),
        evaluation="Source truth only for this diagnostic; first-pose rigid alignment; no scale fit; range centers exclude depth edges. Numbers locate visual review, not quality gates.",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
