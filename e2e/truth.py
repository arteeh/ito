"""Score a recorded drive's camera poses against the simulator's truth.

uv run python e2e/truth.py RECORDING TRUTH.jsonl CAMERA_POSES.jsonl

slam_replay.py scores a replay with it; run alone, it scores the poses a live
e2e/slam_drive.py --record drive showed the pilot (its camera-poses.jsonl).

The virtual Microduck writes one line per rendered head-camera frame when started with
--truth FILE (microduck-sim, drivers/microduck/sim.py): when it was taken on the driver's
clock, the camera's true pose and a thumbnail. A recording from e2e/slam_drive.py --record
carries every robot Motion sample on both clocks, so a recorded frame's driver capture time
is known; its exposure is the rendered frame, among those up to a quarter second before it,
whose thumbnail it matches best.

Scores use a fixed alignment at the first frame (heading and position only; both worlds
are gravity-aligned), never a fit: the robot's world is supposed to be the room's.
"""

import base64
import json
import math

import numpy as np

THUMB = 24  # drivers/microduck/sim_body.py
SEARCH = (-0.25, 0.02)  # exposures this far around the frame's capture time, seconds
USABLE = (math.radians(2), 0.05)  # a usable pose: within 2 degrees and 5 cm


def load(path):
    """(times, 4x4 world-from-camera poses, thumbnails) of every rendered frame."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    times = np.array([r["t"] for r in rows])
    poses = np.array([np.array(r["camera"]).reshape(4, 4) for r in rows])
    thumbs = np.array(
        [np.frombuffer(base64.b64decode(r["thumb"]), np.uint8).reshape(THUMB, THUMB) for r in rows],
        dtype=np.float32,
    )
    order = np.argsort(times)
    return times[order], poses[order], thumbs[order]


def thumbnail(rgb):
    height, width = rgb.shape[:2]
    rows = ((np.arange(THUMB) + 0.5) * height / THUMB).astype(int)
    columns = ((np.arange(THUMB) + 0.5) * width / THUMB).astype(int)
    return rgb[rows][:, columns].mean(axis=2).astype(np.float32)


def clock_offset(motion):
    """Pilot minus driver clock, from motion samples recorded on both."""
    return float(np.median([m["local"] - m["time"] for m in motion]))


class Truth:
    def __init__(self, path, motion):
        self.times, self.poses, self.thumbs = load(path)
        self.offset = clock_offset(motion)
        self.turns = None  # quarter turns between rendered and received images

    def _candidates(self, captured):
        driver = captured - self.offset
        lo, hi = np.searchsorted(self.times, (driver + SEARCH[0], driver + SEARCH[1]))
        return np.arange(lo, hi)

    def _errors(self, image, candidates, turns):
        rotated = np.rot90(self.thumbs[candidates], turns, axes=(1, 2))
        return np.mean((rotated - image[None]) ** 2, axis=(1, 2))

    def match(self, rgb, captured):
        """(exposure time on the driver clock, true pose, error) of a received frame."""
        image = thumbnail(rgb)
        candidates = self._candidates(captured)
        if not len(candidates):
            return None
        if self.turns is None:
            self.turns = min(range(4), key=lambda k: self._errors(image, candidates, k).min())
        errors = self._errors(image, candidates, self.turns)
        best = candidates[int(np.argmin(errors))]
        return float(self.times[best]), self.poses[best], float(errors.min())


def rotation_angle(rotation):
    return math.acos(max(-1.0, min(1.0, (np.trace(rotation) - 1) / 2)))


def heading(pose):
    """Yaw of a world-from-camera pose about the vertical, positive to the left."""
    return math.atan2(pose[0, 2], pose[2, 2])


def align(estimate, truth):
    """World-from-truth-world: the first truth pose onto the first estimate, by heading and
    position only."""
    yaw = heading(estimate) - heading(truth)
    c, s = math.cos(yaw), math.sin(yaw)
    result = np.eye(4)
    result[:3, :3] = ((c, 0, s), (0, 1, 0), (-s, 0, c))
    result[:3, 3] = estimate[:3, 3] - result[:3, :3] @ truth[:3, 3]
    return result


def score(estimates, truths):
    """Rotation and translation errors of world-from-camera estimates against truth."""
    estimates, truths = np.asarray(estimates), np.asarray(truths)
    if not len(estimates):
        return {}
    alignment = align(estimates[0], truths[0])
    aligned = alignment @ truths
    rotation = np.degrees(
        [rotation_angle(e[:3, :3].T @ t[:3, :3]) for e, t in zip(estimates, aligned, strict=True)]
    )
    translation = np.linalg.norm(estimates[:, :3, 3] - aligned[:, :3, 3], axis=1)
    path = float(np.linalg.norm(np.diff(aligned[:, :3, 3], axis=0), axis=1).sum())
    usable = (np.radians(rotation) < USABLE[0]) & (translation < USABLE[1])

    def stats(values, digits):
        return dict(
            p50=round(float(np.median(values)), digits),
            p95=round(float(np.percentile(values, 95)), digits),
            max=round(float(np.max(values)), digits),
        )

    return dict(
        frames=len(estimates),
        rotation_deg=stats(rotation, 2),
        translation_m=stats(translation, 3),
        camera_path_m=round(path, 2),
        drift_per_m=round(float(translation[-1]) / path, 3) if path > 0.2 else None,
        end_error_m=round(float(translation[-1]), 3),
        usable_fraction=round(float(usable.mean()), 3),
    )


def main():
    import argparse
    from pathlib import Path

    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("truth", type=Path)
    parser.add_argument("poses", type=Path)
    args = parser.parse_args()
    lines = (args.recording / "motion.jsonl").read_text().splitlines()
    truth = Truth(args.truth, [json.loads(line) for line in lines if line.strip()])
    frames = {}
    for line in (args.recording / "frames.jsonl").read_text().splitlines():
        row = json.loads(line)
        frames[row["captured"]] = row["name"]
    estimates, truths, missing = [], [], 0
    for line in args.poses.read_text().splitlines():
        row = json.loads(line)
        name = frames.get(row["captured"])
        if name is None:
            missing += 1
            continue
        rgb = np.asarray(Image.open(args.recording / name).convert("RGB"))
        found = truth.match(rgb, row["captured"])
        if found is not None:
            estimates.append(np.array(row["camera"]).reshape(4, 4))
            truths.append(found[1])
    print(json.dumps(dict(score(estimates, truths), unrecorded=missing), indent=2))


if __name__ == "__main__":
    main()
