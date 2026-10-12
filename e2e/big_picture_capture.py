"""Capture exact-main replay maps for offline eye checks; no display or live robot.

Run with the SLAM environment under the GPU lock:
python e2e/big_picture_capture.py --models PATH --at 6,15,23 RECORDING
    --out OUTPUT --pose L1 --truth TRUTH.jsonl
Remaining arguments go to slam_replay. Snapshot I/O affects replay scheduling: these
are visual diagnostics, not timing benchmarks. No model or recording is modified.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ito.reconstruction import mast3r_runtime
from ito.reconstruction.slam import SLAMBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--at", required=True, help="Comma-separated capture offsets in seconds")
    options, remaining = parser.parse_known_args()
    recording = Path(remaining[0])
    out = Path(remaining[remaining.index("--out") + 1])
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(s) for s in (recording / "frames.jsonl").read_text().splitlines()]
    times = np.array([r["captured"] for r in rows])
    targets = sorted(float(s) for s in options.at.split(","))
    mast3r_runtime.MODELS = options.models
    original = SLAMBackend.integrate
    initial = []

    def capture(self, *args, captured, **kwargs):
        original(self, *args, captured=captured, **kwargs)
        if not self.tracked:
            return
        index = int(np.argmin(abs(times - captured)))
        if not initial:
            initial.append((index, self.camera_pose.copy()))
        if not targets or captured - times[0] < targets[0]:
            return
        target = targets.pop(0)
        live = self.keys >= 0
        np.savez_compressed(
            out / f"snapshot-{target:g}.npz",
            records=self.records[live],
            pose=self.camera_pose,
            first_index=initial[0][0],
            first_pose=initial[0][1],
            index=index,
            captured=captured,
            offset=captured - times[0],
        )
        print(f"Saved snapshot {target:g}s at frame {index + 1}", flush=True)

    SLAMBackend.integrate = capture
    import slam_replay

    sys.argv = ["slam_replay.py", *remaining]
    slam_replay.main()


if __name__ == "__main__":
    main()
