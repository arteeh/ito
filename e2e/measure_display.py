"""Time 1,000 stereo frames through production GL under ceres.lock, without a window.

python e2e/measure_display.py REPEAT_DIRECTORY --work WORK
Completion is a GPU fence, not headset scanout. No inference runs concurrently here.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from measure_gl import View


def main():
    from truth import align

    from ito.protocol import Intrinsics

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--ipd", type=float, default=0.064)
    args = parser.parse_args()
    report = {"scope": "isolated offscreen stereo GL completion; no inference or compositor"}
    for run, index in (("sweep180", 692), ("tour", 2283)):
        intrinsics = Intrinsics.model_validate_json(
            (args.work / f"rf-run/out/{run}/rec/camera.json").read_text()
        )
        snap = np.load(args.output / run / f"snapshot-{index:06d}.npz")
        truths = np.load(args.work / f"gs/out/{run}/data.npz")["poses"]
        world = align(snap["first_pose"], truths[int(snap["first_index"])])
        camera = np.asarray(world @ truths[index], np.float32)
        view = View(intrinsics)
        view.upload(snap["records"])
        durations, longest, streak = [], 0, 0
        with view.context:
            right_target = view.context.simple_framebuffer(view.size, components=4, dtype="f4")
            for n in range(args.frames + 10):
                began = time.perf_counter()
                head = np.eye(4, dtype=np.float32)
                head[0, 3] = 0.03 * np.sin(n / 90 * 2 * np.pi)
                view.renderer.sort(camera, head)
                for side, target in ((-1, view.target), (1, right_target)):
                    eye = head.copy()
                    eye[0, 3] += side * args.ipd / 2
                    view.renderer.draw(
                        camera, eye, view.projection, target, clear=(0, 0, 0, 0), sort=False
                    )
                view.context.finish()
                if n >= 10:
                    duration = (time.perf_counter() - began) * 1000
                    durations.append(duration)
                    streak = streak + 1 if duration > 1000 / 90 else 0
                    longest = max(longest, streak)
            right_target.release()
        report[run] = dict(
            renderer=view.device,
            frames=args.frames,
            ipd_m=args.ipd,
            occupied=view.renderer.count,
            sort_capacity=view.renderer.capacity,
            uploaded_bytes=view.renderer.uploaded_bytes,
            upload_ms=view.upload_ms,
            stereo_ms={f"p{p:g}": float(np.percentile(durations, p)) for p in (50, 95, 99, 99.9)},
            missed_fraction=float(np.mean(np.array(durations) > 1000 / 90)),
            longest_miss_run=longest,
            deadline_ms=1000 / 90,
        )
        np.save(args.output / f"{run}-stereo-ms.npy", durations)
        view.close()
    (args.output / "display.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
