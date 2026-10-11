"""Replay every exact source RGB through the unmodified shipped CUDA SLAM backend.

python e2e/truth_replay.py DATASET --out OUTPUT
Uses monocular (unmeasured) mode: no truth poses/depth enter reconstruction. Export
observers call the original methods unchanged; snapshot records use the shipped fade state.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ito.protocol import Intrinsics
from ito.reconstruction.slam import SLAMBackend


class ExportBackend(SLAMBackend):
    def integrate_points(self, points, colors, now, sizes=None):
        self.export_points = self.xp.asnumpy(points)
        self.export_colors = self.xp.asnumpy(colors)
        return super().integrate_points(points, colors, now, sizes)

    def carve(self, camera, fx, fy, cx, cy, depth, now):
        self.export_depth = self.xp.asnumpy(depth)
        self.export_k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return super().carve(camera, fx, fy, cx, cy, depth, now)


def snapshot(backend, now):
    live = backend.keys >= 0
    records = backend.records[live].copy()
    # Offline acknowledgements have no rendering delay; emulate publication deadlines.
    retiring = backend.retiring[live]
    deadlines = backend.release_after[live] - 0.05
    records[retiring, 0, 3] *= np.clip((deadlines[retiring] - now) / backend.fade, 0, 1)
    return records[records[:, 0, 3] > 1 / 255]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--budget", type=int, default=1_048_576)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "frames").mkdir()
    (args.out / "snapshots").mkdir()
    (args.out / "keyframes").mkdir()
    rows = [json.loads(s) for s in (args.dataset / "frames.jsonl").read_text().splitlines()]
    if args.limit:
        rows = rows[: args.limit]
    k = Intrinsics.model_validate_json((args.dataset / "camera.json").read_text())
    messages = []
    backend = ExportBackend(args.budget, k, report=messages.append)
    import torch

    targets = set(np.linspace(0, len(rows) - 1, 13, dtype=int))
    origin = rows[0]["sim_time"]
    poses, durations = [], []
    started = time.perf_counter()
    with (args.out / "frames.jsonl").open("w", buffering=1) as index:
        for i, row in enumerate(rows):
            rgb = np.asarray(Image.open(args.dataset / row["rgb"]).convert("RGB"))
            assert hashlib.sha256(rgb.tobytes()).hexdigest() == row["rgb_sha256"]
            now = row["sim_time"] - origin
            before = backend.tracked
            tick = time.perf_counter()
            backend.integrate(rgb, None, np.eye(4), now, measured=False)
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - tick)
            tracked = backend.tracked > before
            result = dict(
                sequence=row["sequence"],
                time=now,
                tracked=tracked,
                camera=backend.camera_pose.tolist(),
                seconds=durations[-1],
                keyframes=[int(f.frame_id) for f in backend.frames],
            )
            if tracked:
                poses.append((i, backend.camera_pose.copy()))
                np.savez_compressed(
                    args.out / "frames" / f"{i:06d}.npz",
                    depth=backend.export_depth,
                    K=backend.export_k,
                    points=backend.export_points,
                    colors=backend.export_colors,
                    camera=backend.camera_pose,
                )
            backend.expire(now, np.inf)
            if tracked:
                for frame in backend.frames:
                    # Latest optimized state of each retained keyframe; ids map to source rows.
                    np.savez_compressed(
                        args.out / "keyframes" / f"{frame.frame_id:06d}.npz",
                        observed_at=i,
                        X_canon=frame.X_canon.cpu().numpy(),
                        confidence=frame.C.cpu().numpy(),
                        T_WC=frame.T_WC.matrix().cpu().numpy(),
                        K=frame.K.cpu().numpy(),
                        rgb=frame.uimg.cpu().numpy(),
                        ito_world=backend.world,
                    )
            if i in targets:
                np.savez_compressed(
                    args.out / "snapshots" / f"{i:06d}.npz",
                    records=snapshot(backend, now),
                    sequence=i,
                    time=now,
                )
            index.write(json.dumps(result) + "\n")
            if i % 25 == 0:
                print(
                    f"{i}/{len(rows)} tracked={backend.tracked} splats={backend.count} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    report = dict(
        dataset=str(args.dataset),
        mode="all source frames, sequential, unmeasured monocular",
        truth_input=False,
        frames=len(rows),
        tracked=backend.tracked,
        restarts=backend.restarts,
        seconds=time.perf_counter() - started,
        frame_ms_median=float(np.median(durations) * 1000),
        frame_ms_p95=float(np.percentile(durations, 95) * 1000),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),
        messages=sorted(set(messages)),
        source_sha256={
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "ito/reconstruction/slam.py",
                "ito/reconstruction/rgbd.py",
                "ito/reconstruction/mast3r_runtime.py",
            )
        },
    )
    (args.out / "summary.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
