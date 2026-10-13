"""Replay the mesh and frozen main in the baseline's three synchronized head/map clips."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio.v2 as imageio
import moderngl
import numpy as np
from depthmesh_evaluate import sheet
from measure_gl import projection
from measure_images import edges, percentiles

from ito.proto.depthmesh import MeshWorld
from ito.protocol import Intrinsics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("../p47-out").resolve())
    p.add_argument("--work", type=Path, default=Path("..").resolve())
    p.add_argument("--variant", default="metric-small")
    p.add_argument("--pose", choices=["oracle", "robot"], default="robot")
    args = p.parse_args()
    source = args.root / "prepared/sweep180"
    data = np.load(source / "trajectory.npz")
    i = Intrinsics.model_validate_json((source / "camera.json").read_text())
    timings = json.loads((args.root / f"depth/{args.variant}/sweep180/inference.json").read_text())[
        "timings"
    ]
    durations = {r["frame"]: r["seconds"] for r in timings}
    publications = []
    finish = 0
    for frame, arrival in zip(data["frames"], data["arrivals"], strict=True):
        finish = max(finish, float(arrival)) + durations[int(frame)] + 0.01
        publications.append(finish)
    plan = json.loads((source / "schedule.json").read_text())
    start = next(s["deadline"] for s in plan["snapshots"] if s["frame"] == 495)
    out = args.root / f"clips/{args.variant}-{args.pose}"
    out.mkdir(parents=True, exist_ok=True)
    ctx = moderngl.create_standalone_context(require=430, backend="egl")
    if "llvmpipe" not in ctx.info["GL_RENDERER"].lower():
        raise RuntimeError("Clips must render on Mesa")
    target = ctx.simple_framebuffer((i.width, i.height))
    report = []
    # Each main clip already preserves the exact truth/head trajectory and diagnostic margin.
    for prefix in ("", "bookshelf-"):
        for mode in ("frozen-map-moving-head", "fixed-head-map-updates", "both"):
            world = MeshWorld(ctx, i)
            basename = prefix + mode
            baseline = imageio.get_reader(args.work / f"m1-out/verified/repeat-1/{basename}.mp4")
            writer = imageio.get_writer(out / f"{basename}.mp4", fps=30, macro_block_size=1)
            slow = imageio.get_writer(out / f"{basename}-slow.mp4", fps=10, macro_block_size=1)
            poses = np.load(args.work / "gs/out/sweep180/data.npz")["poses"]
            next_patch = 0
            previous = None
            residual_changes, edge_changes, trace, panels, captions = [], [], [], [], []
            for n in range(61):
                deadline = start + n / 30
                map_deadline = start if mode == "frozen-map-moving-head" else deadline
                while next_patch < len(publications) and publications[next_patch] <= map_deadline:
                    k = next_patch
                    frame = int(data["frames"][k])
                    sample = np.load(source / f"{frame:06d}.npz")
                    depth = np.load(args.root / f"depth/{args.variant}/sweep180/{frame:06d}.npy")
                    world.add(
                        sample["rgb"],
                        depth,
                        data["truth" if args.pose == "oracle" else "robot"][k],
                        frame=frame,
                        captured=float(data["captured"][k]),
                    )
                    next_patch += 1
                camera = poses[452 if prefix else 525].copy()
                if mode != "fixed-head-map-updates":
                    camera[:3, 3] += camera[:3, 0] * 0.03 * np.sin(n / 60 * 2 * np.pi)
                world.draw(camera, projection(i), target)
                pixels = np.frombuffer(target.read(components=4), "u1").reshape(
                    i.height, i.width, 4
                )[::-1]
                frozen = baseline.get_data(n)[-i.height :, : 2 * i.width]
                truth = frozen[:, : i.width].astype("f4") / 255
                candidate = pixels[:, :, :3].astype("f4") / 255
                residual = candidate - truth
                edge = edges(candidate)
                if previous is not None:
                    residual_changes.append(float(np.abs(residual - previous[0]).mean()))
                    edge_changes.append(float(np.mean(edge != previous[1])))
                previous = residual, edge
                row = dict(
                    frame=n,
                    deadline=deadline,
                    observation_frame=int(data["frames"][next_patch - 1]) + 1
                    if next_patch
                    else None,
                    publication=publications[next_patch - 1] if next_patch else None,
                    coverage=float((pixels[:, :, 3] > 127).mean()),
                )
                trace.append(row)
                triplet = np.concatenate((frozen, pixels[:, :, :3]), axis=1)
                caption = f"{basename} t={n / 30:.2f}s | truth / main / {args.variant}-{args.pose}"
                margin = np.zeros((32, triplet.shape[1], 3), "u1")
                from PIL import Image, ImageDraw

                labelled = Image.fromarray(np.concatenate((margin, triplet)))
                ImageDraw.Draw(labelled).text((5, 8), caption, fill="white")
                writer.append_data(np.asarray(labelled))
                slow.append_data(np.asarray(labelled))
                if n in (0, 15, 30, 45, 60):
                    panels.append(triplet)
                    captions.append(caption)
            writer.close()
            slow.close()
            baseline.close()
            world.close()
            sheet(panels, captions, out / f"{basename}-filmstrip.jpg")
            (out / f"{basename}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in trace))
            report.append(
                dict(
                    clip=basename,
                    residual_change=percentiles(residual_changes),
                    edge_changed_fraction=percentiles(edge_changes),
                    minimum_coverage=min(r["coverage"] for r in trace),
                )
            )
            print(basename, report[-1], flush=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    target.release()
    ctx.release()


if __name__ == "__main__":
    main()
