"""Matched temporal rendering against #47's pinned policy, using cached oracle inputs.

The reference implementation is loaded read only from p47 at 932c7d6. This creates
new matched temporal evidence, without rerunning any depth inference or #47 sweep.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from measure_gl import projection
from viewcache_render import context, sequence

from ito.proto.viewcache import ViewCache

REFERENCE_SHA256 = "2c275be655d4ccaf2d321fe093deafdb2b34a16df1e20ad9d7fb214c162eabc7"


def reference(work):
    path = work / "p47/ito/proto/depthmesh/renderer.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("The frozen #47 renderer changed")
    spec = importlib.util.spec_from_file_location("depthmesh_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MeshWorld


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, default=Path("..").resolve())
    p.add_argument("--out", type=Path, default=Path("../p48-out").resolve())
    p.add_argument("--variant", default="truth")
    p.add_argument("--pose", choices=["oracle", "robot"], default="oracle")
    p.add_argument("--hysteresis", type=float, default=0.15)
    args = p.parse_args()
    ctx = context()
    MeshWorld = reference(args.work)
    report = dict(
        variant=args.variant,
        pose=args.pose,
        hysteresis=args.hysteresis,
        gl_renderer=ctx.info["GL_RENDERER"],
        reference_renderer_sha256=REFERENCE_SHA256,
        clips=[],
    )
    for run in ("sweep180", "tour"):
        folder, data, i, schedule, rows = sequence(args.work, run, args.variant, args.pose)
        true = np.load(args.work / f"gs/out/{run}/data.npz")["poses"]
        poses = data["truth" if args.pose == "oracle" else "robot"]
        start = next(
            s["deadline"]
            for s in schedule["snapshots"]
            if s["frame"] == (495 if run == "sweep180" else 2283)
        )
        # Tour's frozen-head clip starts at the final available prefix. Map clips
        # approach that prefix from two seconds earlier, never admitting future RGB.
        for prefix, index in (
            [("", 525), ("bookshelf-", 452)] if run == "sweep180" else [("", 2283)]
        ):
            for mode in ("frozen-map-moving-head", "fixed-head-map-updates", "both"):
                begin = (
                    start if run == "sweep180" or mode == "frozen-map-moving-head" else start - 2
                )
                basename = prefix + mode
                out = args.out / f"clips/{args.variant}-{args.pose}/{run}/{basename}"
                out.mkdir(parents=True, exist_ok=True)
                cache = ViewCache(ctx, i, hysteresis=args.hysteresis)
                mesh = MeshWorld(ctx, i)
                targets = [
                    ctx.simple_framebuffer((i.width, i.height), components=4) for _ in range(2)
                ]
                next_source = 0
                trace, cameras, cache_pixels, mesh_pixels = [], [], [], []
                for n in range(61):
                    deadline = begin + n / 30
                    map_deadline = begin if mode == "frozen-map-moving-head" else deadline
                    while (
                        next_source < len(rows) and rows[next_source]["published"] <= map_deadline
                    ):
                        k = next_source
                        row = rows[k]
                        sample = np.load(folder / f"{row['frame']:06d}.npz")
                        depth = (
                            sample["truth_depth"]
                            if args.variant == "truth"
                            else np.load(
                                args.work
                                / f"p47-out/depth/{args.variant}/{run}/{row['frame']:06d}.npy"
                            )
                        )
                        for world in (cache, mesh):
                            world.add(
                                sample["rgb"],
                                depth,
                                poses[k],
                                frame=row["frame"],
                                captured=row["capture"],
                            )
                        next_source += 1
                    camera = true[index].copy()
                    if mode != "fixed-head-map-updates":
                        camera[:3, 3] += camera[:3, 0] * 0.03 * np.sin(n / 60 * 2 * np.pi)
                    cache.prepare(camera, projection(i))
                    for world, target, pixels in zip(
                        (cache, mesh), targets, (cache_pixels, mesh_pixels), strict=True
                    ):
                        world.draw(camera, projection(i), target)
                        pixels.append(
                            np.frombuffer(target.read(components=4), "u1")
                            .reshape(i.height, i.width, 4)[::-1]
                            .copy()
                        )
                    cameras.append(camera)
                    trace.append(
                        dict(
                            frame=n,
                            deadline=deadline,
                            map_deadline=map_deadline,
                            publication=rows[next_source - 1]["published"] if next_source else None,
                            observation_frame=rows[next_source - 1]["frame"] + 1
                            if next_source
                            else None,
                            retained_frames=[s.frame for s in cache.sources],
                            selected_frames=[s.frame for s, _ in cache.selected],
                        )
                    )
                    if next_source and (
                        trace[-1]["publication"] > map_deadline
                        or rows[next_source - 1]["capture"] > deadline
                    ):
                        raise AssertionError("Future publication or observation in temporal clip")
                np.savez_compressed(
                    out / "frames.npz", cache=cache_pixels, mesh=mesh_pixels, cameras=cameras
                )
                (out / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
                report["clips"].append(dict(run=run, clip=basename, frames=61, start=begin))
                print(run, basename, "matched 61 frames", flush=True)
                cache.close()
                mesh.close()
                for target in targets:
                    target.release()
    out = args.out / f"clips/{args.variant}-{args.pose}"
    (out / "render.json").write_text(json.dumps(report, indent=2) + "\n")
    ctx.release()


if __name__ == "__main__":
    main()
