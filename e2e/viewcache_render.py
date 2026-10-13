# ruff: noqa: B023
"""Render the frozen #47 admissions; no simulator, inference or target RGB in the cache."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl" if sys.platform != "win32" else "win32")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import moderngl
import numpy as np
from measure import MOMENTS
from measure_gl import projection
from PIL import Image

from ito.proto.viewcache import ViewCache
from ito.protocol import Intrinsics

VIEWS = [
    ("camera", 0),
    ("left", -0.032),
    ("right", 0.032),
    ("head-minus3cm", -0.03),
    ("head-plus3cm", 0.03),
]


def context():
    if sys.platform == "win32":
        import pygame

        pygame.display.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
        )
        pygame.display.set_mode((64, 64), pygame.OPENGL | pygame.DOUBLEBUF | pygame.HIDDEN)
        ctx = moderngl.create_context(require=430)
        if "NVIDIA" not in ctx.info["GL_VENDOR"]:
            raise RuntimeError(f"Expected locked NVIDIA context: {ctx.info}")
        return ctx
    ctx = moderngl.create_standalone_context(require=430, backend="egl")
    if "llvmpipe" not in ctx.info["GL_RENDERER"].lower():
        raise RuntimeError("Use Mesa on Linux, or Windows NVIDIA under ceres.lock")
    return ctx


def digest(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def sequence(work, run, variant, pose):
    folder = work / "p47-out/prepared" / run
    data = np.load(folder / "trajectory.npz")
    i = Intrinsics.model_validate_json((folder / "camera.json").read_text())
    schedule = json.loads((folder / "schedule.json").read_text())
    prior = work / f"p47-out/rendered/{variant}-{pose}-cap64/metrics.json"
    rows = [r for r in json.loads(prior.read_text())["publications"] if r["run"] == run]
    return folder, data, i, schedule, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, default=Path("..").resolve())
    p.add_argument("--out", type=Path, default=Path("../p48-out").resolve())
    p.add_argument("--variant", default="truth")
    p.add_argument("--pose", choices=["oracle", "robot"], default="oracle")
    p.add_argument("--candidates", type=int, default=12)
    p.add_argument("--hysteresis", type=float, default=0.15)
    args = p.parse_args()
    name = f"{args.variant}-{args.pose}"
    out = args.out / "rendered" / name
    out.mkdir(parents=True, exist_ok=True)
    ctx = context()
    report = dict(
        variant=args.variant,
        pose=args.pose,
        gl_renderer=ctx.info["GL_RENDERER"],
        cap=64,
        candidates=args.candidates,
        hysteresis=args.hysteresis,
        grid=[90, 160],
        publications=[],
        moments=[],
        inputs={},
        availability="Exact #47 queued inference + declared 10ms publication budget",
        freshness_certified=False,
    )
    baseline = args.work / "m1-out/verified/repeat-1"
    for run, moments in MOMENTS.items():
        folder, data, i, schedule, rows = sequence(args.work, run, args.variant, args.pose)
        poses = data["truth" if args.pose == "oracle" else "robot"]
        true = np.load(args.work / f"gs/out/{run}/data.npz")["poses"]
        cache = ViewCache(ctx, i, candidates=args.candidates, hysteresis=args.hysteresis)
        target = ctx.simple_framebuffer((i.width, i.height), components=4)
        pending = iter(
            sorted(
                [s for s in schedule["snapshots"] if s["frame"] in moments],
                key=lambda s: s["deadline"],
            )
        )
        snapshot = next(pending, None)
        last = None
        used_paths = [folder / "trajectory.npz", folder / "camera.json", folder / "schedule.json"]

        def capture(s):
            index, deadline = s["frame"], s["deadline"]
            if last is not None and (last["published"] > deadline or last["frame"] >= index):
                raise AssertionError("Target/future observation entered the cache")
            midpoint = true[index].copy()
            began = time.perf_counter()
            cache.prepare(midpoint, projection(i))
            ctx.finish()
            selection_ms = (time.perf_counter() - began) * 1000
            front = np.frombuffer(cache.layers[1][0].read(), "f4").reshape(i.height, i.width, 4)[
                ::-1
            ]
            back = np.frombuffer(cache.layers[3][0].read(), "f4").reshape(i.height, i.width, 4)[
                ::-1
            ]
            np.savez_compressed(
                out / f"{run}-{index + 1:04d}-ownership.npz",
                front=front,
                back=back,
                retained_frames=[s.frame for s in cache.sources],
            )
            for label, offset in VIEWS:
                eye = midpoint.copy()
                eye[:3, 3] += eye[:3, 0] * offset
                began = time.perf_counter()
                cache.draw(eye, projection(i), target)
                ctx.finish()
                draw_ms = (time.perf_counter() - began) * 1000
                pixels = (
                    np.frombuffer(target.read(components=4), "u1")
                    .reshape(i.height, i.width, 4)[::-1]
                    .copy()
                )
                filename = f"{run}-{index + 1:04d}-{label}"
                frozen_path = baseline / f"{filename}.png"
                mesh_path = args.work / f"p47-out/rendered/{name}-cap64/{filename}.png"
                frozen = np.asarray(Image.open(frozen_path).convert("RGB"))
                mesh = np.asarray(Image.open(mesh_path).convert("RGB"))[:, -i.width :]
                Image.fromarray(np.concatenate((frozen, mesh, pixels[:, :, :3]), axis=1)).save(
                    out / f"{filename}.png"
                )
                masks = np.load(baseline / f"{filename}-mask.npz")
                np.savez_compressed(
                    out / f"{filename}-mask.npz", covered=pixels[:, :, 3] > 127, seen=masks["seen"]
                )
                report["moments"].append(
                    dict(
                        run=run,
                        frame=index + 1,
                        view=label,
                        prefix_frame=last["frame"] + 1 if last else None,
                        publication=last["published"] if last else None,
                        deadline=deadline,
                        selection_ms=selection_ms,
                        draw_ms=draw_ms,
                        retained=len(cache.sources),
                        active=len(cache.selected),
                    )
                )
                print(name, filename, "rendered", flush=True)
            if index == max(moments):
                cache.save(out / f"{run}-world.npz")
                np.save(out / f"{run}-camera.npy", midpoint)

        for k, row in enumerate(rows):
            while snapshot is not None and snapshot["deadline"] < row["published"]:
                capture(snapshot)
                snapshot = next(pending, None)
            frame = row["frame"]
            source_path = folder / f"{frame:06d}.npz"
            sample = np.load(source_path)
            used_paths.append(source_path)
            if args.variant == "truth":
                depth = sample["truth_depth"]
            else:
                depth_path = args.work / f"p47-out/depth/{args.variant}/{run}/{frame:06d}.npy"
                depth = np.load(depth_path)
                used_paths.append(depth_path)
            cache.add(sample["rgb"], depth, poses[k], frame=frame, captured=row["capture"])
            last = row
            report["publications"].append(row)
        while snapshot is not None:
            capture(snapshot)
            snapshot = next(pending, None)
        for path in used_paths:
            report["inputs"][str(path.relative_to(args.work))] = digest(path)
        cache.close()
        target.release()
    (out / "render.json").write_text(json.dumps(report, indent=2) + "\n")
    ctx.release()
    if sys.platform == "win32":
        import pygame

        pygame.display.quit()


if __name__ == "__main__":
    main()
