"""Fenced stereo mesh draw and retirement costs on Windows' NVIDIA OpenGL context."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import moderngl
import numpy as np
import pygame
from measure_gl import projection

from ito.proto.depthmesh import MeshWorld


def percentiles(values):
    return dict(
        zip(
            ("p50", "p95", "p99", "p99.9"),
            np.percentile(values, [50, 95, 99, 99.9]).tolist(),
            strict=True,
        )
    )


def timing(values):
    misses = np.asarray(values) > 1000 / 90
    streak = longest = 0
    for miss in misses:
        streak = streak + 1 if miss else 0
        longest = max(streak, longest)
    return dict(
        ms=percentiles(values),
        missed_fraction=float(misses.mean()),
        longest_miss_run=longest,
        samples=len(values),
        deadline_ms=1000 / 90,
    )


def main():
    print("Starting mesh GL timing", flush=True)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("world", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frames", type=int, default=1000)
    p.add_argument("--pace", action="store_true", help="Pace at 90Hz for concurrent inference")
    args = p.parse_args()
    pygame.display.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_mode((64, 64), pygame.OPENGL | pygame.DOUBLEBUF | pygame.HIDDEN)
    ctx = moderngl.create_context(require=430)
    print(ctx.info["GL_RENDERER"], flush=True)
    if "NVIDIA" not in ctx.info["GL_VENDOR"]:
        raise RuntimeError(f"Expected NVIDIA under the GPU lock: {ctx.info['GL_RENDERER']}")
    began = time.perf_counter()
    world = MeshWorld.load(ctx, args.world)
    print("Loaded saved world", flush=True)
    ctx.finish()
    report = dict(
        scope="Isolated fenced offscreen stereo; no inference/compositor/scanout",
        gl_renderer=ctx.info["GL_RENDERER"],
        patches=len(world.patches),
        load_including_disk_ms=(time.perf_counter() - began) * 1000,
        frames_per_row=args.frames,
        rows=[],
    )
    anchor_file = args.world.with_name(args.world.name.replace("-world.npz", "-camera.npy"))
    anchor = np.load(anchor_file)
    for width, height in ((360, 640), (1080, 1920)):
        i = world.i.model_copy(
            update=dict(
                width=width,
                height=height,
                fx=world.i.fx * width / world.i.width,
                fy=world.i.fy * height / world.i.height,
                cx=width / 2,
                cy=height / 2,
            )
        )
        targets = [ctx.simple_framebuffer((width, height)) for _ in range(2)]
        durations = []
        for n in range(args.frames + 10):
            if args.pace:
                time.sleep(1 / 90)
            camera = anchor.copy()
            camera[:3, 3] += camera[:3, 0] * 0.03 * np.sin(n / 90 * 2 * np.pi)
            start = time.perf_counter()
            for side, target in zip((-1, 1), targets, strict=True):
                eye = camera.copy()
                eye[:3, 3] += eye[:3, 0] * side * 0.032
                world.draw(eye, projection(i), target)
            ctx.finish()
            if n >= 10:
                durations.append((time.perf_counter() - start) * 1000)
        report["rows"].append(dict(size=[width, height], **timing(durations)))
        for target in targets:
            target.release()
    newest = world.patches[-1]
    rgb = np.frombuffer(newest.color.read(), "u1").reshape(world.i.height, world.i.width, 3).copy()
    depth = np.frombuffer(newest.depth.read(), "f4").reshape(world.i.height, world.i.width).copy()
    camera = newest.pose.copy()
    updates = []
    for n in range(30):
        # Keep the same bounded 64-slot world; identical revisits permanently retire old surfaces.
        start = time.perf_counter()
        world.add(
            rgb, depth, camera, frame=newest.frame + n + 1, captured=newest.captured + (n + 1) * 0.2
        )
        ctx.finish()
        updates.append((time.perf_counter() - start) * 1000)
    report["retirement_update_ms"] = percentiles(updates)
    report["payload_upper_bound_mb"] = world.cap * world.i.width * world.i.height * 8 / 2**20
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    world.close()
    ctx.release()
    pygame.display.quit()


if __name__ == "__main__":
    main()
