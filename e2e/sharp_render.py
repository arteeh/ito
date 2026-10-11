"""Synchronized renderer cost at the requested 400k and maximum 4,194,304 budgets.

xvfb-run -a uv run python e2e/sharp_render.py
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pygame

from ito.render import GaussianBuffer, GaussianRenderer, current_context, perspective, pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, action="append", help="time a scene_truth snapshot")
    args = parser.parse_args()
    pygame.display.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_mode((64, 64), pygame.OPENGL | pygame.HIDDEN)
    context = current_context()
    renderer = GaussianRenderer(context)
    target = context.simple_framebuffer((960, 540))
    try:
        limit = context.info["GL_MAX_SHADER_STORAGE_BLOCK_SIZE"] // 64
        maximum = min(4_194_304, limit)
        print(json.dumps(dict(requested_max=4_194_304, supported_max=maximum)), flush=True)
        for case in args.scene or (400_000, maximum):
            if isinstance(case, Path):
                with np.load(case) as snapshot:
                    records = snapshot["records"]
                    camera = snapshot["view_cameras"][-1]
                    fovy = snapshot["view_fovy"][-1]
                count = len(records)
            else:
                count = case
                camera, fovy = pose(), 65
                records = np.zeros((count, 4, 4), np.float32)
                side = math.ceil(math.sqrt(count))
                index = np.arange(count)
                records[:, 0, 0] = ((index % side) / side - 0.5) * 5
                records[:, 0, 1] = ((index // side) / side - 0.5) * 3
                records[:, 0, 2:] = [-3, 0.98]
                records[:, 1, :3] = [2.5 / side, 1.5 / side, 0.075 / side]
                records[:, 2, 0] = 1
                records[:, 1, 3] = 1e30  # Live surfels, retained until pressure evicts them.
            renderer.upload(GaussianBuffer(records))
            times = []
            for _ in range(12):
                context.finish()
                start = time.perf_counter()
                renderer.draw(
                    camera,
                    pose(),
                    perspective(math.radians(fovy), 16 / 9),
                    target,
                    clear=(0, 0, 0, 0),
                )
                context.finish()
                times.append((time.perf_counter() - start) * 1000)
            if not isinstance(case, Path):
                pixels = np.frombuffer(target.read(components=4), np.uint8).reshape(540, 960, 4)
                assert np.percentile(pixels[135:405, 240:720, 3], 1) > 250, (
                    "opaque surface has gaps"
                )
            print(
                json.dumps(
                    dict(
                        scene=str(case),
                        splats=count,
                        renderer=context.info["GL_RENDERER"],
                        frame_p50_ms=float(np.median(times[2:])),
                        frame_max_ms=max(times[2:]),
                        frame_p95_ms=float(np.percentile(times[2:], 95)),
                    )
                ),
                flush=True,
            )
        assert context.error == "GL_NO_ERROR", context.error
    finally:
        renderer.close()
        target.release()
        pygame.quit()


if __name__ == "__main__":
    main()
