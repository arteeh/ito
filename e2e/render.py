"""Drive the real GL renderer with randomized scenes: xvfb-run -a uv run python e2e/render.py."""

import math
import time
from pathlib import Path

import numpy as np
import pygame
from sample_scene import write_scene

from ito.reconstruction.ring import SplatUpdate
from ito.render import (
    GaussianBuffer,
    GaussianRenderer,
    current_context,
    load_ply,
    perspective,
    pose,
)


def main():
    output = Path("e2e/out/render")
    output.mkdir(parents=True, exist_ok=True)
    pygame.display.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_mode((320, 240), pygame.OPENGL | pygame.DOUBLEBUF)
    context = current_context()
    renderer = GaussianRenderer(context)
    target = context.simple_framebuffer((320, 240))
    projection = perspective(math.radians(70), 4 / 3)
    rng = np.random.default_rng(239)
    try:
        for count in (0, 1, 255, 256, 257, 513, 4097, 65537):
            records = np.zeros((count, 4, 4), dtype=np.float32)
            records[:, 0, :3] = rng.uniform(-10, 10, (count, 3))
            records[:, 0, 3] = 0.7
            records[:, 1, :3] = 0.015
            records[:, 2, 0] = 1
            records[:, 3, :3] = rng.uniform(-0.5, 0.5, (count, 3))
            renderer.upload(GaussianBuffer(records))
            for _ in range(3):
                anchor = pose(rng.uniform(-1, 1, 3), yaw=rng.uniform(-3, 3))
                head = pose(rng.uniform(-0.2, 0.2, 3), yaw=rng.uniform(-1, 1), pitch=0.2)
                renderer.draw(anchor, head, projection, target)
                if count:
                    order = np.frombuffer(renderer.order.read(), dtype=np.uint32).reshape(-1, 2)
                    assert np.all(order[:-1, 0] >= order[1:, 0]), f"GPU order failed: {count}"
                    assert np.array_equal(np.sort(order[:count, 1]), np.arange(count))
                    eye = anchor @ head
                    depth = -((records[:, 0, :3] - eye[:3, 3]) @ eye[:3, 2])
                    gpu_depth = order[:count, 0].copy().view(np.float32)
                    assert np.allclose(gpu_depth, np.maximum(0, depth[order[:count, 1]]), atol=5e-6)
                assert context.error == "GL_NO_ERROR"
            print(f"GPU sort and draw: {count} splats")

        # Two overlapping splats: camera reversal must reverse alpha compositing.
        records = np.zeros((2, 4, 4), dtype=np.float32)
        records[:, 0] = ((0, 0, -2, 0.8), (0, 0, -3, 0.8))
        records[:, 1, :3] = 0.3
        records[:, 2, 0] = 1
        records[:, 3, :3] = (np.array(((1, 0, 0), (0, 0, 1))) - 0.5) / 0.2820947918
        renderer.upload(GaussianBuffer(records))
        for anchor, channel in ((pose(), 0), (pose((0, 0, -5), yaw=math.pi), 2)):
            renderer.draw(anchor, pose(), projection, target)
            pixels = np.frombuffer(target.read(components=3), np.uint8).reshape(240, 320, 3)
            assert pixels[120, 160, channel] > pixels[120, 160, 2 - channel] * 3

        # Stream sparse append/evict packets, growing GPU capacity without losing old slots.
        epoch = time.monotonic()
        live = records.copy()
        live[:, 1, 3] = 60
        before_bytes = renderer.uploaded_bytes
        renderer.apply(SplatUpdate(np.array([0, 4], np.uint32), live, 8, 2, 0, epoch, epoch, 0.5))
        renderer.apply(
            SplatUpdate(np.array([12], np.uint32), live[:1], 16, 3, 1, epoch, epoch, 0.5)
        )
        removed = np.zeros((1, 4, 4), np.float32)
        renderer.apply(SplatUpdate(np.array([4], np.uint32), removed, 16, 2, 2, epoch, epoch, 0.5))
        renderer.draw(pose(), pose(), projection, target)
        resident = np.frombuffer(renderer.scene_buffer.read(), np.float32).reshape(16, 4, 4)
        assert np.array_equal(resident[0], live[0])
        assert np.array_equal(resident[12], live[0])
        assert not resident[4].any()
        assert renderer.uploaded_bytes - before_bytes == 4 * (64 + 4)
        assert context.error == "GL_NO_ERROR"
        late_eviction = live[:1].copy()
        late_eviction[:, 1, 3] = -1
        late_eviction[:, 3, 3] = 1
        renderer.apply(
            SplatUpdate(np.array([0], np.uint32), late_eviction, 16, 2, 3, epoch, epoch, 0.5)
        )
        stale = np.frombuffer(renderer.scene_buffer.read(size=64), np.float32).reshape(4, 4)
        assert stale[1, 3] == -1, "Delayed eviction revived expired geometry"
        print("Sparse GPU append/evict and capacity growth preserve resident splats")

        for degree in (0, 1, 2, 3):
            for ascii_ply in (False, True):
                path = output / f"scene-{degree}-{'ascii' if ascii_ply else 'binary'}.ply"
                count = write_scene(path, text=ascii_ply, degree=degree)
                scene = load_ply(path)
                assert scene.count == count and scene.sh_degree == degree
                renderer.upload(scene)
                renderer.draw(pose(), pose(), projection, target)
                pixels = target.read(components=3)
                rgb = np.frombuffer(pixels, np.uint8).reshape(240, 320, 3)
                assert np.count_nonzero(rgb.max(axis=2) - rgb.min(axis=2) > 45) > 2500
                image = pygame.image.frombytes(pixels, target.size, "RGB")
                pygame.image.save(
                    pygame.transform.flip(image, False, True), output / f"{path.stem}.png"
                )
        # Pass beside the closest floor splats without letting them flood the view.
        renderer.draw(pose(), pose((0, 0, -0.975)), projection, target)
        rgb = np.frombuffer(target.read(components=3), np.uint8).reshape(240, 320, 3)
        assert np.count_nonzero(rgb.max(axis=2) - rgb.min(axis=2) > 60) > 2500
        print(
            "PASS: GPU sorting, compositing, posed views, ASCII/binary PLY SH 0–3; "
            f"{context.info['GL_RENDERER']}"
        )
    finally:
        target.release()
        renderer.close()
        context.release()
        pygame.quit()


if __name__ == "__main__":
    main()
