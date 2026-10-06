"""Drive the real GL renderer with randomized scenes: xvfb-run -a uv run python e2e/render.py."""

from pathlib import Path
import math

import moderngl
import numpy as np
import pygame

from ito.render import GaussianBuffer, GaussianRenderer, load_ply, perspective, pose
from sample_scene import write_scene


def main():
    output = Path("e2e/out/render")
    output.mkdir(parents=True, exist_ok=True)
    pygame.display.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_mode((320, 240), pygame.OPENGL | pygame.DOUBLEBUF)
    context = moderngl.create_context(require=430, libgl=None)
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
                pygame.image.save(pygame.transform.flip(image, False, True), output / f"{path.stem}.png")
        # Pass beside the closest floor splats without letting them flood the view.
        renderer.draw(pose(), pose((0, 0, -0.975)), projection, target)
        rgb = np.frombuffer(target.read(components=3), np.uint8).reshape(240, 320, 3)
        assert np.count_nonzero(rgb.max(axis=2) - rgb.min(axis=2) > 60) > 2500
        print(f"PASS: GPU sorting, compositing, posed views, ASCII/binary PLY SH 0–3; {context.info['GL_RENDERER']}")
    finally:
        target.release()
        renderer.close()
        context.release()
        pygame.quit()


if __name__ == "__main__":
    main()
