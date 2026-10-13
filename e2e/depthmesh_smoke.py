"""Drive the real offscreen mesh renderer through projection, carving and eviction.

LIBGL_ALWAYS_SOFTWARE=1 ../p47-out/venv/bin/python e2e/depthmesh_smoke.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import moderngl
import numpy as np
from measure_gl import projection
from PIL import Image

from ito.proto.depthmesh import MeshWorld
from ito.protocol import Intrinsics


def main():
    ctx = moderngl.create_standalone_context(require=430, backend="egl")
    i = Intrinsics(width=160, height=90, fx=140, fy=140, cx=80, cy=45)
    target = ctx.simple_framebuffer((i.width, i.height), components=4)
    world = MeshWorld(ctx, i, cap=2, grid=(80, 45))
    camera = np.eye(4, dtype="f4")
    y, x = np.mgrid[: i.height, : i.width]
    rgb = np.stack((x * 255 // i.width, y * 255 // i.height, np.full_like(x, 100)), -1).astype("u1")
    depth = np.full((i.height, i.width), 2, "f4")
    world.add(rgb, depth, camera, frame=0, captured=0)
    world.draw(camera, projection(i), target)
    rendered = np.frombuffer(target.read(components=4), "u1").reshape(i.height, i.width, 4)[::-1]
    assert rendered[..., 3].min() == 255, "Pinhole projection leaves holes at capture pose"
    assert np.abs(rendered[..., :3].astype(float) - rgb).max() <= 2, (
        "Texture orientation/calibration"
    )
    # A farther fresh wall contradicts the entire old foreground surface.
    world.add(rgb, depth * 2, camera, frame=1, captured=1)
    assert not np.frombuffer(world.patches[0].active.read(), "u1").any(), "Free space not retired"
    # Newer source eviction cannot resurrect a retired surface.
    far_side = camera.copy()
    far_side[0, 3] = 20
    world.add(rgb, depth, far_side, frame=2, captured=2)
    assert len(world.patches) == 2
    world.draw(camera, projection(i), target)
    rendered = np.frombuffer(target.read(components=4), "u1").reshape(i.height, i.width, 4)[::-1]
    assert rendered[..., 3].min() == 255, "Remaining wall should stay visible"
    invalid = np.full_like(depth, np.nan)
    assert not world.add(rgb, invalid, camera, frame=3, captured=3)
    assert len(world.patches) == 2, "Invalid depth must preserve last good world"
    try:
        world.add(rgb, depth, camera * 0, frame=3, captured=3)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid pose admitted")
    out = Path("../p47-out/smoke")
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rendered[..., :3]).save(out / "projection.png")
    world.close()
    target.release()
    ctx.release()
    print("PASS: GL projection, texture, free-space retirement, eviction, invalid inputs")


if __name__ == "__main__":
    main()
