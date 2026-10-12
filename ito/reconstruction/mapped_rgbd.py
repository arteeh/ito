"""The same mapper for cameras with registered metric driver depth."""

import numpy as np

from ito import clock

from .mapper import Keyframe
from .mapper_online import OnlineMapper
from .rgbd import RGBDBackend


class MappedRGBDBackend(RGBDBackend):
    def __init__(self, *args, report, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapper = None
        self.last_keyframe = -np.inf
        self.node = 0
        if self.xp is not np:
            try:
                self.mapper = OnlineMapper(self.budget, report)
            except (ImportError, OSError, RuntimeError) as exc:
                report(f"Posed RGB-D | point map fallback: {exc}")
        if self.mapper is not None:
            report("Posed RGB-D | optimised Gaussian map (SH 0)")
        elif self.xp is np:
            report("Posed RGB-D | point map fallback (CPU)")

    @property
    def scene(self):
        return self.mapper if self.mapper is not None else self

    def integrate(self, rgb, depth, camera, now, *, captured=None):
        if self.mapper is None:
            return super().integrate(rgb, depth, camera, now)
        if now - self.last_keyframe < 0.2:
            return
        i = self.intrinsics
        y, x = np.mgrid[: i.height, : i.width]
        points = np.stack(((x - i.cx) / i.fx, -(y - i.cy) / i.fy, -np.ones_like(x)), -1)
        points = (points * depth[..., None]).reshape(-1, 3).astype(np.float32)
        valid = np.isfinite(depth) & (depth > 0) & (depth < 1000)
        self.mapper.submit(
            Keyframe(
                self.node, rgb, camera, i, points, rgb.reshape(-1, 3), valid.ravel(), now, depth
            )
        )
        self.node += 1
        self.last_keyframe = now

    def close(self):
        if self.mapper is not None:
            self.mapper.close()
