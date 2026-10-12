"""Map adapter; tracking's fusion/SLAM modules remain independent of gsplat."""

from .mapper import Keyframe
from .mapper_online import OnlineMapper
from .slam import SLAMBackend


class MappedSLAMBackend(SLAMBackend):
    """Robot-first tracking feeding the optimised Gaussian map: keyframes seed and anchor
    Gaussians, every other frame the front end placed trains them from its own view."""

    def __init__(self, *args, **kwargs):
        self.mapper = None
        super().__init__(*args, **kwargs)
        try:
            self.mapper = OnlineMapper(self.budget, self.report)
        except (ImportError, OSError, RuntimeError) as exc:
            self.mapper_status = f"Point map fallback: {exc}"
        else:
            self.mapper_status = "Optimised Gaussian map (SH 0)"
        self.report(self.mapper_status)

    @property
    def scene(self):
        return self.mapper if self.mapper is not None else self

    def integrate(self, rgb, depth, camera, now, measured=False, *, captured):
        self.map_rgb, self.map_captured = rgb, captured
        super().integrate(rgb, depth, camera, now, measured, captured=captured)
        if self.mapper is not None:
            poses = {node: self.graph.pose(node) for node in self.graph.keyframes}
            self.mapper.correct(poses)
        self.report(f"{self.mapper_status} | tracking {self.tracked} frames")

    def place(self, cloud, pose, when):
        if self.mapper is None:
            return super().place(cloud, pose, when)
        # A frame the front end just made a keyframe was promoted out of "current".
        keyframe = self.graph.current is None
        anchor = self.graph.latest_keyframe()
        node = anchor if keyframe else self.graph.current
        if anchor is None:
            return
        self.mapper.submit(
            Keyframe(
                node=node,
                rgb=self.map_rgb,
                pose=pose,
                intrinsics=self.intrinsics,
                points=cloud.points if keyframe else None,
                colors=cloud.colors if keyframe else None,
                confidence=cloud.valid if keyframe else None,
                captured_at=self.map_captured,
                anchor=anchor,
            )
        )

    def close(self):
        if self.mapper is not None:
            self.mapper.close()
