import math

import numpy as np
import vulkan as vk

from ito import clock

from .vk import Pipeline, mat, pack

DISTANCE = 2.0  # Metres ahead of the pilot's head.
FILL = 0.9
DEFAULT_FOV = math.radians(60)
CLEAR = (0.015, 0.02, 0.03, 1)


class VideoPanel:
    def __init__(self, device, format):
        self.device = device
        self.pipeline = Pipeline(
            device, ("panel.vert", "panel.frag"), {0: "sampler"}, push=76, colours=(format,),
            topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
        )  # fmt: skip
        self.fit = None  # half extents every view of the frame holds at DISTANCE

    def prepare(self, projections):
        """Once per frame, from every view's projection: one panel size for all of them, so a
        headset's eyes, whose fields differ, see the same panel."""
        self.fit = tuple(
            min(DISTANCE * (1 - abs(float(p[i, 2]))) / float(p[i, i]) for p in projections)
            for i in (0, 1)
        )

    def half_size(self, aspect, fov=None):
        width = DISTANCE * math.tan((fov or DEFAULT_FOV) / 2)
        width = min(width, FILL * self.fit[0], FILL * self.fit[1] * aspect)
        return width, width / aspect

    def draw(self, texture, stamp, eye, projection, target, *, center=None, fov=None):
        """`texture`: the straightened camera frame (camera.CameraFeed), captured at `stamp`;
        target: the view's image (an OpenGL-ordered offscreen one, vk.Presenter)."""
        frame = self.device.frame
        live = texture is not None and clock.now() - stamp <= 2
        if live:
            frame.sampled(texture)
        with frame.render([target], clear=CLEAR):
            if not live:
                return
            size = texture.size
            eye_from_head = np.eye(4) if center is None else np.linalg.inv(eye) @ center
            frame.use(
                self.pipeline,
                {0: (texture, self.device.sampler())},
                pack(
                    "16f2ff",
                    mat(projection @ eye_from_head),
                    self.half_size(size[0] / size[1], fov),
                    DISTANCE,
                ),  # fmt: skip
            )
            frame.draw(4)

    def close(self):
        self.pipeline.close()
