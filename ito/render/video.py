import math

import moderngl
import numpy as np

from ito import clock

DISTANCE = 2.0  # Metres ahead of the pilot's head.
FILL = 0.9
DEFAULT_FOV = math.radians(60)


class VideoPanel:
    def __init__(self, context):
        self.context = context
        self.program = context.program(
            vertex_shader="""#version 430
                uniform mat4 clip_from_panel;
                uniform vec2 half_size;
                uniform float distance;
                out vec2 uv;
                const vec2 corners[4] = vec2[4](
                    vec2(-1,-1), vec2(1,-1), vec2(-1,1), vec2(1,1));
                void main() {
                    vec2 p = corners[gl_VertexID];
                    uv = vec2(p.x * .5 + .5, .5 - p.y * .5);
                    gl_Position = clip_from_panel * vec4(p * half_size, -distance, 1);
                }
            """,
            fragment_shader="""#version 430
                uniform sampler2D camera;
                in vec2 uv;
                out vec4 color;
                void main() { color = vec4(texture(camera, uv).rgb, 1); }
            """,
        )
        self.vao = context.vertex_array(self.program, [])
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

    def draw(self, texture, stamp, eye, projection, target, viewport, *, center=None, fov=None):
        """`texture`: the straightened camera frame (camera.CameraFeed), captured at `stamp`."""
        target.use()
        self.context.viewport = viewport
        target.clear(0.015, 0.02, 0.03, 1)
        if texture is None or clock.now() - stamp > 2:
            return
        size = texture.size
        self.context.disable(moderngl.DEPTH_TEST | moderngl.BLEND | moderngl.CULL_FACE)
        texture.use(0)
        self.program["camera"] = 0
        self.program["half_size"] = self.half_size(size[0] / size[1], fov)
        self.program["distance"] = DISTANCE
        eye_from_head = np.eye(4) if center is None else np.linalg.inv(eye) @ center
        self.program["clip_from_panel"].write(
            np.asarray(projection @ eye_from_head, dtype="f4").T.copy().tobytes()
        )
        self.vao.render(moderngl.TRIANGLE_STRIP, vertices=4)

    def close(self):
        self.vao.release()
        self.program.release()
