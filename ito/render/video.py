"""A world-locked flat video panel; late camera frames never steer the pilot's view."""

import moderngl
import numpy as np

from ito import clock


class VideoPanel:
    def __init__(self, context):
        self.context = context
        self.texture = None
        self.stamp = None
        self.program = context.program(
            vertex_shader="""#version 430
                uniform mat4 clip_from_panel;
                uniform float aspect;
                out vec2 uv;
                const vec2 corners[4] = vec2[4](
                    vec2(-1,-1), vec2(1,-1), vec2(-1,1), vec2(1,1));
                void main() {
                    vec2 p = corners[gl_VertexID];
                    uv = vec2(p.x * .5 + .5, .5 - p.y * .5);
                    gl_Position = clip_from_panel * vec4(p.x * 1.2, p.y * 1.2 / aspect, -2, 1);
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

    def draw(self, rgb, stamp, head, projection, target, viewport):
        target.use()
        self.context.viewport = viewport
        target.clear(0.015, 0.02, 0.03, 1)
        if rgb is None or clock.now() - stamp > 2:
            return
        size = (rgb.shape[1], rgb.shape[0])
        if self.texture is None or self.texture.size != size:
            if self.texture:
                self.texture.release()
            self.texture = self.context.texture(size, 3, alignment=1)
            self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.stamp = None
        if self.stamp != stamp:
            self.texture.write(rgb.tobytes(), alignment=1)
            self.stamp = stamp
        self.context.disable(moderngl.DEPTH_TEST | moderngl.BLEND | moderngl.CULL_FACE)
        self.texture.use(0)
        self.program["camera"] = 0
        self.program["aspect"] = size[0] / size[1]
        self.program["clip_from_panel"].write(
            np.asarray(projection @ np.linalg.inv(head), dtype="f4").T.copy().tobytes()
        )
        self.vao.render(moderngl.TRIANGLE_STRIP, vertices=4)

    def close(self):
        if self.texture:
            self.texture.release()
        self.vao.release()
        self.program.release()
