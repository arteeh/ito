"""The flat camera feed on a panel held straight ahead of the pilot.

Without a 3D scene there is nothing to look around in: a panel left in the world drifts out
of view when the pilot turns, and one steered by the camera's own motion shakes. The panel
moves with the pilot's head instead, at a fixed distance, as wide as the camera sees when that
fits the view. In a headset both eyes see the same panel from their own positions.
"""

import math

import moderngl
import numpy as np

from ito import clock

DISTANCE = 2.0  # Metres ahead of the pilot's head.
FILL = 0.9  # Largest share of the view the panel may cover.
DEFAULT_FOV = math.radians(60)  # Horizontal angle when the camera's is unknown.


class VideoPanel:
    def __init__(self, context):
        self.context = context
        self.texture = None
        self.stamp = None
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

    @staticmethod
    def half_size(projection, aspect, fov=None):
        """Panel half extent at DISTANCE: the camera's own angle, shrunk to fit the view."""
        width = DISTANCE * math.tan((fov or DEFAULT_FOV) / 2)
        # An off-axis (headset) frustum is narrower on one side; fit the narrower side.
        fit_x = DISTANCE * (1 - abs(float(projection[0, 2]))) / float(projection[0, 0])
        fit_y = DISTANCE * (1 - abs(float(projection[1, 2]))) / float(projection[1, 1])
        width = min(width, FILL * fit_x, FILL * fit_y * aspect)
        return width, width / aspect

    def draw(self, rgb, stamp, eye, projection, target, viewport, *, center=None, fov=None):
        """Draw for one eye; center is the head pose between the eyes (the eye on a desktop)."""
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
        self.program["half_size"] = self.half_size(projection, size[0] / size[1], fov)
        self.program["distance"] = DISTANCE
        eye_from_head = np.eye(4) if center is None else np.linalg.inv(eye) @ center
        self.program["clip_from_panel"].write(
            np.asarray(projection @ eye_from_head, dtype="f4").T.copy().tobytes()
        )
        self.vao.render(moderngl.TRIANGLE_STRIP, vertices=4)

    def close(self):
        if self.texture:
            self.texture.release()
        self.vao.release()
        self.program.release()
