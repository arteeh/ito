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


# The live frame sits this far out along the camera's view: its pixels then cover exactly
# what the camera saw from the pilot's anchored eye, whatever the distance.
LIVE_DISTANCE = 2.0
# Its edge blends into the scene over this share of the frame, and it fades out between
# LIVE_HOLD and LIVE_GONE seconds after capture once frames stop coming.
LIVE_FEATHER = 0.08
LIVE_HOLD, LIVE_GONE = 0.3, 1.0


class LiveFrame:
    """The newest camera frame, hung in the scene where the camera looked when it took it.

    Reconstruction places surroundings at its own pace (SLAM, a few frames a second); the
    pilot sees every frame the robot sends regardless. The frame is world-locked by its own
    measured camera orientation, so the robot's head trailing the pilot's shows as the frame
    trailing the view, never as the world swinging, and it reprojects for each display frame
    from the pilot's head pose at that moment.
    """

    def __init__(self, context):
        self.context = context
        self.texture = None
        self.stamp = None
        self.program = context.program(
            vertex_shader="""#version 430
                uniform mat4 clip_from_camera;
                uniform vec4 corners;  // left, right, top, bottom at distance, camera axes
                uniform float distance;
                out vec2 uv;
                const vec2 at[4] = vec2[4](vec2(0,0), vec2(1,0), vec2(0,1), vec2(1,1));
                void main() {
                    uv = at[gl_VertexID];
                    vec2 p = vec2(mix(corners.x, corners.y, uv.x), mix(corners.z, corners.w, uv.y));
                    gl_Position = clip_from_camera * vec4(p, -distance, 1);
                }
            """,
            fragment_shader="""#version 430
                uniform sampler2D camera;
                uniform float feather;
                uniform float opacity;
                in vec2 uv;
                out vec4 color;
                void main() {
                    vec2 edge = min(uv, 1 - uv) / feather;
                    float a = opacity * smoothstep(0, 1, min(min(edge.x, edge.y), 1));
                    color = vec4(texture(camera, uv).rgb * a, a);
                }
            """,
        )
        self.vao = context.vertex_array(self.program, [])

    def draw(self, rgb, stamp, orientation, intrinsics, world_from_eye, projection, origin):
        """Blend over the current target; True when this call showed a frame not shown before.

        orientation: world-from-camera (Ito axes, looking down -Z) at the frame's capture;
        world_from_eye: the eye being drawn; origin: the world point the frame is hung around,
        the pilot's head (between the eyes) at the robot's anchored camera.
        """
        if rgb is None or orientation is None or intrinsics is None:
            return False
        opacity = 1 - (clock.now() - stamp - LIVE_HOLD) / (LIVE_GONE - LIVE_HOLD)
        if opacity <= 0:
            return False
        size = (rgb.shape[1], rgb.shape[0])
        if self.texture is None or self.texture.size != size:
            if self.texture:
                self.texture.release()
            self.texture = self.context.texture(size, 3, alignment=1)
            self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.stamp = None
        fresh = self.stamp != stamp
        if fresh:
            self.texture.write(rgb.tobytes(), alignment=1)
            self.stamp = stamp
        # The camera's own rays through its image corners.
        i, d = intrinsics, LIVE_DISTANCE
        corners = (
            -i.cx / i.fx * d,
            (i.width - i.cx) / i.fx * d,
            i.cy / i.fy * d,
            -(i.height - i.cy) / i.fy * d,
        )
        world_from_camera = np.eye(4)
        world_from_camera[:3, :3] = np.asarray(orientation)[:3, :3]
        world_from_camera[:3, 3] = np.asarray(origin)[:3, 3]
        clip = np.asarray(projection) @ np.linalg.inv(world_from_eye) @ world_from_camera
        self.context.enable_only(moderngl.BLEND)
        self.context.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        self.texture.use(0)
        self.program["camera"] = 0
        self.program["corners"] = corners
        self.program["distance"] = d
        self.program["feather"] = LIVE_FEATHER
        self.program["opacity"] = min(1.0, opacity)
        self.program["clip_from_camera"].write(np.asarray(clip, dtype="f4").T.copy().tobytes())
        self.vao.render(moderngl.TRIANGLE_STRIP, vertices=4)
        return fresh

    def close(self):
        if self.texture:
            self.texture.release()
        self.vao.release()
        self.program.release()
