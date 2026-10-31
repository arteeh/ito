"""Draws the ToF mesh, textured by projecting camera frames onto it.

The mesh carries a colour per vertex (the volume's voxel colours, a few centimetres coarse:
the map's memory). On top, a handful of camera frames act as slide projectors: every fragment
looks itself up in each projector that sees it unoccluded (a distance map per projector,
rendered from the mesh) and takes the frames' colours, weighted towards the projector facing it
most squarely and nearest, so detail comes at camera resolution where a frame saw it. Slot 0 is
the live frame; the others are kept views (fusion.py chooses them).

Same draw contract as the splat renderer (ito/render/renderer.py), so the desktop and XR windows
can use either: draw(robot_camera, head, projection, target, viewport=...).
"""

import moderngl
import numpy as np

from ito.render.pose import validate_pose

SHADOW_SIZE = (256, 144)
NEAR, FAR = 0.05, 30.0

VERTEX = """
#version 430
uniform mat4 view_projection;
in vec3 position;
in vec3 normal;
in vec4 colour;
out vec3 v_world;
out vec3 v_normal;
out vec4 v_colour;
void main() {
    v_world = position;
    v_normal = normal;
    v_colour = colour;
    gl_Position = view_projection * vec4(position, 1.0);
}
"""

FRAGMENT = """
#version 430
#define K %(K)d
uniform sampler2DArray images;
uniform sampler2D shadows;
uniform mat4 camera_from_world[K];
uniform vec4 intrinsics[K];      // fx, fy, cx, cy in image pixels
uniform vec3 origin[K];
uniform float boost[K];          // 0: empty slot
uniform vec2 image_size;
uniform float projection;        // 0: vertex colours only
uniform vec3 eye;
in vec3 v_world;
in vec3 v_normal;
in vec4 v_colour;
out vec4 f_colour;

void main() {
    vec3 n = normalize(v_normal);
    // Normals point to free space; a surface seen from behind shows its back the same colour.
    if (dot(n, eye - v_world) < 0.0) n = -n;
    vec3 base;
    if (v_colour.a > 0.5) {
        base = v_colour.rgb;
    } else {
        // Never coloured by a camera frame: plain grey, shaded so its shape reads.
        float light = 0.5 + 0.35 * max(dot(n, normalize(vec3(0.3, 1.0, 0.2))), 0.0);
        base = vec3(0.5) * light;
    }
    vec3 sum = vec3(0.0);
    float total = 0.0;
    float best = 0.0;
    for (int k = 0; k < K; ++k) {
        if (boost[k] <= 0.0 || projection <= 0.0) continue;
        vec4 c = camera_from_world[k] * vec4(v_world, 1.0);
        float z = -c.z;
        if (z < 0.05) continue;
        vec4 q = intrinsics[k];
        float u = q.z + q.x * c.x / z;
        float v = q.w - q.y * c.y / z;
        if (u < 0.0 || v < 0.0 || u > image_size.x - 1.0 || v > image_size.y - 1.0) continue;
        vec2 st = vec2((u + 0.5) / image_size.x, (v + 0.5) / image_size.y);
        float d = length(v_world - origin[k]);
        vec2 atlas = vec2((float(k) + st.x) / float(K), 1.0 - st.y);
        float nearest = texture(shadows, atlas).r;
        if (d > nearest + 0.06 + 0.03 * d) continue;
        vec3 toward = (origin[k] - v_world) / max(d, 1e-4);
        float facing = dot(n, toward);
        if (facing < 0.15) continue;
        float edge = clamp(min(min(u, image_size.x - u), min(v, image_size.y - v))
                           / (0.06 * image_size.x), 0.0, 1.0);
        float w = boost[k] * facing * facing * edge / (1.0 + 0.5 * d * d);
        float w3 = w * w * w;
        sum += w3 * texture(images, vec3(st, float(k))).rgb;
        total += w3;
        best = max(best, w);
    }
    vec3 colour = base;
    if (total > 0.0) colour = mix(base, sum / total, clamp(best * 6.0, 0.0, 1.0));
    f_colour = vec4(colour, 1.0);
}
"""

SHADOW_VERTEX = """
#version 430
uniform mat4 world_to_clip;
uniform vec3 origin;
in vec3 position;
out vec3 v_world;
void main() {
    v_world = position;
    gl_Position = world_to_clip * vec4(position, 1.0);
}
"""

SHADOW_FRAGMENT = """
#version 430
uniform vec3 origin;
in vec3 v_world;
out float range_out;
void main() { range_out = length(v_world - origin); }
"""


def _view(world_from_eye):
    rotation = world_from_eye[:3, :3]
    view = np.eye(4)
    view[:3, :3] = rotation.T
    view[:3, 3] = -rotation.T @ world_from_eye[:3, 3]
    return view


def pinhole_projection(fx, fy, cx, cy, width, height, near=NEAR, far=FAR):
    """OpenGL clip from camera (Ito axes) for an image whose row 0 is the top."""
    p = np.zeros((4, 4))
    p[0, 0] = 2 * fx / width
    p[0, 2] = -((2 * cx + 1) / width - 1)
    p[1, 1] = 2 * fy / height
    p[1, 2] = -(1 - (2 * cy + 1) / height)
    p[2, 2] = -(far + near) / (far - near)
    p[2, 3] = -2 * far * near / (far - near)
    p[3, 2] = -1
    return p


class MeshRenderer:
    def __init__(self, context: moderngl.Context, slots=12, image_size=(640, 360)):
        self.context = context
        self.slots = slots
        self.image_size = tuple(image_size)
        self.program = context.program(
            vertex_shader=VERTEX, fragment_shader=FRAGMENT % {"K": slots}
        )
        self.shadow_program = context.program(
            vertex_shader=SHADOW_VERTEX, fragment_shader=SHADOW_FRAGMENT
        )
        self.images = context.texture_array((*self.image_size, slots), 3, dtype="f1")
        self.images.filter = moderngl.LINEAR, moderngl.LINEAR
        self.images.repeat_x = self.images.repeat_y = False
        w, h = SHADOW_SIZE
        self.shadows = context.texture((w * slots, h), 1, dtype="f4")
        self.shadows.filter = moderngl.NEAREST, moderngl.NEAREST
        self.shadow_depth = context.depth_renderbuffer((w * slots, h))
        self.shadow_fbo = context.framebuffer([self.shadows], self.shadow_depth)
        self.shadow_fbo.clear(1e9, 0, 0, 0)
        self.projectors = [None] * slots  # (camera_from_world, intrinsics, origin)
        self.shadow_dirty = set()
        self.vbo = self.nbo = self.cbo = self.ibo = None
        self.vao = self.shadow_vao = None
        self.capacity = (0, 0)
        self.count = 0  # triangles
        self.sorts = 0
        self.draw_count = 0
        self.target = None
        self.projection_enabled = True

    # -- updates -------------------------------------------------------------------------
    def upload(self, frame):
        """Apply one fusion.MeshFrame."""
        if frame.positions is not None:
            self._mesh(frame.positions, frame.normals, frame.colours, frame.triangles)
            self.shadow_dirty.update(k for k, p in enumerate(self.projectors) if p is not None)
        for p in frame.projectors:
            self._projector(p)
        return True

    apply = upload

    def _mesh(self, positions, normals, colours, triangles):
        n, t = len(positions), len(triangles)
        if n > self.capacity[0] or t > self.capacity[1]:
            for b in (self.vbo, self.nbo, self.cbo, self.ibo, self.vao, self.shadow_vao):
                if b is not None:
                    b.release()
            cap = (max(1024, int(n * 1.5)), max(1024, int(t * 1.5)))
            self.vbo = self.context.buffer(reserve=cap[0] * 12)
            self.nbo = self.context.buffer(reserve=cap[0] * 12)
            self.cbo = self.context.buffer(reserve=cap[0] * 4)
            self.ibo = self.context.buffer(reserve=cap[1] * 12)
            self.vao = self.context.vertex_array(
                self.program,
                [
                    (self.vbo, "3f", "position"),
                    (self.nbo, "3f", "normal"),
                    (self.cbo, "4f1", "colour"),
                ],
                self.ibo,
                index_element_size=4,
            )
            self.shadow_vao = self.context.vertex_array(
                self.shadow_program, [(self.vbo, "3f", "position")], self.ibo, index_element_size=4
            )
            self.capacity = cap
        if n:
            self.vbo.write(np.ascontiguousarray(positions, np.float32))
            self.nbo.write(np.ascontiguousarray(normals, np.float32))
            self.cbo.write(np.ascontiguousarray(colours, np.uint8))
            self.ibo.write(np.ascontiguousarray(triangles, np.uint32))
        self.count = t

    def _projector(self, p):
        fx, fy, cx, cy, w, h = p.intrinsics
        if p.image is not None:
            image = p.image
            if (w, h) != self.image_size:
                # Rescale intrinsics to the slot size; nearest resampling is enough here.
                sx, sy = self.image_size[0] / w, self.image_size[1] / h
                ys = np.minimum((np.arange(self.image_size[1]) / sy).astype(int), h - 1)
                xs = np.minimum((np.arange(self.image_size[0]) / sx).astype(int), w - 1)
                image = image[ys][:, xs]
                fx, fy = fx * sx, fy * sy
                cx, cy = (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5
            self.images.write(
                np.ascontiguousarray(image, np.uint8).tobytes(),
                viewport=(0, 0, p.slot, *self.image_size, 1),
            )
        else:
            sx, sy = self.image_size[0] / w, self.image_size[1] / h
            fx, fy = fx * sx, fy * sy
            cx, cy = (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5
        world_from_camera = np.asarray(p.world_from_camera, np.float64)
        self.projectors[p.slot] = (
            np.linalg.inv(world_from_camera),
            (fx, fy, cx, cy),
            world_from_camera[:3, 3].copy(),
        )
        self.shadow_dirty.add(p.slot)

    def _shadows(self):
        if not self.shadow_dirty:
            return
        self.shadow_fbo.use()
        self.context.enable_only(moderngl.DEPTH_TEST)
        w, h = SHADOW_SIZE
        for k in sorted(self.shadow_dirty):
            self.context.viewport = (k * w, 0, w, h)
            self.context.scissor = (k * w, 0, w, h)
            self.shadow_fbo.clear(1e9, 0, 0, 0, depth=1.0, viewport=(k * w, 0, w, h))
            entry = self.projectors[k]
            if entry is None or self.shadow_vao is None or not self.count:
                continue
            camera_from_world, (fx, fy, cx, cy), origin = entry
            clip = pinhole_projection(fx, fy, cx, cy, *self.image_size) @ camera_from_world
            self.shadow_program["world_to_clip"].write(clip.T.astype(np.float32).tobytes())
            self.shadow_program["origin"] = tuple(origin)
            self.shadow_vao.render(moderngl.TRIANGLES, vertices=self.count * 3)
        self.context.scissor = None
        self.shadow_dirty.clear()

    # -- drawing -------------------------------------------------------------------------
    def sort(self, robot_camera, head):
        """Meshes need no sorting; kept for the windows' splat-renderer contract."""

    def _offscreen(self, size):
        if self.target is None or self.target[0] != size:
            if self.target is not None:
                for r in self.target[1:]:
                    r.release()
            colour = self.context.renderbuffer(size, 4)
            depth = self.context.depth_renderbuffer(size)
            self.target = (size, self.context.framebuffer([colour], depth), colour, depth)
        return self.target[1]

    def draw(
        self,
        robot_camera,
        head,
        projection,
        target=None,
        *,
        clear=(0.025, 0.035, 0.055, 1.0),
        viewport=None,
        sort=True,
    ):
        world_from_eye = validate_pose(robot_camera) @ validate_pose(head)
        projection = np.asarray(projection, dtype=np.float64)
        if projection.shape != (4, 4) or not np.isfinite(projection).all():
            raise ValueError("Projection must be a finite 4x4 matrix")
        target = target if target is not None else self.context.screen
        if target is None:
            raise ValueError("An offscreen context needs an explicit framebuffer")
        viewport = viewport if viewport is not None else (0, 0, *target.size)
        if min(viewport[2:]) <= 0:
            raise ValueError("Viewport dimensions must be positive")
        self._shadows()
        size = tuple(int(v) for v in viewport[2:])
        fbo = self._offscreen(size)
        fbo.use()
        self.context.viewport = (0, 0, *size)
        self.context.scissor = None
        self.context.enable_only(moderngl.DEPTH_TEST)
        fbo.clear(*(clear or (0, 0, 0, 1)), depth=1.0)
        if self.count and self.vao is not None:
            view_projection = projection @ _view(world_from_eye)
            program = self.program
            program["view_projection"].write(view_projection.T.astype(np.float32).tobytes())
            program["eye"] = tuple(world_from_eye[:3, 3])
            program["image_size"] = self.image_size
            program["projection"] = 1.0 if self.projection_enabled else 0.0
            matrices = np.zeros((self.slots, 4, 4), np.float32)
            intrinsics = np.zeros((self.slots, 4), np.float32)
            origins = np.zeros((self.slots, 3), np.float32)
            boost = np.zeros(self.slots, np.float32)
            for k, entry in enumerate(self.projectors):
                if entry is None:
                    continue
                matrices[k] = entry[0].T
                intrinsics[k] = entry[1]
                origins[k] = entry[2]
                boost[k] = 1.5 if k == 0 else 1.0
            program["camera_from_world"].write(matrices.tobytes())
            program["intrinsics"].write(intrinsics.tobytes())
            program["origin"].write(origins.tobytes())
            program["boost"].write(boost.tobytes())
            self.images.use(0)
            self.shadows.use(1)
            program["images"] = 0
            program["shadows"] = 1
            self.vao.render(moderngl.TRIANGLES, vertices=self.count * 3)
            self.draw_count += 1
        x, y, w, h = viewport
        self.context.copy_framebuffer(target, fbo) if (
            (x, y) == (0, 0) and tuple(target.size) == size
        ) else self._blit(fbo, target, viewport)
        target.use()
        self.context.viewport = viewport
        self.context.enable_only(moderngl.BLEND)

    def _blit(self, source, target, viewport):
        from OpenGL import GL

        x, y, w, h = viewport
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, source.glo)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, target.glo)
        GL.glBlitFramebuffer(0, 0, w, h, x, y, x + w, y + h, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def close(self):
        for r in (
            self.vbo, self.nbo, self.cbo, self.ibo, self.vao, self.shadow_vao, self.program,
            self.shadow_program, self.images, self.shadows, self.shadow_depth, self.shadow_fbo,
        ):  # fmt: skip
            if r is not None:
                r.release()
        if self.target is not None:
            for r in self.target[1:]:
                r.release()
