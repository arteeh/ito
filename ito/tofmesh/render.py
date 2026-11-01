"""Draws the ToF mesh, textured by projecting camera frames onto it.

The mesh carries a colour per vertex (the volume's voxel colours, a few centimetres coarse:
the map's memory). On top, a handful of camera frames act as slide projectors: every fragment
looks itself up in each projector that sees it unoccluded (a distance map per projector,
rendered from the mesh) and takes the frames' colours, weighted towards the projector facing it
most squarely and nearest, so detail comes at camera resolution where a frame saw it. Slot 0 is
the live frame; the others are kept views (fusion.py chooses them).

Guessed depth (#87): the ToF sees ~45 degrees, the camera far more, so most of the live frame
has no mesh behind it. A guess places it in the world anyway, on the GPU, every time the live
frame or the mesh changes: the live projector's distance map (the mesh, i.e. everything the ToF
ever measured, seen from the live camera) has its holes filled by push-pull (an image pyramid
that averages inverse range down, then fills each hole from the coarser level up: the nearest
measured surfaces extended outwards), and a ray that meets the floor (y = 0 in the robot's
odometry frame) before that is put on the floor. The live frame is then drawn as a grid at
those ranges, in the frame's own colours like any measured surface (user, 1 Nov: no dimming or
hatching), and broken where neighbouring guesses jump (an edge). Guesses are drawn into a layer
of their own with ordinary depth (so nearer guesses hide farther ones from any eye position),
and that layer is composited only into pixels where no mesh was drawn: a guess never covers a
measured surface and gives way as soon as the ToF measures that spot.

Same draw contract as the splat renderer (ito/render/renderer.py), so the desktop and XR windows
can use either: draw(robot_camera, head, projection, target, viewport=...).
"""

import os

import moderngl
import numpy as np

from ito.render.pose import validate_pose

from .fusion import PROJECTORS

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
        // Kept views only where they face the surface; the live frame colours everything it
        // sees, however grazing (the low camera sees the floor at a shallow angle, and a bumpy
        // ToF floor tips its normals further away).
        if (k == 0) facing = abs(facing);
        else if (facing < 0.15) continue;
        float edge = clamp(min(min(u, image_size.x - u), min(v, image_size.y - v))
                           / (0.06 * image_size.x), 0.0, 1.0);
        // The floor keeps a grazing or border sample of the live frame in the blend
        // (w^6 stays a normal float), where it is the only view.
        float w = max(boost[k] * facing * facing * edge / (1.0 + 0.5 * d * d), 1e-5);
        // A steep power: the best view nearly alone, so small pose errors between views do
        // not show as double images.
        float w2 = w * w;
        float w6 = w2 * w2 * w2;
        sum += w6 * texture(images, vec3(st, float(k))).rgb;
        total += w6;
    }
    // Any frame that sees the fragment colours it fully; the voxel colours (or grey) only
    // where no frame does.
    vec3 colour = total > 0.0 ? sum / total : base;
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

# Guessed depth: one vertex per GUESS_GRID pixels of the shadow-map resolution, ranges capped at
# GUESS_FAR (beyond the ToF's 4 m nothing is known), a break where neighbouring guesses differ
# by more than GUESS_EDGE.
GUESS_GRID = (128, 72)
GUESS_FAR = 8.0
GUESS_EDGE = 1.3
FLOOR_HEIGHT = 0.0  # the odometry frame's floor
# The floor rule needs the camera above it: odometry that drifted the camera down to the floor
# would put every downward ray on the floor a few centimetres away.
FLOOR_CLEARANCE = 0.05

FULLSCREEN = """
#version 430
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

# Push: level 0 from the live projector's distance map (inverse depth along the camera axis,
# weight 1 where measured: a hole filled with a mean depth continues a surface as a plane facing
# the camera, where a mean range would bend it into a sphere around it), each next level the
# weighted mean of 2x2 texels.
PUSH = """
#version 430
uniform sampler2D source;
uniform int first;
uniform ivec2 source_size;
uniform vec4 intrinsics;  // fx, fy, cx, cy of the level-0 (shadow map) pixels
out vec2 result;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    if (first == 1) {
        float r = texelFetch(source, p, 0).r;
        // Row 0 of the map is the image's bottom row.
        vec2 q = vec2(gl_FragCoord.x - 0.5, float(source_size.y) - gl_FragCoord.y - 0.5);
        vec2 d = (q - intrinsics.zw) / intrinsics.xy;
        float z = r / length(vec3(d, 1.0));
        result = (r > 0.0 && r < 1e8) ? vec2(1.0 / z, 1.0) : vec2(0.0);
        return;
    }
    vec2 sum = vec2(0.0);
    for (int j = 0; j < 2; ++j)
        for (int i = 0; i < 2; ++i) {
            vec2 c = texelFetch(source, min(p * 2 + ivec2(i, j), source_size - 1), 0).rg;
            sum += vec2(c.x * c.y, c.y);
        }
    result = sum.y > 0.0 ? vec2(sum.x / sum.y, min(sum.y, 1.0)) : vec2(0.0);
}
"""

# Pull: each level's holes (weight < 1) filled from the coarser, already filled level.
PULL = """
#version 430
uniform sampler2D pushed;
uniform sampler2D coarse;
uniform vec2 size;
out vec2 result;
void main() {
    vec2 own = texelFetch(pushed, ivec2(gl_FragCoord.xy), 0).rg;
    vec2 c = texture(coarse, gl_FragCoord.xy / size).rg;
    result = vec2(own.x * own.y + c.x * (1.0 - own.y), max(own.y, c.y));
}
"""

GUESS_VERTEX = """
#version 430
uniform mat4 view_projection;
uniform mat3 world_from_camera;
uniform vec3 origin;
uniform vec4 intrinsics;     // fx, fy, cx, cy in image pixels
uniform vec2 image_size;
uniform sampler2D filled;    // inverse depth, holes filled (push-pull)
uniform sampler2D measured;  // weight 1 where the mesh itself is
uniform vec2 grid_step;      // one grid cell in image uv
uniform float floor_height;
uniform float floor_clearance;
uniform float far;
uniform float edge;
in vec2 uv;                  // over the image, v down
out vec2 v_uv;
out vec3 v_world;
out float v_keep;
out float v_measured;

vec3 camera_ray(vec2 st) {  // not normalized: z = -1
    vec2 p = st * image_size - 0.5;
    return vec3((p.x - intrinsics.z) / intrinsics.x, -(p.y - intrinsics.w) / intrinsics.y, -1.0);
}

vec3 ray(vec2 st) {
    return world_from_camera * normalize(camera_ray(st));
}

float range_at(vec2 st) {
    vec3 d = ray(st);
    float inv = texture(filled, vec2(st.x, 1.0 - st.y)).r;
    float r = inv > 1.0 / far ? length(camera_ray(st)) / inv : far;
    float height = origin.y - floor_height;
    if (height > floor_clearance && d.y < -1e-3) r = min(r, height / -d.y);
    return clamp(r, 0.1, far);
}

void main() {
    float r = range_at(uv);
    float lo = r, hi = r;
    for (int k = 0; k < 4; ++k) {
        vec2 o = vec2(k == 0 ? 1.0 : k == 1 ? -1.0 : 0.0, k == 2 ? 1.0 : k == 3 ? -1.0 : 0.0);
        float n = range_at(clamp(uv + o * grid_step, 0.0, 1.0));
        lo = min(lo, n);
        hi = max(hi, n);
    }
    v_keep = hi <= lo * edge ? 1.0 : 0.0;
    v_measured = texture(measured, vec2(uv.x, 1.0 - uv.y)).g;
    v_uv = uv;
    v_world = origin + ray(uv) * r;
    gl_Position = view_projection * vec4(v_world, 1.0);
}
"""

GUESS_FRAGMENT = """
#version 430
uniform sampler2DArray images;
in vec2 v_uv;
in vec3 v_world;
in float v_keep;
in float v_measured;
out vec4 f_colour;

void main() {
    if (v_keep < 0.999 || v_measured > 0.999) discard;
    // Alpha 1: this pixel holds a guess (the composite reads it).
    f_colour = vec4(texture(images, vec3(v_uv, 0.0)).rgb, 1.0);
}
"""

# The guess layer into the view: only where the view's depth is still the clear value, i.e. no
# mesh was drawn there (depth test <= at depth 1.0), and only where the layer holds a guess.
COMPOSITE_VERTEX = """
#version 430
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 1.0, 1.0);
}
"""

COMPOSITE_FRAGMENT = """
#version 430
uniform sampler2D layer;
out vec4 f_colour;
void main() {
    vec4 c = texelFetch(layer, ivec2(gl_FragCoord.xy), 0);
    if (c.a < 0.5) discard;
    f_colour = vec4(c.rgb, 1.0);
    gl_FragDepth = 1.0;
}
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


def _guess_setting():
    """On unless ITO_GUESS_DEPTH=0. (Not an app setting: a settings file with a field the main
    edition does not know would make main builds drop the user's settings.)"""
    return os.environ.get("ITO_GUESS_DEPTH", "1").lower() not in {"0", "off", "false", "no"}


def _set(program, **values):
    for name, value in values.items():
        if name in program:  # the compiler drops unused uniforms
            program[name] = value


class MeshRenderer:
    def __init__(
        self, context: moderngl.Context, slots=PROJECTORS, image_size=(640, 360), guess=None
    ):
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
        self.epoch = None  # the worker run whose mesh and projectors are shown
        # Kept views whose distance map is older than the mesh: refreshed one per drawn frame,
        # so a mesh update (~12.7 a second) redraws only the live view's map at once.
        self.shadow_stale = set()
        self.guess_enabled = _guess_setting() if guess is None else bool(guess)
        self.guess_dirty = False
        self.levels = []
        self.push_program = self.pull_program = self.push_vao = self.pull_vao = None
        self.guess_program = self.guess_vbo = self.guess_ibo = self.guess_vao = None
        self.composite_program = self.composite_vao = None
        self.guess_target = None  # (size, framebuffer, colour texture, depth)
        if self.guess_enabled:  # ITO_GUESS_DEPTH=0: none of its GPU resources
            self._guess_resources()

    def _guess_resources(self):
        """Push-pull pyramid over the live projector's distance map, the guess grid and the
        program that composites the guess layer."""
        context = self.context
        self.composite_program = context.program(
            vertex_shader=COMPOSITE_VERTEX, fragment_shader=COMPOSITE_FRAGMENT
        )
        self.composite_vao = context.vertex_array(self.composite_program, [])
        self.push_program = context.program(vertex_shader=FULLSCREEN, fragment_shader=PUSH)
        self.pull_program = context.program(vertex_shader=FULLSCREEN, fragment_shader=PULL)
        self.push_vao = context.vertex_array(self.push_program, [])
        self.pull_vao = context.vertex_array(self.pull_program, [])
        self.levels = []  # (size, pushed texture, its fbo, pulled texture, its fbo)
        w, h = SHADOW_SIZE
        while True:
            level = [(w, h)]
            for _ in range(2):
                texture = context.texture((w, h), 2, dtype="f4")
                texture.filter = moderngl.LINEAR, moderngl.LINEAR
                texture.repeat_x = texture.repeat_y = False
                level += [texture, context.framebuffer([texture])]
            self.levels.append(level)
            if w == 1 and h == 1:
                break
            w, h = (w + 1) // 2, (h + 1) // 2
        self.guess_program = context.program(
            vertex_shader=GUESS_VERTEX, fragment_shader=GUESS_FRAGMENT
        )
        gw, gh = GUESS_GRID
        u, v = np.meshgrid(np.linspace(0, 1, gw + 1), np.linspace(0, 1, gh + 1))
        uv = np.stack([u, v], -1).astype(np.float32).reshape(-1, 2)
        i = np.arange(gh)[:, None] * (gw + 1) + np.arange(gw)[None, :]
        quads = np.stack([i, i + 1, i + gw + 1, i + 1, i + gw + 2, i + gw + 1], -1)
        self.guess_vbo = context.buffer(uv.tobytes())
        self.guess_ibo = context.buffer(quads.astype(np.uint32).tobytes())
        self.guess_vao = context.vertex_array(
            self.guess_program, [(self.guess_vbo, "2f", "uv")], self.guess_ibo,
            index_element_size=4,
        )  # fmt: skip

    # -- updates -------------------------------------------------------------------------
    def upload(self, frame):
        """Apply one fusion.MeshFrame."""
        if frame.epoch != self.epoch:
            # A new worker (restart, reconnect): its map starts empty in its own odometry frame,
            # so nothing of the old one may stay, mesh or projected views.
            self.epoch = frame.epoch
            self.count = 0
            self.projectors = [None] * self.slots
            self.shadow_dirty.clear()
            self.shadow_stale.clear()
        if frame.positions is not None:
            self._mesh(frame.positions, frame.normals, frame.colours, frame.triangles)
            if self.projectors[0] is not None:
                self.shadow_dirty.add(0)
            self.shadow_stale.update(
                k for k, p in enumerate(self.projectors) if k and p is not None
            )
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
        self.shadow_stale -= self.shadow_dirty
        if self.shadow_stale:
            self.shadow_dirty.add(self.shadow_stale.pop())
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
        if 0 in self.shadow_dirty:
            self.guess_dirty = True
        self.shadow_dirty.clear()

    def _guess_pyramid(self):
        """Fill the holes of the live projector's distance map (slot 0 of the shadow atlas)."""
        if not self.guess_dirty:
            return
        self.guess_dirty = False
        self.context.enable_only(moderngl.NOTHING)
        self.context.scissor = None
        _, (fx, fy, cx, cy), _ = self.projectors[0]
        sx, sy = SHADOW_SIZE[0] / self.image_size[0], SHADOW_SIZE[1] / self.image_size[1]
        intrinsics = (fx * sx, fy * sy, (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5)
        source, source_size = self.shadows, SHADOW_SIZE
        for index, (size, pushed, fbo, _, _) in enumerate(self.levels):
            fbo.use()
            self.context.viewport = (0, 0, *size)
            source.use(0)
            _set(
                self.push_program, source=0, first=int(index == 0), source_size=source_size,
                intrinsics=intrinsics,
            )  # fmt: skip
            self.push_vao.render(moderngl.TRIANGLES, vertices=3)
            source, source_size = pushed, size
        coarse = self.levels[-1][1]  # the top level's pull is its push
        for size, pushed, _, pulled, fbo in reversed(self.levels):
            fbo.use()
            self.context.viewport = (0, 0, *size)
            pushed.use(0)
            coarse.use(1)
            _set(self.pull_program, pushed=0, coarse=1, size=size)
            self.pull_vao.render(moderngl.TRIANGLES, vertices=3)
            coarse = pulled

    def _draw_guess(self, view_projection):
        camera_from_world, (fx, fy, cx, cy), origin = self.projectors[0]
        world_from_camera = np.linalg.inv(camera_from_world)
        program = self.guess_program
        program["view_projection"].write(view_projection.T.astype(np.float32).tobytes())
        program["world_from_camera"].write(
            np.ascontiguousarray(world_from_camera[:3, :3].T, np.float32).tobytes()
        )
        self.images.use(0)
        self.levels[0][3].use(2)  # filled
        self.levels[0][1].use(3)  # measured
        _set(
            program,
            origin=tuple(float(x) for x in origin),
            intrinsics=(fx, fy, cx, cy),
            image_size=self.image_size,
            grid_step=(1 / GUESS_GRID[0], 1 / GUESS_GRID[1]),
            floor_height=FLOOR_HEIGHT,
            far=GUESS_FAR,
            edge=GUESS_EDGE,
            images=0,
            filled=2,
            measured=3,
            floor_clearance=FLOOR_CLEARANCE,
        )
        self.guess_vao.render(moderngl.TRIANGLES)

    def _guess_layer(self, size):
        if self.guess_target is None or self.guess_target[0] != size:
            if self.guess_target is not None:
                for r in self.guess_target[1:]:
                    r.release()
            colour = self.context.texture(size, 4)
            colour.filter = moderngl.NEAREST, moderngl.NEAREST
            depth = self.context.depth_renderbuffer(size)
            self.guess_target = (size, self.context.framebuffer([colour], depth), colour, depth)
        return self.guess_target

    def _guesses(self, view_projection, fbo, size):
        """Draw the guesses into their own layer (ordinary depth among themselves), then into
        `fbo` where it still has its clear depth: wherever no mesh was drawn."""
        _, layer, colour, _ = self._guess_layer(size)
        layer.use()
        self.context.viewport = (0, 0, *size)
        self.context.enable_only(moderngl.DEPTH_TEST)
        layer.clear(0, 0, 0, 0, depth=1.0)
        self._draw_guess(view_projection)
        fbo.use()
        self.context.viewport = (0, 0, *size)
        self.context.depth_func = "<="
        colour.use(0)
        _set(self.composite_program, layer=0)
        self.composite_vao.render(moderngl.TRIANGLES, vertices=3)
        self.context.depth_func = "<"

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
        guess = self.guess_enabled and self.projection_enabled and self.projectors[0] is not None
        if guess:
            self._guess_pyramid()
        size = tuple(int(v) for v in viewport[2:])
        fbo = self._offscreen(size)
        fbo.use()
        self.context.viewport = (0, 0, *size)
        self.context.scissor = None
        self.context.enable_only(moderngl.DEPTH_TEST)
        fbo.clear(*(clear or (0, 0, 0, 1)), depth=1.0)
        view_projection = projection @ _view(world_from_eye)
        if self.count and self.vao is not None:
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
                boost[k] = 2.0 if k == 0 else 1.0
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
        if guess:
            self._guesses(view_projection, fbo, size)
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
            self.push_vao, self.pull_vao, self.push_program, self.pull_program, self.guess_vao,
            self.guess_vbo, self.guess_ibo, self.guess_program, self.composite_vao,
            self.composite_program,
            *(r for level in self.levels for r in level[1:]),
            *(self.guess_target[1:] if self.guess_target is not None else ()),
        ):  # fmt: skip
            if r is not None:
                r.release()
        if self.target is not None:
            for r in self.target[1:]:
                r.release()
