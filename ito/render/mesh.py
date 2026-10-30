"""A world of depth meshes, no Gaussians: the current frame and a memory of the room.

NOW: the newest frame with depth, drawn as a mesh of its depth image at its pose and textured
with its own pixels. It is sharp the moment it arrives; nothing is trained.

MEMORY (ITO_MESH_MEMORY):
- tsdf: every frame fused into one coloured signed distance field on the GPU (a dense box of
  voxels). Free space a new frame sees clears old surfaces there, and a voxel's colour goes to
  the newest, closest, most head-on view (an old view's claim fades with age).
- ibr: the field's surfaces, coloured from the stored keyframe images instead (the best view
  that sees the point: head-on, close, recent), so they stay as sharp as the frames.
- atlas: every keyframe's own textured depth mesh; a newer keyframe retires the parts of older
  ones it sees again (they lie in its free space or on its surface).
- none: NOW only.

Frames carry axial depth in metres and a world-from-camera pose (Ito axes: x right, y up,
looking down -z). Everything runs in GL 4.3 (compute shaders), in the window's context, so the
same class draws the desktop window, each VR eye, and offline videos. draw() renders into its
own framebuffer and composites onto the target with premultiplied alpha: over splats, or alone.
"""

import math
import os
from dataclasses import dataclass

import moderngl
import numpy as np

from ito.meshout import MeshFrame

MEMORY = os.environ.get("ITO_MESH_MEMORY", "tsdf")
VOXEL = float(os.environ.get("ITO_MESH_VOXEL", "0.02"))  # metres
# The field's box around the first frame's camera: half width (x, z) and height below/above, m.
BOUNDS = tuple(float(v) for v in os.environ.get("ITO_MESH_BOUNDS", "4,1.2,2.2").split(","))
KEYFRAMES = int(os.environ.get("ITO_MESH_KEYFRAMES", "48"))
# Seconds after which an old view's claim on a voxel's colour has faded to 1/e of what it was.
COLOR_AGE = float(os.environ.get("ITO_MESH_COLOR_AGE", "30"))
# A triangle whose corners' depths span more than JUMP[0] + JUMP[1] * nearest is a silhouette
# (foreground and background joined), not a surface: never drawn, never fused.
JUMP = (0.02, 0.06)
# A pixel whose inverse depth bends by more than this share against its neighbours' is a
# silhouette (see bend() below).
EDGE_BEND = float(os.environ.get("ITO_MESH_EDGE", "0.08"))
NOW_BIAS = 0.03  # NOW wins over memory within this share of its depth
# A frame becomes a memory view (keyframe) once it moved this far (m) or turned (degrees).
KEY_MOVE = (0.15, 12.0)
KEY_RGB = (640, 360)  # keyframe texture size (width, height)
KEY_DEPTH = (256, 144)


EDGE = """
// A silhouette pixel: its inverse depth bends sharply against its neighbours'. A plane's inverse
// depth is affine in the image, so a floor seen at a grazing angle (depth climbing fast from
// pixel to pixel) still counts as a surface; foreground against background does not.
float bend(sampler2D depth, ivec2 q) {
    ivec2 last = textureSize(depth, 0) - 1;
    float d = texelFetch(depth, q, 0).r;
    float l = texelFetch(depth, clamp(q - ivec2(1, 0), ivec2(0), last), 0).r;
    float r = texelFetch(depth, clamp(q + ivec2(1, 0), ivec2(0), last), 0).r;
    float u = texelFetch(depth, clamp(q - ivec2(0, 1), ivec2(0), last), 0).r;
    float b = texelFetch(depth, clamp(q + ivec2(0, 1), ivec2(0), last), 0).r;
    if (min(min(d, l), min(r, min(u, b))) < 0.05) return 1e3;
    return max(abs(1.0 / l + 1.0 / r - 2.0 / d), abs(1.0 / u + 1.0 / b - 2.0 / d)) * d;
}
"""

GRID_VERTEX = (
    """
#version 430
uniform sampler2D depth;
uniform vec4 calibration;   // fx fy cx cy, depth pixels
uniform mat4 world_from_camera;
uniform mat4 view;
out VS { vec3 uvz; vec3 view_pos; float z; float bend; } v;
"""
    + EDGE
    + """
void main() {
    ivec2 size = textureSize(depth, 0);
    ivec2 cell = ivec2(gl_VertexID % size.x, gl_VertexID / size.x);
    float z = texelFetch(depth, cell, 0).r;
    vec2 pixel = vec2(cell) + 0.5;
    vec3 local = vec3((pixel.x - calibration.z) / calibration.x,
                      (calibration.w - pixel.y) / calibration.y, -1.0) * z;
    vec4 world = world_from_camera * vec4(local, 1.0);
    vec2 uv = pixel / vec2(size);
    v.uvz = vec3(uv * z, z);
    v.view_pos = (view * world).xyz;
    v.z = z;
    v.bend = bend(depth, cell);
    gl_Position = vec4(0);
}
"""
)

GRID_GEOMETRY = """
#version 430
layout(triangles) in;
layout(triangle_strip, max_vertices=3) out;
in VS { vec3 uvz; vec3 view_pos; float z; float bend; } v[];
out vec3 uvz;
out vec3 view_pos;
uniform mat4 projection;
uniform float edge;
void main() {
    float hi = max(v[0].z, max(v[1].z, v[2].z));
    if (hi > 30.0 || isnan(hi) || max(v[0].bend, max(v[1].bend, v[2].bend)) > edge) return;
    for (int k = 0; k < 3; k++) {
        uvz = v[k].uvz;
        view_pos = v[k].view_pos;
        gl_Position = projection * vec4(v[k].view_pos, 1.0);
        EmitVertex();
    }
    EndPrimitive();
}
"""

GRID_FRAGMENT = """
#version 430
in vec3 uvz;
in vec3 view_pos;
uniform sampler2D rgb;
uniform usampler2DArray alive;  // atlas: what newer keyframes have not retired
uniform int layer;              // -1: no mask (NOW)
uniform mat4 projection;
uniform float bias;             // NOW: pulled toward the eye by this share of its depth
uniform float feather;          // NOW: alpha falls to 0 over this share at the image border
out vec4 color;
void main() {
    vec2 uv = uvz.xy / uvz.z;
    if (layer >= 0 && texture(alive, vec3(uv, layer)).r == 0u) discard;
    float a = 1.0;
    if (feather > 0.0) {
        vec2 edge = min(uv, 1.0 - uv);
        a = smoothstep(0.0, feather, min(edge.x, edge.y * 16.0 / 9.0));
    }
    vec3 c = texture(rgb, uv).rgb;
    color = vec4(c * a, a);
    vec4 clip = projection * vec4(view_pos * (1.0 - bias), 1.0);
    gl_FragDepth = 0.5 * clip.z / clip.w + 0.5;
}
"""

INTEGRATE = (
    """
#version 430
layout(local_size_x=8, local_size_y=8, local_size_z=8) in;
layout(rg16f, binding=0) uniform image3D field;    // distance / truncation x weight, weight
layout(rgba8, binding=1) uniform image3D colors;   // rgb, quality of the view that set it
layout(r32f, binding=2) uniform image3D stamps;    // when that view was captured
uniform sampler2D depth;
uniform sampler2D rgb;
uniform ivec3 offset;
uniform ivec3 extent;
uniform vec3 origin;
uniform float voxel;
uniform mat4 camera_from_world;
uniform vec4 calibration;
uniform float edge;
uniform float now;
uniform float color_age;
uniform float max_weight;
"""
    + EDGE
    + """
void main() {
    if (any(greaterThanEqual(gl_GlobalInvocationID, uvec3(extent)))) return;
    ivec3 c = offset + ivec3(gl_GlobalInvocationID);
    vec3 world = origin + (vec3(c) + 0.5) * voxel;
    vec3 p = (camera_from_world * vec4(world, 1.0)).xyz;
    float range = -p.z;
    if (range < 0.05) return;
    vec2 size = vec2(textureSize(depth, 0));
    vec2 pixel = vec2(calibration.x * p.x / range + calibration.z,
                      calibration.w - calibration.y * p.y / range);
    if (any(lessThan(pixel, vec2(1.0))) || any(greaterThan(pixel, size - 1.0))) return;
    ivec2 q = ivec2(pixel);
    float d = texelFetch(depth, q, 0).r;
    if (d < 0.05) return;
    float dr = texelFetch(depth, q + ivec2(1, 0), 0).r;
    float dd = texelFetch(depth, q + ivec2(0, 1), 0).r;
    // A silhouette pixel lies between foreground and background: it neither places a surface
    // nor clears space (it would carve the foreground object).
    if (bend(depth, q) > edge) return;
    float trunc = clamp(0.04 * d, 2.0 * voxel, 0.12);
    float sdf = d - range;
    if (sdf < -trunc) return;
    vec2 tangent = (pixel - calibration.zw) / calibration.xy;
    float radial = 1.0 / (1.0 + dot(tangent, tangent));      // cos^2 of the ray to the axis
    float w = radial * radial * min(1.0, 1.0 / (d * d));
    // Stored weighted (distance x weight, weight): interpolated, a voxel nobody saw adds
    // nothing, instead of a false empty sample just behind a surface.
    vec2 old = imageLoad(field, c).rg;
    float t = min(1.0, sdf / trunc);
    float W = min(old.g + w, max_weight);
    imageStore(field, c, vec4((old.r + t * w) / (old.g + w) * W, W, 0, 0));
    if (abs(sdf) > 1.5 * voxel) return;
    // Colour: head-on, near and recent views win.
    vec3 pc = vec3((pixel.x - calibration.z) / calibration.x,
                   (calibration.w - pixel.y) / calibration.y, -1.0) * d;
    vec3 px = vec3((pixel.x + 1.0 - calibration.z) / calibration.x,
                   (calibration.w - pixel.y) / calibration.y, -1.0) * dr;
    vec3 py = vec3((pixel.x - calibration.z) / calibration.x,
                   (calibration.w - pixel.y - 1.0) / calibration.y, -1.0) * dd;
    vec3 n = normalize(cross(px - pc, py - pc));
    float facing = abs(dot(n, normalize(pc)));
    float quality = clamp(facing * facing * radial * min(1.0, 0.8 / d), 0.004, 1.0);
    vec4 oc = imageLoad(colors, c);
    float claim = oc.a * exp(-(now - imageLoad(stamps, c).r) / color_age);
    float share = quality / (quality + claim + 1e-4);
    share = quality >= claim ? max(share, 0.6) : share * 0.5;
    vec3 seen = texture(rgb, pixel / size).rgb;
    imageStore(colors, c, vec4(mix(oc.rgb, seen, share), max(mix(claim, quality, share), 0.01)));
    if (share >= 0.5) imageStore(stamps, c, vec4(now));
}
"""
)

FULLSCREEN_VERTEX = """
#version 430
out vec2 ndc;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2) * 2.0 - 1.0;
    ndc = p;
    gl_Position = vec4(p, 0.0, 1.0);
}
"""

RAYCAST = """
#version 430
in vec2 ndc;
uniform sampler3D field;
uniform sampler3D colors;
uniform mat4 world_from_clip;
uniform mat4 view_projection;
uniform vec3 eye;
uniform vec3 origin;
uniform vec3 box;          // metres
uniform float voxel;
uniform float min_weight;
uniform int keys;          // ibr: how many keyframes to try (0: field colour)
uniform sampler2DArray key_rgb;
uniform sampler2DArray key_depth;
uniform float now;
struct Key { mat4 camera_from_world; vec4 calibration; vec4 info; };
layout(std430, binding=5) readonly buffer Keys { Key key[]; };
layout(std430, binding=6) readonly buffer Order { int order[]; };
out vec4 color;

vec2 at(vec3 world) {
    vec2 s = texture(field, (world - origin) / box).rg;
    return vec2(s.g > 1e-4 ? s.r / s.g : 1.0, s.g);
}

void main() {
    vec4 far = world_from_clip * vec4(ndc, 1.0, 1.0);
    vec3 dir = normalize(far.xyz / far.w - eye);
    vec3 inv = 1.0 / dir;
    vec3 t0 = (origin - eye) * inv, t1 = (origin + box - eye) * inv;
    vec3 tn = min(t0, t1), tf = max(t0, t1);
    float t = max(max(tn.x, tn.y), max(tn.z, 0.02));
    float end = min(min(tf.x, tf.y), min(tf.z, 30.0));
    if (t >= end) discard;
    // A hit needs an observed sample in front (positive) then one behind: entering a surface's
    // back from unseen space is not one.
    float previous = -1.0;
    float previous_t = t;
    bool hit = false;
    for (int i = 0; i < 1200 && t < end; i++) {
        vec2 s = at(eye + dir * t);
        if (s.g >= min_weight) {
            if (s.r <= 0.0 && previous > 0.0) {
                t = previous_t + (t - previous_t) * previous / (previous - s.r);
                hit = true;
                break;
            }
            previous = s.r;
        } else {
            previous = -1.0;
        }
        previous_t = t;
        t += s.g >= min_weight ? voxel * clamp(s.r * 2.0, 0.5, 2.0) : voxel * 2.0;
    }
    if (!hit) discard;
    vec3 p = eye + dir * t;
    // Trilinear over the voxels that hold a colour only (the rest are unseen, not black).
    vec3 g = (p - origin) / voxel - 0.5;
    ivec3 base = ivec3(floor(g));
    vec3 f = g - vec3(base);
    vec4 sum = vec4(0);
    for (int k = 0; k < 8; k++) {
        ivec3 o = ivec3(k & 1, (k >> 1) & 1, k >> 2);
        vec4 c = texelFetch(colors, base + o, 0);
        vec3 w3 = mix(1.0 - f, f, vec3(o));
        float w = w3.x * w3.y * w3.z * float(c.a > 0.0);
        sum += vec4(c.rgb * w, w);
    }
    vec3 rgb = sum.a > 0.0 ? sum.rgb / sum.a : vec3(0.3);
    if (keys > 0) {
        vec3 e = vec3(voxel, 0, 0);
        vec3 n = normalize(vec3(at(p + e.xyy).r - at(p - e.xyy).r,
                                at(p + e.yxy).r - at(p - e.yxy).r,
                                at(p + e.yyx).r - at(p - e.yyx).r) + 1e-6);
        float best = 0.0, second = 0.0;
        vec3 c1 = vec3(0), c2 = vec3(0);
        for (int j = 0; j < keys; j++) {
            int k = order[j];
            vec3 q = (key[k].camera_from_world * vec4(p, 1.0)).xyz;
            float range = -q.z;
            if (range < 0.05) continue;
            vec4 cal = key[k].calibration;   // fx fy cx cy as shares of the image
            vec2 uv = vec2(cal.x * q.x / range + cal.z, cal.w - cal.y * q.y / range);
            if (any(lessThan(uv, vec2(0.005))) || any(greaterThan(uv, vec2(0.995)))) continue;
            float seen = texture(key_depth, vec3(uv, k)).r;
            if (seen < 0.05 || abs(seen - range) > 0.03 + 0.05 * range) continue;
            vec3 toward = normalize(-(transpose(mat3(key[k].camera_from_world)) * q));
            float facing = max(dot(n, toward), 0.0);
            vec2 edge = min(uv, 1.0 - uv);
            float score = facing * facing / range * smoothstep(0.0, 0.06, min(edge.x, edge.y))
                * (0.4 + 0.6 * exp(-(now - key[k].info.x) / 60.0));
            vec3 c = texture(key_rgb, vec3(uv, k)).rgb;
            if (score > best) { second = best; c2 = c1; best = score; c1 = c; }
            else if (score > second) { second = score; c2 = c; }
        }
        if (best > 0.0) rgb = (c1 * best + c2 * second) / (best + second);
    }
    color = vec4(rgb, 1.0);
    vec4 clip = view_projection * vec4(p, 1.0);
    gl_FragDepth = 0.5 * clip.z / clip.w + 0.5;
}
"""

RETIRE = """
#version 430
layout(local_size_x=16, local_size_y=16) in;
layout(r8ui, binding=0) uniform uimage2DArray alive;
uniform sampler2DArray key_depth;
uniform int old_layer;
uniform int new_layer;
uniform mat4 new_from_old;
uniform vec4 old_cal;   // shares of the image
uniform vec4 new_cal;
uniform vec2 jump;
void main() {
    ivec2 size = imageSize(alive).xy;
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(pixel, size))) return;
    if (imageLoad(alive, ivec3(pixel, old_layer)).r == 0u) return;
    vec2 uv = (vec2(pixel) + 0.5) / vec2(size);
    float z = texture(key_depth, vec3(uv, old_layer)).r;
    if (z < 0.05) return;
    vec3 p = vec3((uv.x - old_cal.z) / old_cal.x, (old_cal.w - uv.y) / old_cal.y, -1.0) * z;
    p = (new_from_old * vec4(p, 1.0)).xyz;
    float range = -p.z;
    if (range < 0.05) return;
    vec2 t = vec2(new_cal.x * p.x / range + new_cal.z, new_cal.w - new_cal.y * p.y / range);
    if (any(lessThan(t, vec2(0.01))) || any(greaterThan(t, vec2(0.99)))) return;
    float lo = 1e9, hi = 0.0;
    for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
        float d = texture(key_depth, vec3(t + vec2(dx, dy) * 0.006, new_layer)).r;
        lo = min(lo, d); hi = max(hi, d);
    }
    if (lo < 0.05 || hi - lo > jump.x + jump.y * lo) return;
    float fresh = texture(key_depth, vec3(t, new_layer)).r;
    // In the new view's free space, or on its surface: the newer keyframe shows it now.
    if (range <= fresh + 0.03 + 0.04 * fresh) imageStore(alive, ivec3(pixel, old_layer), uvec4(0));
}
"""

COMPOSITE = """
#version 430
in vec2 ndc;
uniform sampler2D layer;
out vec4 color;
void main() { color = texture(layer, ndc * 0.5 + 0.5); }
"""


def _inverse_rigid(m):
    r = m[:3, :3]
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ m[:3, 3]
    return out


def _set(program, name, value, *, raw=False):
    # A uniform the shader compiler found unused is not there to set.
    if name in program:
        if raw:
            program[name].write(value)
        else:
            program[name].value = value


def _gl(m):
    return np.ascontiguousarray(np.asarray(m, dtype="f4").T).tobytes()


@dataclass
class Key:
    id: int
    layer: int
    frame: MeshFrame  # its pose and scale follow corrections of its anchor


def _similarity_parts(m):
    scale = float(np.cbrt(np.linalg.det(m[:3, :3])))
    return scale, m[:3, :3] / scale, m[:3, 3]


def _turn_degrees(a, b):
    cos = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2
    return math.degrees(math.acos(float(np.clip(cos, -1, 1))))


class MeshWorld:
    def __init__(self, context: moderngl.Context, *, memory: str = MEMORY, now: bool = True):
        if memory not in ("tsdf", "ibr", "atlas", "none"):
            raise ValueError(f"ITO_MESH_MEMORY must be tsdf, ibr, atlas or none, not {memory!r}")
        self.ctx, self.memory, self.show_now = context, memory, now
        ctx = context
        self.grid = ctx.program(
            vertex_shader=GRID_VERTEX, geometry_shader=GRID_GEOMETRY, fragment_shader=GRID_FRAGMENT
        )
        self.grid_vao = None
        self.grid_size = None
        self.grid_indices = None
        self.composite = ctx.program(vertex_shader=FULLSCREEN_VERTEX, fragment_shader=COMPOSITE)
        self.fullscreen = ctx.vertex_array(self.composite, [])
        self.now = None  # (rgb texture, depth texture, frame)
        self.field = self.colors = self.stamps = None
        self.origin = None
        self.frames = 0
        self.epoch = None
        self.target = None
        self.dummy_alive = ctx.texture_array((1, 1, 1), 1, b"\x01", dtype="u1")
        self.dummy_rgb = ctx.texture_array((1, 1, 1), 4, bytes(4))
        self.dummy_depth = ctx.texture_array((1, 1, 1), 1, bytes(4), dtype="f4")
        self.keys: dict[int, Key] = {}
        self.free_layers = list(range(KEYFRAMES))
        self.last_key = None
        self.key_count = 0
        if memory in ("tsdf", "ibr"):
            self.integrate = ctx.compute_shader(INTEGRATE)
            self.raycast = ctx.program(vertex_shader=FULLSCREEN_VERTEX, fragment_shader=RAYCAST)
            self.raycast_vao = ctx.vertex_array(self.raycast, [])
        if memory in ("ibr", "atlas"):
            self.key_rgb = ctx.texture_array((*KEY_RGB, KEYFRAMES), 4)
            self.key_rgb.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.key_depth = ctx.texture_array((*KEY_DEPTH, KEYFRAMES), 1, dtype="f4")
            self.key_depth.filter = (moderngl.NEAREST, moderngl.NEAREST)
            for t in (self.key_rgb, self.key_depth):
                t.repeat_x = t.repeat_y = False
            self.key_buffer = ctx.buffer(reserve=KEYFRAMES * 96)
            self.order_buffer = ctx.buffer(reserve=KEYFRAMES * 4)
            self.key_textures = {}  # atlas: layer -> (rgb, depth) at the frame's own size
        if memory == "atlas":
            self.alive = ctx.texture_array((*KEY_DEPTH, KEYFRAMES), 1, dtype="u1")
            self.alive.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.retire = ctx.compute_shader(RETIRE)

    # --- input ---------------------------------------------------------------------------

    def add(self, frame: MeshFrame) -> None:
        depth = np.asarray(frame.depth, dtype="f4")
        depth = np.where(np.isfinite(depth) & (depth > 0.05) & (depth < 30), depth, 0)
        frame.depth = np.ascontiguousarray(depth, dtype="f4")
        frame.pose = np.asarray(frame.pose, dtype="f4")
        if self.epoch is None:
            self.epoch = frame.captured
        if frame.key is None:
            frame.key = self._is_key(frame.pose)
        if self.show_now:
            self._set_now(frame)
        if self.memory in ("tsdf", "ibr"):
            if self.field is None:
                self._allocate_field(frame.pose)
            self._integrate(frame)
        if frame.key:
            self._add_key(frame)
        self.frames += 1

    def _is_key(self, pose):
        last = self.last_key
        if last is not None and np.linalg.norm(pose[:3, 3] - last[:3, 3]) < KEY_MOVE[0]:
            if _turn_degrees(pose, last) < KEY_MOVE[1]:
                return False
        self.last_key = pose
        return True

    def repose(self, deltas: dict[int, np.ndarray]) -> None:
        """Corrections from SLAM's global graph: anchor (keyframe) -> 4x4 world similarity.
        Every stored view and the NOW frame anchored to a moved keyframe follow it at once; the
        field is rebuilt from the stored views when one of them moved by more than a voxel."""
        moved = False
        frames = [k.frame for k in self.keys.values()]
        if self.now is not None and all(self.now[2] is not f for f in frames):
            frames.append(self.now[2])
        for frame in frames:
            delta = deltas.get(frame.anchor)
            if delta is None:
                continue
            scale, rotation, shift = _similarity_parts(np.asarray(delta, np.float64))
            pose = np.eye(4)
            pose[:3, :3] = rotation @ frame.pose[:3, :3]
            pose[:3, 3] = scale * rotation @ frame.pose[:3, 3] + shift
            if np.linalg.norm(pose[:3, 3] - frame.pose[:3, 3]) > VOXEL:
                moved = True
            moved |= _turn_degrees(pose, frame.pose) > math.degrees(VOXEL / 2)
            moved |= abs(scale - 1) > 0.01
            frame.pose = pose.astype("f4")
            frame.scale *= scale
        for key in self.keys.values():
            if key.frame.anchor in deltas:
                self._upload_key(key)
        if moved and self.field is not None:
            self._clear_field()
            for key in sorted(self.keys.values(), key=lambda k: k.frame.captured):
                self._integrate(key.frame)

    def _texture(self, image, components, dtype="f1"):
        h, w = image.shape[:2]
        if components == 3:  # RGB8 is emulated on some drivers (D3D12): upload RGBA
            image = np.concatenate((image, np.full((h, w, 1), 255, np.uint8)), -1)
            components = 4
        t = self.ctx.texture(
            (w, h), components, np.ascontiguousarray(image).tobytes(), dtype=dtype, alignment=1
        )
        t.repeat_x = t.repeat_y = False
        return t

    def _set_now(self, frame):
        if self.now is not None:
            self.now[0].release()
            self.now[1].release()
        rgb = self._texture(frame.rgb, 3)
        rgb.filter = (moderngl.LINEAR, moderngl.LINEAR)
        d = self._texture(frame.depth * frame.scale, 1, "f4")
        d.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.now = (rgb, d, frame)

    def _allocate_field(self, first_pose):
        half, below, above = BOUNDS
        centre = first_pose[:3, 3]
        self.origin = np.array((centre[0] - half, centre[1] - below, centre[2] - half), np.float32)
        dims = (
            int(math.ceil(2 * half / VOXEL)),
            int(math.ceil((below + above) / VOXEL)),
            int(math.ceil(2 * half / VOXEL)),
        )
        self.dims = dims
        self.box = np.array(dims, np.float32) * VOXEL
        ctx = self.ctx
        self.field = ctx.texture3d(dims, 2, dtype="f2")
        self.colors = ctx.texture3d(dims, 4, dtype="f1")
        self.stamps = ctx.texture3d(dims, 1, dtype="f4")
        for t in (self.field, self.colors):
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = t.repeat_z = False
        self._clear_field()

    def _clear_field(self):
        # Unknown space: weight 0.
        nx, ny, nz = self.dims
        self.field.write(bytes(nx * ny * nz * 4))
        self.colors.write(bytes(nx * ny * nz * 4))
        self.stamps.write(bytes(nx * ny * nz * 4))

    def _frustum_box(self, frame, depth, pose):
        depth = depth * frame.scale
        h, w = depth.shape
        fx, fy, cx, cy = frame.intrinsics
        far = float(np.percentile(depth[depth > 0], 99)) + 0.15 if (depth > 0).any() else 0
        if far <= 0.1:
            return None
        corners = [np.zeros(3)]
        for u, v in ((0, 0), (w, 0), (0, h), (w, h)):
            corners.append(np.array(((u - cx) / fx, (cy - v) / fy, -1.0)) * far)
        world = (pose[:3, :3] @ np.array(corners).T).T + pose[:3, 3]
        lo = np.floor((world.min(0) - self.origin) / VOXEL).astype(int)
        hi = np.ceil((world.max(0) - self.origin) / VOXEL).astype(int)
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, self.dims)
        if (hi <= lo).any():
            return None
        return lo, hi - lo

    def _integrate(self, frame):
        depth, pose = frame.depth, frame.pose
        box = self._frustum_box(frame, depth, pose)
        if box is None:
            return
        offset, extent = box
        rgb = self._texture(frame.rgb, 3)
        rgb.filter = (moderngl.LINEAR, moderngl.LINEAR)
        d = self._texture(depth * frame.scale, 1, "f4")
        d.filter = (moderngl.NEAREST, moderngl.NEAREST)
        d.use(0)
        rgb.use(1)
        p = self.integrate
        _set(p, "depth", 0)
        _set(p, "rgb", 1)
        _set(p, "offset", tuple(int(v) for v in offset))
        _set(p, "extent", tuple(int(v) for v in extent))
        _set(p, "origin", tuple(self.origin))
        _set(p, "voxel", VOXEL)
        _set(p, "camera_from_world", _gl(_inverse_rigid(pose)), raw=True)
        _set(p, "calibration", tuple(float(v) for v in frame.intrinsics))
        _set(p, "edge", EDGE_BEND)
        _set(p, "now", float(frame.captured - self.epoch))
        _set(p, "color_age", COLOR_AGE)
        _set(p, "max_weight", 24.0)
        self.field.bind_to_image(0, read=True, write=True)
        self.colors.bind_to_image(1, read=True, write=True)
        self.stamps.bind_to_image(2, read=True, write=True)
        p.run(*((int(e) + 7) // 8 for e in extent))
        self.ctx.memory_barrier()
        rgb.release()
        d.release()

    def _add_key(self, frame):
        # tsdf keeps its views too, untextured: a correction rebuilds the field from them.
        if self.memory == "none":
            return
        if not self.free_layers:
            self._evict()
        key = Key(self.key_count, self.free_layers.pop(0), frame)
        self.key_count += 1
        self.keys[key.id] = key
        self._upload_key(key)
        if self.memory == "atlas":
            ones = np.ones(KEY_DEPTH[::-1], np.uint8)
            self.alive.write(ones.tobytes(), viewport=(0, 0, key.layer, *KEY_DEPTH, 1))
            self._retire_older(key)

    def _upload_key(self, key):
        import cv2

        frame = key.frame
        if self.memory not in ("ibr", "atlas"):
            return
        depth = frame.depth * frame.scale
        rgb = cv2.resize(frame.rgb, KEY_RGB, interpolation=cv2.INTER_AREA)
        rgba = np.concatenate((rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)), -1)
        small = cv2.resize(depth, KEY_DEPTH, interpolation=cv2.INTER_NEAREST)
        self.key_rgb.write(rgba.tobytes(), viewport=(0, 0, key.layer, *KEY_RGB, 1))
        self.key_depth.write(
            small.astype("f4").tobytes(), viewport=(0, 0, key.layer, *KEY_DEPTH, 1)
        )
        if self.memory == "atlas":
            old = self.key_textures.pop(key.layer, None)
            if old is not None:
                old[0].release()
                old[1].release()
            t_rgb = self._texture(frame.rgb, 3)
            t_rgb.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t_depth = self._texture(depth, 1, "f4")
            t_depth.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.key_textures[key.layer] = (t_rgb, t_depth)

    @staticmethod
    def _calibration(frame):
        h, w = frame.depth.shape
        fx, fy, cx, cy = frame.intrinsics
        return (fx / w, fy / h, cx / w, cy / h)

    def _evict(self):
        """Drop the view a newer one covers best: the nearest pose to a newer view's."""
        keys = sorted(self.keys.values(), key=lambda k: k.frame.captured)
        best, victim = None, keys[0]
        for i, k in enumerate(keys[:-1]):
            for n in keys[i + 1 :]:
                cost = float(np.linalg.norm(k.frame.pose[:3, 3] - n.frame.pose[:3, 3]))
                cost += _turn_degrees(k.frame.pose, n.frame.pose) / 30
                if best is None or cost < best:
                    best, victim = cost, k
        del self.keys[victim.id]
        self.free_layers.append(victim.layer)
        textures = getattr(self, "key_textures", {}).pop(victim.layer, None)
        if textures is not None:
            textures[0].release()
            textures[1].release()

    def _retire_older(self, new):
        r = self.retire
        self.key_depth.use(1)
        _set(r, "key_depth", 1)
        _set(r, "new_layer", new.layer)
        _set(r, "new_cal", self._calibration(new.frame))
        _set(r, "jump", JUMP)
        self.alive.bind_to_image(0, read=True, write=True)
        inverse = _inverse_rigid(new.frame.pose)
        for key in self.keys.values():
            if key is new:
                continue
            _set(r, "old_layer", key.layer)
            _set(r, "old_cal", self._calibration(key.frame))
            _set(r, "new_from_old", _gl(inverse @ key.frame.pose), raw=True)
            r.run((KEY_DEPTH[0] + 15) // 16, (KEY_DEPTH[1] + 15) // 16)
        self.ctx.memory_barrier()

    # --- drawing -------------------------------------------------------------------------

    def _grid_for(self, size):
        if self.grid_size == size:
            return self.grid_vao
        if self.grid_vao is not None:
            self.grid_vao.release()
            self.grid_indices.release()
        w, h = size
        a = (np.arange(h - 1)[:, None] * w + np.arange(w - 1)).ravel()
        triangles = np.stack((a, a + w, a + 1, a + 1, a + w, a + w + 1), -1).astype("u4")
        self.grid_indices = self.ctx.buffer(triangles.tobytes())
        self.grid_vao = self.ctx.vertex_array(self.grid, [], index_buffer=self.grid_indices)
        self.grid_size = size
        return self.grid_vao

    def _draw_grid(self, rgb, depth, intrinsics, pose, view, projection, *, layer, bias, feather):
        vao = self._grid_for(depth.size)
        depth.use(0)
        rgb.use(1)
        (self.alive if layer >= 0 else self.dummy_alive).use(2)
        g = self.grid
        _set(g, "depth", 0)
        _set(g, "rgb", 1)
        _set(g, "alive", 2)
        _set(g, "layer", layer)
        _set(g, "calibration", tuple(float(v) for v in intrinsics))
        _set(g, "world_from_camera", _gl(pose), raw=True)
        _set(g, "view", _gl(view), raw=True)
        _set(g, "projection", _gl(projection), raw=True)
        _set(g, "edge", EDGE_BEND)
        _set(g, "bias", bias)
        _set(g, "feather", feather)
        vao.render(moderngl.TRIANGLES)

    def _layer_target(self, size):
        if self.target is None or self.target[0].size != size:
            if self.target is not None:
                for r in self.target:
                    r.release()
            color = self.ctx.texture(size, 4)
            color.filter = (moderngl.LINEAR, moderngl.LINEAR)
            depth = self.ctx.depth_renderbuffer(size)
            fbo = self.ctx.framebuffer([color], depth)
            self.target = (color, depth, fbo)
        return self.target

    def draw(
        self,
        world_from_eye,
        projection,
        target: moderngl.Framebuffer,
        viewport=None,
        *,
        clear=None,
        now_only=False,
        scale=1.0,
    ) -> None:
        """Draw for one eye. clear=None composites over what target holds (splats)."""
        ctx = self.ctx
        viewport = viewport if viewport is not None else (0, 0, *target.size)
        size = (max(1, int(viewport[2] * scale)), max(1, int(viewport[3] * scale)))
        color, _, fbo = self._layer_target(size)
        fbo.use()
        ctx.viewport = (0, 0, *size)
        ctx.scissor = None
        fbo.clear(0, 0, 0, 0, depth=1.0)
        world_from_eye = np.asarray(world_from_eye, dtype="f4")
        projection = np.asarray(projection, dtype="f4")
        view = _inverse_rigid(world_from_eye)
        ctx.enable_only(moderngl.DEPTH_TEST)
        ctx.depth_func = "<="
        if not now_only:
            self._draw_memory(world_from_eye, view, projection)
        if self.show_now and self.now is not None:
            ctx.enable_only(moderngl.DEPTH_TEST | moderngl.BLEND)
            ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
            rgb, depth, frame = self.now
            self._draw_grid(
                rgb,
                depth,
                frame.intrinsics,
                np.asarray(frame.pose, "f4"),
                view,
                projection,
                layer=-1,
                bias=NOW_BIAS if self.memory != "none" else 0.0,
                feather=0.05,
            )
        target.use()
        ctx.viewport = viewport
        if clear is not None:
            target.clear(*clear, viewport=viewport)
        ctx.enable_only(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        color.use(0)
        self.composite["layer"] = 0
        self.fullscreen.render(moderngl.TRIANGLES, vertices=3)

    def _draw_memory(self, world_from_eye, view, projection):
        if self.memory in ("tsdf", "ibr") and self.field is not None:
            r = self.raycast
            self.field.use(0)
            self.colors.use(1)
            _set(r, "field", 0)
            _set(r, "colors", 1)
            view_projection = projection @ view
            _set(r, "world_from_clip", _gl(np.linalg.inv(view_projection)), raw=True)
            _set(r, "view_projection", _gl(view_projection), raw=True)
            _set(r, "eye", tuple(float(v) for v in world_from_eye[:3, 3]))
            _set(r, "origin", tuple(float(v) for v in self.origin))
            _set(r, "box", tuple(float(v) for v in self.box))
            _set(r, "voxel", VOXEL)
            _set(r, "min_weight", 0.3)
            _set(r, "now", float(self.now[2].captured - self.epoch) if self.now else 0.0)
            keys = 0
            # Every sampler gets its own unit, used or not: two sampler types on one unit
            # make the draw invalid.
            _set(r, "key_rgb", 2)
            _set(r, "key_depth", 3)
            if self.memory == "ibr" and self.keys:
                keys = self._upload_keys(world_from_eye)
                self.key_rgb.use(2)
                self.key_depth.use(3)
                self.key_buffer.bind_to_storage_buffer(5)
                self.order_buffer.bind_to_storage_buffer(6)
            else:
                self.dummy_rgb.use(2)
                self.dummy_depth.use(3)
            _set(r, "keys", keys)
            self.raycast_vao.render(moderngl.TRIANGLES, vertices=3)
        elif self.memory == "atlas":
            for key in sorted(self.keys.values(), key=lambda k: -k.frame.captured):
                rgb, depth = self.key_textures[key.layer]
                self._draw_grid(
                    rgb,
                    depth,
                    key.frame.intrinsics,
                    key.frame.pose,
                    view,
                    projection,
                    layer=key.layer,
                    bias=0.0,
                    feather=0.0,
                )

    def _upload_keys(self, world_from_eye):
        records = np.zeros((KEYFRAMES, 24), np.float32)
        chosen = []
        forward = -world_from_eye[:3, 2]
        for key in self.keys.values():
            camera_from_world = _inverse_rigid(key.frame.pose)
            records[key.layer, :16] = camera_from_world.T.ravel()
            records[key.layer, 16:20] = self._calibration(key.frame)
            records[key.layer, 20] = key.frame.captured - self.epoch
            # A keyframe looking away from where this eye looks cannot colour what it sees.
            if float(np.dot(-key.frame.pose[:3, 2], forward)) > -0.2:
                chosen.append(key.layer)
        self.key_buffer.write(records.tobytes())
        order = np.array(chosen or [0], np.int32)
        self.order_buffer.write(order.tobytes())
        return len(chosen)

    def close(self):
        for r in (
            self.grid,
            self.grid_vao,
            self.grid_indices,
            self.composite,
            self.fullscreen,
            self.field,
            self.colors,
            self.stamps,
            self.dummy_alive,
            self.dummy_rgb,
            self.dummy_depth,
            getattr(self, "integrate", None),
            getattr(self, "raycast", None),
            getattr(self, "raycast_vao", None),
            getattr(self, "key_rgb", None),
            getattr(self, "key_depth", None),
            getattr(self, "key_buffer", None),
            getattr(self, "order_buffer", None),
            getattr(self, "alive", None),
            getattr(self, "retire", None),
        ):
            if r is not None:
                r.release()
        for t in getattr(self, "key_textures", {}).values():
            t[0].release()
            t[1].release()
        if self.now is not None:
            self.now[0].release()
            self.now[1].release()
        if self.target is not None:
            for r in self.target:
                r.release()


def projection_from_k(K, width, height, near=0.02, far=100.0):
    """OpenGL projection of an OpenCV pinhole K (pixel i centred at i) for an Ito camera."""
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2] + 0.5, K[1][2] + 0.5
    return np.array(
        (
            (2 * fx / width, 0, 1 - 2 * cx / width, 0),
            (0, 2 * fy / height, 2 * cy / height - 1, 0),
            (0, 0, (far + near) / (near - far), 2 * far * near / (near - far)),
            (0, 0, -1, 0),
        ),
        np.float32,
    )


class TapeFilm:
    """The mesh world for lane tapes' film tool (tools/tapes/replay.py): fed from a
    Reconstruction's poll_mesh(), drawn at a tape camera (Ito world-from-camera, OpenCV K) as
    a BGR image, over the splat image (ITO_MESH=over) or alone (ITO_MESH=only)."""

    def __init__(self, width, height):
        from ito.meshout import MESH

        self.mode = MESH
        memory = os.environ.get("ITO_MESH_MEMORY", "tsdf" if MESH == "only" else "none")
        self.size = (width, height)
        self.ctx = moderngl.create_standalone_context(require=430)
        self.world = MeshWorld(self.ctx, memory=memory)
        self.under = self.ctx.texture(self.size, 3)
        self.target = self.ctx.framebuffer([self.under])
        self.frames = self.moves = 0

    def feed(self, worker):
        for kind, item in worker.poll_mesh():
            if kind == "frame":
                self.world.add(item)
                self.frames += 1
            else:
                self.world.repose(item)
                self.moves += 1

    def draw(self, camera, K, under=None, background=(0.04, 0.04, 0.06)):
        width, height = self.size
        clear = None
        if under is not None and self.mode == "over":
            self.under.write(np.ascontiguousarray(np.flipud(under[..., ::-1])).tobytes())
        else:
            clear = (*background, 1.0)
        self.world.draw(
            np.asarray(camera, np.float32),
            projection_from_k(K, width, height),
            self.target,
            clear=clear,
        )
        rgb = np.frombuffer(self.target.read(components=3, alignment=1), np.uint8)
        return np.ascontiguousarray(np.flipud(rgb.reshape(height, width, 3))[..., ::-1])
