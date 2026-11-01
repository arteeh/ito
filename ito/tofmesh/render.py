"""Draws the ToF mesh, textured by projecting camera frames onto it.

The mesh carries a colour per vertex (the volume's voxel colours, a few centimetres coarse:
the map's memory). On top, a handful of camera frames act as slide projectors: every fragment
looks itself up in each projector that sees it unoccluded (a distance map per projector,
rendered from the mesh) and takes the frames' colours, weighted towards the projector looking at
it most nearly along the eye's ray, so detail comes at camera resolution where a frame saw it.
Slot 0 is the live frame; the others are kept views (fusion.py chooses them).

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

Texture memory: the kept views are few (PROJECTORS) and the oldest is replaced, so a slow 360
degree turn outlived them and everything seen more than about half a turn ago fell back to the
vertex colours: 4 cm voxels averaged over many frames, a smear (the floor's planks gone). So
every live frame is also baked into a texture atlas at about the camera's resolution: every
block owns six square tiles, one per axis direction (+X, -X, +Y, ...), and a triangle lives in
the tiles of the directions its face points along (a planar projection of the block onto that
axis, fixed in the world, so re-meshing a block keeps its texture in place). A texel keeps the
frame that saw it best (facing it most squarely, nearest and nearest the image's middle) and
is replaced only by a view about as good (BAKE_KEEP): seen
areas stay as sharp as their best view, and a view as good refreshes them (a moved object). The
atlas joins the projector blend with the weight of the best view it had. Tiles of a block the
volume forgot are cleared when its slot is given to a new block (MeshFrame.cleared).
"""

import logging

import moderngl
import numpy as np
from OpenGL import GL

from ito.render.pose import validate_pose

from .fusion import PROJECTORS

log = logging.getLogger(__name__)

SHADOW_SIZE = (256, 144)
NEAR, FAR = 0.05, 30.0

# Texture atlas: a tile is TILE texels square and spans TILE_SPAN voxels of its block, from one
# voxel before the block's corner (a block's vertices reach half a voxel past its faces): 6.25 mm
# texels at 4 cm voxels, the camera frame's own resolution (640 wide) about 2 m away.
TILE = 64
TILE_SPAN = 10.0
DIRECTIONS = 6
LAYER = 4096  # an atlas layer is LAYER texels square: (LAYER / TILE)^2 tiles
LAYER_TILES = (LAYER // TILE) ** 2
# A triangle is baked into the tile of every direction its face is within ~78 degrees of, and
# drawn from those within ~75 (so a drawn tile was always baked).
BAKE_FACING = 0.2
DRAW_FACING = 0.25
# A texel takes a new view's colour when that view's weight is at least this share of the best
# weight a view of it ever had.
BAKE_KEEP = 0.85

VERTEX = """
#version 430
uniform mat4 view_projection;
in vec3 position;
in vec3 normal;
in vec4 colour;
in vec4 block;
out vec3 v_world;
out vec3 v_normal;
out vec4 v_colour;
out vec3 v_local;
flat out int v_slot;
void main() {
    v_world = position;
    v_normal = normal;
    v_colour = colour;
    v_local = block.xyz;
    v_slot = int(block.w + 0.5);
    gl_Position = view_projection * vec4(position, 1.0);
}
"""

# A tile's texel coordinates (in its atlas layer) of a point at `local` voxels of its block, for
# direction d (axis d / 2, + for even d), and the tile's layer and corner.
TILE_GLSL = """
const int TILE = %(TILE)d;
const int LAYER_SIDE = %(SIDE)d;  // tiles per layer side
const float TILE_SPAN = %(SPAN)f;

ivec3 tile_corner(int slot, int d) {
    int t = slot * 6 + d;
    int within = t %% (LAYER_SIDE * LAYER_SIDE);
    return ivec3((within %% LAYER_SIDE) * TILE, (within / LAYER_SIDE) * TILE,
                 t / (LAYER_SIDE * LAYER_SIDE));
}

vec2 tile_uv(vec3 local, int d) {
    int a = d / 2;
    vec2 p = a == 0 ? local.yz : a == 1 ? local.zx : local.xy;
    return (p + 1.0) / TILE_SPAN * float(TILE);
}

float along(vec3 n, int d) {
    return (d %% 2 == 0 ? 1.0 : -1.0) * n[d / 2];
}
"""

FRAGMENT = """
#version 430
#define K %(K)d
// Radians between a view's ray and the eye's at which the view's weight halves (before ^6);
// EDGE_VIEW_ANGLE near a silhouette in that view: within EDGE_TEXELS of the view's distance map
// (~1.3 degrees) the inverse range bends by more than EDGE_JUMP of its own value.
#define VIEW_ANGLE 0.1
#define EDGE_VIEW_ANGLE 0.03
#define EDGE_TEXELS 3.0
#define EDGE_JUMP 0.15
uniform sampler2DArray images;
uniform sampler2D shadows;
uniform mat4 camera_from_world[K];
uniform vec4 intrinsics[K];      // fx, fy, cx, cy in image pixels
uniform vec3 origin[K];
uniform float boost[K];          // 0: empty slot
uniform vec2 image_size;
uniform vec3 eye;
uniform sampler2DArray atlas;
uniform int atlas_layers;        // 0: no atlas yet
in vec3 v_world;
in vec3 v_normal;
in vec4 v_colour;
in vec3 v_local;
flat in int v_slot;
out vec4 f_colour;
%(TILE_GLSL)s
// The atlas at this fragment: rgb and the weight of the view it kept (alpha^4), blended over the
// tiles of the directions the triangle faces; w 0 where none of them was baked here.
vec4 baked(vec3 face) {
    if (v_slot < 0) return vec4(0.0);
    vec3 rgb = vec3(0.0);
    float quality = 0.0, total = 0.0;
    for (int d = 0; d < 6; ++d) {
        float a = along(face, d);
        if (a < %(DRAW_FACING)f) continue;
        ivec3 corner = tile_corner(v_slot, d);
        if (corner.z >= atlas_layers) continue;
        // Bilinear over the texels that were baked (an unbaked one is no colour, not black).
        vec2 p = tile_uv(v_local, d) - 0.5;
        ivec2 i0 = ivec2(floor(p));
        vec2 f = p - vec2(i0);
        for (int j = 0; j < 2; ++j)
            for (int i = 0; i < 2; ++i) {
                ivec2 q = clamp(i0 + ivec2(i, j), ivec2(0), ivec2(TILE - 1));
                vec4 t = texelFetch(atlas, ivec3(corner.xy + q, corner.z), 0);
                if (t.a <= 0.0) continue;
                float w = a * a * (i == 1 ? f.x : 1.0 - f.x) * (j == 1 ? f.y : 1.0 - f.y);
                float t2 = t.a * t.a;
                rgb += w * t.rgb;
                quality += w * t2 * t2;
                total += w;
            }
    }
    return total > 1e-6 ? vec4(rgb / total, quality / total) : vec4(0.0);
}

float inverse_range(vec2 atlas, vec2 offset, float lo, float hi) {
    vec2 texel = 1.0 / vec2(textureSize(shadows, 0));
    vec2 at = atlas + offset * texel;
    // Within this view's slot of the atlas (which wraps around).
    at = clamp(at, vec2(lo, 0.0) + 0.5 * texel, vec2(hi, 1.0) - 0.5 * texel);
    return 1.0 / texture(shadows, at).r;  // empty: 1e9, nearly 0
}

bool silhouette(vec2 atlas, int k, float inverse) {
    float lo = float(k) / float(K), hi = float(k + 1) / float(K);
    float r = EDGE_TEXELS;
    float across = inverse_range(atlas, vec2(-r, 0.0), lo, hi)
                   + inverse_range(atlas, vec2(r, 0.0), lo, hi) - 2.0 * inverse;
    float down = inverse_range(atlas, vec2(0.0, -r), lo, hi)
                 + inverse_range(atlas, vec2(0.0, r), lo, hi) - 2.0 * inverse;
    return max(abs(across), abs(down)) > EDGE_JUMP * inverse;
}

void main() {
    // The triangle's own normal (towards free space), which picks its atlas tiles as the bake did.
    vec3 face = normalize(cross(dFdx(v_world), dFdy(v_world)));
    if (dot(face, v_normal) < 0.0) face = -face;
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
    vec3 to_eye = eye - v_world;
    float eye_distance = max(length(to_eye), 1e-4);
    to_eye /= eye_distance;
    vec3 sum = vec3(0.0);
    float total = 0.0;
    for (int k = 0; k < K; ++k) {
        if (boost[k] <= 0.0) continue;
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
        // A view clearly from behind (n is turned to the eye) saw the other side of a one-sheet
        // surface (a board, a table top from below, a chair back), not this one; a bumpy floor's
        // normals tip far less, so floor seen at a grazing angle is still coloured.
        if (dot(n, toward) < -0.25) continue;
        // The mesh is coarse (8x8 ToF zones, 4 cm voxels): its edges miss the image's by
        // centimetres. A frame seen from the eye's own direction puts every pixel back where
        // that frame saw it, whatever the mesh's error; one from aside shifts the error onto the
        // screen (background on a table leg's edge, a flap of wall hanging off a table top). So
        // the view looking at the fragment most nearly along the eye's ray wins (unstructured
        // lumigraph weighting), not the one facing the surface most squarely.
        float angle = acos(clamp(dot(toward, to_eye), -1.0, 1.0));
        // Near a silhouette in this view (its distance map jumps within a few texels: the
        // inverse range is not linear across the fragment, as it is on a plane) the mesh's edge
        // and the image's disagree most, so there the view must look along the eye's ray more
        // closely still. The live frame seen from where it was taken is exact at any edge.
        float view_angle = silhouette(atlas, k, 1.0 / d) ? EDGE_VIEW_ANGLE : VIEW_ANGLE;
        float along = 1.0 / (1.0 + angle * angle / (view_angle * view_angle));
        // Of views along the same ray, one nearer than the eye resolves as much detail as the
        // screen shows; a farther one less.
        float detail = clamp((eye_distance + 0.5) / (d + 0.5), 0.5, 1.0);
        float edge = clamp(min(min(u, image_size.x - u), min(v, image_size.y - v))
                           / (0.06 * image_size.x), 0.0, 1.0);
        // A border sample stays in the blend (w^6 stays a normal float) where it is the only
        // view.
        float w = max(boost[k] * along * detail * edge, 1e-5);
        // A steep power: the best view nearly alone, so small pose errors between views do
        // not show as double images.
        float w2 = w * w;
        float w6 = w2 * w2 * w2;
        sum += w6 * texture(images, vec3(st, float(k))).rgb;
        total += w6;
    }
    if (atlas_layers > 0) {
        vec4 b = baked(face);
        if (b.a > 0.0) {
            float w = max(b.a, 1e-5);
            float w2 = w * w;
            float w6 = w2 * w2 * w2;
            sum += w6 * b.rgb;
            total += w6;
        }
    }
    // Any frame that sees the fragment (or the atlas, kept from one) colours it fully; the
    // voxel colours (or grey) only where none does.
    vec3 colour = total > 0.0 ? sum / total : base;
    f_colour = vec4(colour, 1.0);
}
"""

# Baking the live frame into the atlas: every triangle is drawn in atlas space (texel = fragment),
# once per tile it belongs to, grown by about a texel so the texels along its edges are baked
# too (the draw samples them bilinearly); each texel looks its world point up in the live frame
# like the projectors do and keeps the better view.
BAKE_VERTEX = """
#version 430
in vec3 position;
in vec3 normal;
in vec4 block;
out vec3 g_world;
out vec3 g_normal;
out vec3 g_local;
out int g_slot;
void main() {
    g_world = position;
    g_normal = normal;
    g_local = block.xyz;
    g_slot = int(block.w + 0.5);
}
"""

BAKE_GEOMETRY = """
#version 430
layout(triangles) in;
layout(triangle_strip, max_vertices = 9) out;
uniform mat4 world_to_clip;  // the live camera's
in vec3 g_world[];
in vec3 g_normal[];
in vec3 g_local[];
in int g_slot[];
out vec3 v_world;
out vec3 v_normal;
flat out ivec4 v_tile;  // corner x, y, layer, and the tile's side
%(TILE_GLSL)s
void main() {
    // Only triangles the live camera may see.
    vec4 c[3];
    for (int i = 0; i < 3; ++i) c[i] = world_to_clip * vec4(g_world[i], 1.0);
    for (int a = 0; a < 3; ++a) {
        if (c[0][a] > c[0].w && c[1][a] > c[1].w && c[2][a] > c[2].w) return;
        if (c[0][a] < -c[0].w && c[1][a] < -c[1].w && c[2][a] < -c[2].w) return;
    }
    vec3 face = cross(g_world[1] - g_world[0], g_world[2] - g_world[0]);
    float size = length(face);
    if (size < 1e-12) return;
    face /= size;
    if (dot(face, g_normal[0] + g_normal[1] + g_normal[2]) < 0.0) face = -face;
    vec3 world_centre = (g_world[0] + g_world[1] + g_world[2]) / 3.0;
    vec3 normal_centre = (g_normal[0] + g_normal[1] + g_normal[2]) / 3.0;
    for (int d = 0; d < 6; ++d) {
        if (along(face, d) < %(BAKE_FACING)f) continue;
        ivec3 corner = tile_corner(g_slot[0], d);
        vec2 p[3];
        for (int i = 0; i < 3; ++i) p[i] = vec2(corner.xy) + tile_uv(g_local[i], d);
        vec2 centre = (p[0] + p[1] + p[2]) / 3.0;
        // Grown about its centre so its edges move out by about a texel (by its inradius).
        float area = abs((p[1].x - p[0].x) * (p[2].y - p[0].y)
                         - (p[2].x - p[0].x) * (p[1].y - p[0].y));
        float perimeter = distance(p[0], p[1]) + distance(p[1], p[2]) + distance(p[2], p[0]);
        float grow = 1.0 + min(1.0 / max(area / max(perimeter, 1e-6), 1e-3), 2.0);
        for (int i = 0; i < 3; ++i) {
            vec2 q = centre + (p[i] - centre) * grow;
            v_world = world_centre + (g_world[i] - world_centre) * grow;
            v_normal = normal_centre + (g_normal[i] - normal_centre) * grow;
            v_tile = ivec4(corner, TILE);
            gl_Position = vec4(q / float(%(LAYER)d) * 2.0 - 1.0, 0.0, 1.0);
            EmitVertex();
        }
        EndPrimitive();
    }
}
"""

BAKE_FRAGMENT = """
#version 430
layout(binding = 0, rgba8) uniform image2DArray atlas;
uniform sampler2DArray images;
uniform sampler2D shadows;
uniform mat4 camera_from_world;
uniform vec4 intrinsics;
uniform vec3 origin;
uniform vec2 image_size;
uniform float shadow_slots;
in vec3 v_world;
in vec3 v_normal;
flat in ivec4 v_tile;
// A baked view's weight also falls off away from its image's middle (q: fx, fy, cx, cy): of the
// frames of a turn that all see a spot, the one that saw it nearest the middle wins. Lens edges
// and the floor's height error (a centimetre moves a grazing floor point by many pixels) are
// worst at the border.
float central(vec4 q, float u, float v) {
    vec2 r = vec2((u - q.z) / q.x, (v - q.w) / q.y);
    return 1.0 / (1.0 + 2.0 * dot(r, r));
}

void main() {
    ivec2 texel = ivec2(gl_FragCoord.xy);
    if (any(lessThan(texel, v_tile.xy)) || any(greaterThanEqual(texel, v_tile.xy + v_tile.w)))
        return;  // grown past its tile
    vec4 c = camera_from_world * vec4(v_world, 1.0);
    float z = -c.z;
    if (z < 0.05) return;
    float u = intrinsics.z + intrinsics.x * c.x / z;
    float v = intrinsics.w - intrinsics.y * c.y / z;
    if (u < 0.0 || v < 0.0 || u > image_size.x - 1.0 || v > image_size.y - 1.0) return;
    vec2 st = vec2((u + 0.5) / image_size.x, (v + 0.5) / image_size.y);
    float d = length(v_world - origin);
    float nearest = texture(shadows, vec2(st.x / shadow_slots, 1.0 - st.y)).r;
    if (d > nearest + 0.06 + 0.03 * d) return;
    // The projectors' weight (render FRAGMENT), without the live frame's boost.
    float s = dot(normalize(v_normal), (origin - v_world) / max(d, 1e-4));
    if (s < -0.25) return;
    float edge = clamp(min(min(u, image_size.x - u), min(v, image_size.y - v))
                       / (0.06 * image_size.x), 0.0, 1.0);
    float w = s * s * edge * central(intrinsics, u, v) / (1.0 + 0.5 * d * d);
    if (w <= 0.0) return;
    // Alpha holds the best weight any view of this texel had (its fourth root: 8 bits keep the
    // low weights of a grazing floor apart), and the colour is the last view's within BAKE_KEEP of
    // it. A best so far, not the colour's own weight: a turning camera's views of a texel weaken
    // a little each frame as it leaves the middle of the image, and against its own last weight
    // each would win, down to the frame's border.
    ivec3 at = ivec3(texel, v_tile.z);
    float best = imageLoad(atlas, at).a;
    best *= best;
    best *= best;
    if (w < %(BAKE_KEEP)f * best) return;
    float a = max(sqrt(sqrt(max(w, best))), 1.0 / 255.0);
    imageStore(atlas, at, vec4(texture(images, vec3(st, 0.0)).rgb, a));
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


# Guessed depth beyond the live frame (the view is wider and taller than the camera, and leads it
# while turning): every pixel the mesh and the live guess leave empty gets a range from the view's
# own depth map, its holes filled from the map's mipmaps (the nearest level that has something
# there: one mipmap build instead of a push-pull's twenty passes), the floor rule, and the colour of
# the kept view that saw that spot from the most similar direction. A spot no frame ever saw
# stays empty.
EYE_DEPTH_WIDTH = 256

EYE_DEPTH_FRAGMENT = """
#version 430
out vec2 result;
void main() { result = vec2(gl_FragCoord.w, 1.0); }  // 1 / depth along the view axis
"""

FILL_FRAGMENT = """
#version 430
#define K %(K)d
uniform mat4 clip_to_world;
uniform vec2 size;
uniform vec3 eye;
uniform vec3 forward;
uniform sampler2D filled;    // the view's inverse depth and weight, mipmapped
uniform sampler2DArray images;
uniform sampler2D shadows;
uniform mat4 camera_from_world[K];
uniform vec4 intrinsics[K];
uniform vec3 origin[K];
uniform float boost[K];
uniform vec2 image_size;
uniform float floor_height;
uniform float floor_clearance;
uniform float far;
// Depth tested before the shader runs (the triangle is at depth 1.0): only empty pixels pay.
layout(early_fragment_tests) in;
out vec4 f_colour;

void main() {
    vec2 st = gl_FragCoord.xy / size;
    vec4 a = clip_to_world * vec4(st * 2.0 - 1.0, -1.0, 1.0);
    vec4 b = clip_to_world * vec4(st * 2.0 - 1.0, 1.0, 1.0);
    vec3 dir = normalize(b.xyz / b.w - a.xyz / a.w);
    float inv = 0.0;
    int levels = textureQueryLevels(filled);
    for (int l = 0; l < levels; ++l) {
        vec2 c = textureLod(filled, st, float(l)).rg;  // weighted means of 1 / depth
        if (c.y > 0.02) {
            inv = c.x / c.y;
            break;
        }
    }
    float r = inv > 1e-4 ? 1.0 / (inv * max(dot(dir, forward), 1e-3)) : far;
    float height = eye.y - floor_height;
    if (height > floor_clearance && dir.y < -1e-3) r = min(r, height / -dir.y);
    r = clamp(r, 0.1, far);
    vec3 p = eye + dir * r;
    vec3 sum = vec3(0.0);
    float total = 0.0;
    for (int k = 0; k < K; ++k) {
        if (boost[k] <= 0.0) continue;
        vec4 c = camera_from_world[k] * vec4(p, 1.0);
        float z = -c.z;
        if (z < 0.05) continue;
        vec4 q = intrinsics[k];
        float u = q.z + q.x * c.x / z;
        float v = q.w - q.y * c.y / z;
        if (u < 0.0 || v < 0.0 || u > image_size.x - 1.0 || v > image_size.y - 1.0) continue;
        vec2 uv = vec2((u + 0.5) / image_size.x, (v + 0.5) / image_size.y);
        float d = length(p - origin[k]);
        // A measured surface in front of the spot hides it from this view.
        float nearest = texture(shadows, vec2((float(k) + uv.x) / float(K), 1.0 - uv.y)).r;
        if (d > nearest + 0.06 + 0.03 * d) continue;
        // The view that looked at the spot most nearly along the pilot's ray: least parallax
        // if the guessed range is wrong.
        float along = max(dot((p - origin[k]) / max(d, 1e-4), dir), 0.0);
        float edge = clamp(min(min(u, image_size.x - u), min(v, image_size.y - v))
                           / (0.06 * image_size.x), 0.0, 1.0);
        float w = max(boost[k] * along * along * edge, 1e-5);
        float w2 = w * w;
        float w6 = w2 * w2 * w2;
        sum += w6 * texture(images, vec3(uv, float(k))).rgb;
        total += w6;
    }
    if (total <= 0.0) discard;  // no frame ever saw this spot
    f_colour = vec4(sum / total, 1.0);  // depth stays 1.0: the live guess's composite wins
}
"""


def _pyramid(context, size):
    """Push-pull levels from `size` down to 1x1: (size, pushed texture, its fbo, pulled texture,
    its fbo)."""
    levels = []
    w, h = size
    while True:
        level = [(w, h)]
        for _ in range(2):
            texture = context.texture((w, h), 2, dtype="f4")
            texture.filter = moderngl.LINEAR, moderngl.LINEAR
            texture.repeat_x = texture.repeat_y = False
            level += [texture, context.framebuffer([texture])]
        levels.append(level)
        if w == 1 and h == 1:
            return levels
        w, h = (w + 1) // 2, (h + 1) // 2


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


def _shader(source, slots=PROJECTORS):
    """A shader source with the atlas constants (and K, the projector slots) filled in."""
    tile = TILE_GLSL % {"TILE": TILE, "SIDE": LAYER // TILE, "SPAN": TILE_SPAN}
    return source % {
        "K": slots, "TILE_GLSL": tile, "DRAW_FACING": DRAW_FACING, "BAKE_FACING": BAKE_FACING,
        "BAKE_KEEP": BAKE_KEEP, "LAYER": LAYER,
    }  # fmt: skip


def _set(program, **values):
    for name, value in values.items():
        if name in program:  # the compiler drops unused uniforms
            program[name] = value


class MeshRenderer:
    def __init__(self, context: moderngl.Context, slots=PROJECTORS, image_size=(640, 360)):
        self.context = context
        self.slots = slots
        self.image_size = tuple(image_size)
        self.program = context.program(
            vertex_shader=VERTEX, fragment_shader=_shader(FRAGMENT, slots)
        )
        self.bake_program = context.program(
            vertex_shader=BAKE_VERTEX,
            geometry_shader=_shader(BAKE_GEOMETRY, slots),
            fragment_shader=_shader(BAKE_FRAGMENT, slots),
        )
        self.atlas = None  # the texture atlas, layers added as the volume's slots fill
        self.atlas_target = context.empty_framebuffer((LAYER, LAYER))
        self.bake_due = False  # a new live frame to bake
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
        self.live_time = None  # exposure of the image in slot 0
        self.shadow_dirty = set()
        self.vbo = self.nbo = self.cbo = self.ibo = self.bbo = None
        self.vao = self.shadow_vao = self.bake_vao = None
        self.capacity = (0, 0)
        self.count = 0  # triangles
        self.target = None
        self.epoch = None  # the worker run whose mesh and projectors are shown
        # Kept views whose distance map is older than the mesh: refreshed one per drawn frame,
        # so a mesh update (~12.7 a second) redraws only the live view's map at once.
        self.shadow_stale = set()
        self.guess_dirty = False
        self.levels = []
        self.push_program = self.pull_program = self.push_vao = self.pull_vao = None
        self.guess_program = self.guess_vbo = self.guess_ibo = self.guess_vao = None
        self.composite_program = self.composite_vao = None
        self.guess_target = None  # (size, framebuffer, colour texture, depth)
        self.eye_depth_program = self.eye_depth_vao = self.fill_program = self.fill_vao = None
        self.eye_depth = None  # the view's own depth map: texture, depth buffer, framebuffer
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
        self.levels = _pyramid(context, SHADOW_SIZE)
        self.eye_depth_program = context.program(
            vertex_shader=SHADOW_VERTEX, fragment_shader=EYE_DEPTH_FRAGMENT
        )
        self.fill_program = context.program(
            vertex_shader=COMPOSITE_VERTEX, fragment_shader=FILL_FRAGMENT % {"K": self.slots}
        )
        self.fill_vao = context.vertex_array(self.fill_program, [])
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
            self.live_time = None
            self.shadow_dirty.clear()
            self.shadow_stale.clear()
            self._clear_atlas()
        if self.atlas is not None:
            for slot in getattr(frame, "cleared", ()):
                self._clear_tiles(slot)
        if frame.positions is not None:
            blocks = getattr(frame, "blocks", None)
            if blocks is None:
                blocks = np.zeros((len(frame.positions), 4), np.float32)
                blocks[:, 3] = -1e6  # no slot: no atlas tile
            elif len(blocks):
                self._atlas_for(int(blocks[:, 3].max()))
            self._mesh(frame.positions, frame.normals, frame.colours, frame.triangles, blocks)
            if self.projectors[0] is not None:
                self.shadow_dirty.add(0)
            self.shadow_stale.update(
                k for k, p in enumerate(self.projectors) if k and p is not None
            )
        # The live frame first: a kept view is usually that same frame (fusion.py keeps the
        # frame it is projecting live), and is then copied on the GPU from slot 0.
        for p in sorted(frame.projectors, key=lambda p: p.slot):
            self._projector(p)
        return True

    apply = upload

    # -- texture atlas -------------------------------------------------------------------
    def _atlas_for(self, slot):
        """Grow the atlas to hold this volume slot's tiles (a layer, 64 MB, at a time; the old
        layers copied over on the GPU)."""
        layers = (slot * DIRECTIONS + DIRECTIONS - 1) // LAYER_TILES + 1
        have = 0 if self.atlas is None else self.atlas.size[2]
        if layers <= have:
            return
        atlas = self.context.texture_array((LAYER, LAYER, layers), 4, dtype="f1")
        atlas.filter = moderngl.NEAREST, moderngl.NEAREST
        atlas.repeat_x = atlas.repeat_y = False
        zeros = bytes(LAYER * LAYER * 4)
        for layer in range(have, layers):
            atlas.write(zeros, viewport=(0, 0, layer, LAYER, LAYER, 1))
        if self.atlas is not None:
            GL.glCopyImageSubData(
                self.atlas.glo, GL.GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0,
                atlas.glo, GL.GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, LAYER, LAYER, have,
            )  # fmt: skip
            self.atlas.release()
        self.atlas = atlas
        log.info(
            "ToF mesh texture atlas: %d layer(s), %d MB, for volume slots up to %d",
            layers, layers * LAYER * LAYER * 4 // 2**20, layers * LAYER_TILES // DIRECTIONS - 1,
        )  # fmt: skip

    def _clear_tiles(self, slot):
        """Forget the texture of the block that held this slot."""
        side = LAYER // TILE
        zeros = bytes(TILE * TILE * 4)
        for d in range(DIRECTIONS):
            t = slot * DIRECTIONS + d
            layer, within = divmod(t, LAYER_TILES)
            if layer >= self.atlas.size[2]:
                return
            y, x = divmod(within, side)
            self.atlas.write(zeros, viewport=(x * TILE, y * TILE, layer, TILE, TILE, 1))

    def _clear_atlas(self):
        if self.atlas is not None:
            zeros = bytes(LAYER * LAYER * 4)
            for layer in range(self.atlas.size[2]):
                self.atlas.write(zeros, viewport=(0, 0, layer, LAYER, LAYER, 1))

    def _bake(self):
        """The live frame into the atlas, wherever it sees the mesh better than the atlas's view."""
        self.bake_due = False
        camera_from_world, (fx, fy, cx, cy), origin = self.projectors[0]
        clip = pinhole_projection(fx, fy, cx, cy, *self.image_size) @ camera_from_world
        program = self.bake_program
        program["world_to_clip"].write(clip.T.astype(np.float32).tobytes())
        program["camera_from_world"].write(camera_from_world.T.astype(np.float32).tobytes())
        self.images.use(1)
        self.shadows.use(2)
        _set(
            program, images=1, shadows=2, intrinsics=(fx, fy, cx, cy),
            origin=tuple(float(x) for x in origin), image_size=self.image_size,
            shadow_slots=float(self.slots),
        )  # fmt: skip
        self.atlas.bind_to_image(0, read=True, write=True)
        self.atlas_target.use()
        self.context.viewport = (0, 0, LAYER, LAYER)
        self.context.scissor = None
        self.context.enable_only(moderngl.NOTHING)
        self.bake_vao.render(moderngl.TRIANGLES, vertices=self.count * 3)
        self.context.memory_barrier()

    def _mesh(self, positions, normals, colours, triangles, blocks):
        n, t = len(positions), len(triangles)
        if n > self.capacity[0] or t > self.capacity[1]:
            for b in (
                self.vbo, self.nbo, self.cbo, self.ibo, self.bbo, self.vao, self.shadow_vao,
                self.eye_depth_vao, self.bake_vao,
            ):  # fmt: skip
                if b is not None:
                    b.release()
            cap = (max(1024, int(n * 1.5)), max(1024, int(t * 1.5)))
            self.vbo = self.context.buffer(reserve=cap[0] * 12)
            self.nbo = self.context.buffer(reserve=cap[0] * 12)
            self.cbo = self.context.buffer(reserve=cap[0] * 4)
            self.bbo = self.context.buffer(reserve=cap[0] * 16)
            self.ibo = self.context.buffer(reserve=cap[1] * 12)
            self.vao = self.context.vertex_array(
                self.program,
                [
                    (self.vbo, "3f", "position"),
                    (self.nbo, "3f", "normal"),
                    (self.cbo, "4f1", "colour"),
                    (self.bbo, "4f", "block"),
                ],
                self.ibo,
                index_element_size=4,
            )
            self.bake_vao = self.context.vertex_array(
                self.bake_program,
                [
                    (self.vbo, "3f", "position"),
                    (self.nbo, "3f", "normal"),
                    (self.bbo, "4f", "block"),
                ],
                self.ibo,
                index_element_size=4,
            )
            self.shadow_vao = self.context.vertex_array(
                self.shadow_program, [(self.vbo, "3f", "position")], self.ibo, index_element_size=4
            )
            if self.eye_depth_program is not None:
                self.eye_depth_vao = self.context.vertex_array(
                    self.eye_depth_program, [(self.vbo, "3f", "position")], self.ibo,
                    index_element_size=4,
                )  # fmt: skip
            self.capacity = cap
        if n:
            self.vbo.write(np.ascontiguousarray(positions, np.float32))
            self.nbo.write(np.ascontiguousarray(normals, np.float32))
            self.cbo.write(np.ascontiguousarray(colours, np.uint8))
            self.bbo.write(np.ascontiguousarray(blocks, np.float32))
            self.ibo.write(np.ascontiguousarray(triangles, np.uint32))
        self.count = t

    def _projector(self, p):
        fx, fy, cx, cy, w, h = p.intrinsics
        if p.image is not None and p.slot and self.live_time == p.time:
            # The image slot 0 holds already: a GPU copy, no second upload.
            GL.glCopyImageSubData(
                self.images.glo, GL.GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0,
                self.images.glo, GL.GL_TEXTURE_2D_ARRAY, 0, 0, 0, p.slot, *self.image_size, 1,
            )  # fmt: skip
            sx, sy = self.image_size[0] / w, self.image_size[1] / h
            fx, fy = fx * sx, fy * sy
            cx, cy = (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5
        elif p.image is not None:
            if p.slot == 0:
                self.live_time = p.time
                self.bake_due = True
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
        _, (fx, fy, cx, cy), _ = self.projectors[0]
        sx, sy = SHADOW_SIZE[0] / self.image_size[0], SHADOW_SIZE[1] / self.image_size[1]
        intrinsics = (fx * sx, fy * sy, (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5)
        self._push_pull(self.levels, self.shadows, SHADOW_SIZE, intrinsics)

    def _push_pull(self, levels, source, source_size, intrinsics):
        """Push-pull over `levels`; level 0 from `source`, a distance map with these
        intrinsics."""
        self.context.enable_only(moderngl.NOTHING)
        self.context.scissor = None
        for index, (size, pushed, fbo, _, _) in enumerate(levels):
            fbo.use()
            self.context.viewport = (0, 0, *size)
            source.use(0)
            _set(
                self.push_program, source=0, first=int(index == 0), source_size=source_size,
                intrinsics=intrinsics,
            )  # fmt: skip
            self.push_vao.render(moderngl.TRIANGLES, vertices=3)
            source, source_size = pushed, size
        coarse = levels[-1][1]  # the top level's pull is its push
        for size, pushed, _, pulled, fbo in reversed(levels):
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

    def _fill(self, view_projection, world_from_eye, fbo, size):
        """Pixels still empty after the mesh: a guessed range from the view's own depth map,
        coloured by the kept view that saw that spot most nearly along the pilot's ray."""
        if not any(p is not None for p in self.projectors):
            return
        w = EYE_DEPTH_WIDTH
        eye_size = (w, max(1, round(w * size[1] / size[0])))
        if self.eye_depth is None or self.eye_depth[0].size != eye_size:
            for r in self.eye_depth or ():
                r.release()
            texture = self.context.texture(eye_size, 2, dtype="f4")
            texture.repeat_x = texture.repeat_y = False
            texture.build_mipmaps()
            texture.filter = moderngl.LINEAR_MIPMAP_NEAREST, moderngl.LINEAR
            depth = self.context.depth_renderbuffer(eye_size)
            self.eye_depth = (texture, depth, self.context.framebuffer([texture], depth))
        texture, _, target = self.eye_depth
        target.use()
        self.context.viewport = (0, 0, *eye_size)
        self.context.scissor = None
        self.context.enable_only(moderngl.DEPTH_TEST)
        target.clear(0, 0, 0, 0, depth=1.0)
        if self.count and self.eye_depth_vao is not None:
            self.eye_depth_program["world_to_clip"].write(
                view_projection.T.astype(np.float32).tobytes()
            )
            self.eye_depth_vao.render(moderngl.TRIANGLES, vertices=self.count * 3)
        texture.build_mipmaps()
        program = self.fill_program
        program["clip_to_world"].write(
            np.linalg.inv(view_projection).T.astype(np.float32).tobytes()
        )
        program["camera_from_world"].write(self._uniforms["matrices"].tobytes())
        program["intrinsics"].write(self._uniforms["intrinsics"].tobytes())
        program["origin"].write(self._uniforms["origins"].tobytes())
        program["boost"].write(self._uniforms["boost"].tobytes())
        texture.use(2)
        self.images.use(0)
        self.shadows.use(1)
        _set(
            program,
            size=tuple(float(x) for x in size),
            eye=tuple(float(x) for x in world_from_eye[:3, 3]),
            forward=tuple(float(x) for x in -world_from_eye[:3, 2]),
            images=0,
            shadows=1,
            filled=2,
            image_size=self.image_size,
            floor_height=FLOOR_HEIGHT,
            floor_clearance=FLOOR_CLEARANCE,
            far=GUESS_FAR,
        )
        fbo.use()
        self.context.viewport = (0, 0, *size)
        self.context.enable_only(moderngl.DEPTH_TEST)
        self.context.depth_func = "<="
        self.fill_vao.render(moderngl.TRIANGLES, vertices=3)
        self.context.depth_func = "<"

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
        if (
            self.bake_due and self.atlas is not None and self.count and self.bake_vao is not None
            and self.projectors[0] is not None
        ):  # fmt: skip
            self._bake()
        guess = self.projectors[0] is not None
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
        self._uniforms = dict(
            matrices=matrices, intrinsics=intrinsics, origins=origins, boost=boost
        )
        if self.count and self.vao is not None:
            program = self.program
            program["view_projection"].write(view_projection.T.astype(np.float32).tobytes())
            program["eye"] = tuple(world_from_eye[:3, 3])
            program["image_size"] = self.image_size
            program["camera_from_world"].write(matrices.tobytes())
            program["intrinsics"].write(intrinsics.tobytes())
            program["origin"].write(origins.tobytes())
            program["boost"].write(boost.tobytes())
            self.images.use(0)
            self.shadows.use(1)
            program["images"] = 0
            program["shadows"] = 1
            if self.atlas is not None:
                self.atlas.use(2)
            _set(program, atlas=2, atlas_layers=0 if self.atlas is None else self.atlas.size[2])
            self.vao.render(moderngl.TRIANGLES, vertices=self.count * 3)
        if guess:
            # Under the live guess (both at depth 1.0; the live frame's own guess wins).
            self._fill(view_projection, world_from_eye, fbo, size)
            fbo.use()
            self._guesses(view_projection, fbo, size)
        x, y, w, h = viewport
        self.context.copy_framebuffer(target, fbo) if (
            (x, y) == (0, 0) and tuple(target.size) == size
        ) else self._blit(fbo, target, viewport)
        target.use()
        self.context.viewport = viewport
        self.context.enable_only(moderngl.BLEND)

    def _blit(self, source, target, viewport):
        x, y, w, h = viewport
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, source.glo)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, target.glo)
        GL.glBlitFramebuffer(0, 0, w, h, x, y, x + w, y + h, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def close(self):
        for r in (
            self.vbo, self.nbo, self.cbo, self.ibo, self.bbo, self.vao, self.shadow_vao,
            self.bake_vao, self.bake_program, self.atlas, self.atlas_target, self.program,
            self.shadow_program, self.images, self.shadows, self.shadow_depth, self.shadow_fbo,
            self.push_vao, self.pull_vao, self.push_program, self.pull_program, self.guess_vao,
            self.guess_vbo, self.guess_ibo, self.guess_program, self.composite_vao,
            self.composite_program, self.eye_depth_vao, self.eye_depth_program, self.fill_vao,
            self.fill_program,
            *(self.eye_depth or ()),
            *(r for level in self.levels for r in level[1:]),
            *(self.guess_target[1:] if self.guess_target is not None else ()),
        ):  # fmt: skip
            if r is not None:
                r.release()
        if self.target is not None:
            for r in self.target[1:]:
                r.release()
