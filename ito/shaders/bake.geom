#version 460
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
// Baking the live frame into the atlas: every triangle is drawn in atlas space (texel =
// fragment), once per tile it belongs to, grown by about a texel so the texels along its edges
// are baked too (the draw samples them bilinearly).
layout(triangles) in;
layout(triangle_strip, max_vertices = 9) out;
layout(set = 0, binding = 0) uniform Bake {
    mat4 world_to_clip;  // the live camera's
    mat4 camera_from_world;
    vec4 intrinsics;
    vec4 origin;
    vec2 image_size;
    float shadow_slots;
};
layout(location = 0) in vec3 g_world[];
layout(location = 1) in vec3 g_normal[];
layout(location = 2) in vec3 g_local[];
layout(location = 3) in int g_slot[];
layout(location = 0) out vec3 v_world;
layout(location = 1) out vec3 v_normal;
layout(location = 2) flat out ivec4 v_tile;  // corner x, y, layer, and the tile's side
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
        if (along(face, d) < BAKE_FACING) continue;
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
            gl_Position = vec4(q / float(LAYER) * 2.0 - 1.0, 0.0, 1.0);
            EmitVertex();
        }
        EndPrimitive();
    }
}
