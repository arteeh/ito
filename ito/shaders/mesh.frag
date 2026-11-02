#version 460
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
// Radians between a view's ray and the eye's at which the view's weight halves (before ^6);
// EDGE_VIEW_ANGLE near a silhouette in that view: within EDGE_TEXELS of the view's distance map
// (~1.3 degrees) the inverse range bends by more than EDGE_JUMP of its own value.
#define VIEW_ANGLE 0.1
#define EDGE_VIEW_ANGLE 0.03
#define EDGE_TEXELS 3.0
#define EDGE_JUMP 0.15
layout(set = 0, binding = 0) uniform View {
    mat4 view_projection;
    vec4 eye;          // the viewer's: between the pilot's eyes, the same for both
    vec2 image_size;
    int atlas_layers;  // 0: no atlas yet
};
layout(set = 0, binding = 1, std430) readonly buffer Projectors { Projector projector[]; };
layout(set = 0, binding = 2) uniform sampler2DArray images;
layout(set = 0, binding = 3) uniform sampler2D shadows;
layout(set = 0, binding = 4) uniform sampler2DArray atlas;
layout(location = 0) in vec3 v_world;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec4 v_colour;
layout(location = 3) in vec3 v_local;
layout(location = 4) flat in int v_slot;
layout(location = 0) out vec4 f_colour;

// The atlas at this fragment: rgb and the weight of the view it kept (alpha^4), blended over the
// tiles of the directions the triangle faces; w 0 where none of them was baked here.
vec4 baked(vec3 face) {
    if (v_slot < 0) return vec4(0.0);
    vec3 rgb = vec3(0.0);
    float quality = 0.0, total = 0.0;
    for (int d = 0; d < 6; ++d) {
        float a = along(face, d);
        if (a < DRAW_FACING) continue;
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

float inverse_range(vec2 at, vec2 offset, float lo, float hi) {
    vec2 texel = 1.0 / vec2(textureSize(shadows, 0));
    at += offset * texel;
    // Within this view's slot of the atlas (which wraps around).
    at = clamp(at, vec2(lo, 0.0) + 0.5 * texel, vec2(hi, 1.0) - 0.5 * texel);
    return 1.0 / texture(shadows, at).r;  // empty: 1e9, nearly 0
}

bool silhouette(vec2 at, int k, float inverse) {
    float lo = float(k) / float(K), hi = float(k + 1) / float(K);
    float r = EDGE_TEXELS;
    float across = inverse_range(at, vec2(-r, 0.0), lo, hi)
                   + inverse_range(at, vec2(r, 0.0), lo, hi) - 2.0 * inverse;
    float down = inverse_range(at, vec2(0.0, -r), lo, hi)
                 + inverse_range(at, vec2(0.0, r), lo, hi) - 2.0 * inverse;
    return max(abs(across), abs(down)) > EDGE_JUMP * inverse;
}

void main() {
    // The triangle's own normal (towards free space), which picks its atlas tiles as the bake did.
    vec3 face = normalize(cross(dFdx(v_world), dFdy(v_world)));
    if (dot(face, v_normal) < 0.0) face = -face;
    vec3 n = normalize(v_normal);
    // Normals point to free space; a surface seen from behind shows its back the same colour.
    if (dot(n, eye.xyz - v_world) < 0.0) n = -n;
    vec3 base;
    if (v_colour.a > 0.5) {
        base = v_colour.rgb;
    } else {
        // Never coloured by a camera frame: plain grey, shaded so its shape reads.
        float light = 0.5 + 0.35 * max(dot(n, normalize(vec3(0.3, 1.0, 0.2))), 0.0);
        base = vec3(0.5) * light;
    }
    vec3 to_eye = eye.xyz - v_world;
    float eye_distance = max(length(to_eye), 1e-4);
    to_eye /= eye_distance;
    vec3 sum = vec3(0.0);
    float total = 0.0;
    for (int k = 0; k < K; ++k) {
        Projector view = projector[k];
        if (view.origin.w <= 0.0) continue;
        vec4 c = view.camera_from_world * vec4(v_world, 1.0);
        float z = -c.z;
        if (z < 0.05) continue;
        vec4 q = view.intrinsics;
        float u = q.z + q.x * c.x / z;
        float v = q.w - q.y * c.y / z;
        if (u < 0.0 || v < 0.0 || u > image_size.x - 1.0 || v > image_size.y - 1.0) continue;
        vec2 st = vec2((u + 0.5) / image_size.x, (v + 0.5) / image_size.y);
        float d = length(v_world - view.origin.xyz);
        vec2 at = vec2((float(k) + st.x) / float(K), 1.0 - st.y);
        float nearest = texture(shadows, at).r;
        if (d > nearest + 0.06 + 0.03 * d) continue;
        vec3 toward = (view.origin.xyz - v_world) / max(d, 1e-4);
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
        float view_angle = silhouette(at, k, 1.0 / d) ? EDGE_VIEW_ANGLE : VIEW_ANGLE;
        float along_ray = 1.0 / (1.0 + angle * angle / (view_angle * view_angle));
        // Of views along the same ray, one nearer than the eye resolves as much detail as the
        // screen shows; a farther one less.
        float detail = clamp((eye_distance + 0.5) / (d + 0.5), 0.5, 1.0);
        float edge = clamp(min(min(u, image_size.x - u), min(v, image_size.y - v))
                           / (0.06 * image_size.x), 0.0, 1.0);
        // A border sample stays in the blend (w^6 stays a normal float) where it is the only
        // view.
        float w = max(view.origin.w * along_ray * detail * edge, 1e-5);
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
