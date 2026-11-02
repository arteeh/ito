#version 460
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
// Each texel looks its world point up in the live frame like the projectors do and keeps the
// better view.
layout(set = 0, binding = 0) uniform Bake {
    mat4 world_to_clip;
    mat4 camera_from_world;
    vec4 intrinsics;
    vec4 origin;
    vec2 image_size;
    float shadow_slots;
};
layout(set = 0, binding = 1, rgba8) uniform image2DArray atlas;
layout(set = 0, binding = 2) uniform sampler2DArray images;
layout(set = 0, binding = 3) uniform sampler2D shadows;
layout(location = 0) in vec3 v_world;
layout(location = 1) in vec3 v_normal;
layout(location = 2) flat in ivec4 v_tile;
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
    float d = length(v_world - origin.xyz);
    float nearest = texture(shadows, vec2(st.x / shadow_slots, 1.0 - st.y)).r;
    if (d > nearest + 0.06 + 0.03 * d) return;
    // The projectors' weight (mesh.frag), without the live frame's boost.
    float s = dot(normalize(v_normal), (origin.xyz - v_world) / max(d, 1e-4));
    if (s < -0.25) return;
    float edge = clamp(min(min(u, image_size.x - u), min(v, image_size.y - v))
                       / (0.06 * image_size.x), 0.0, 1.0);
    float w = s * s * edge * central(intrinsics, u, v) / (1.0 + 0.5 * d * d);
    if (w <= 0.0) return;
    // Alpha holds the best weight any view of this texel had (its fourth root: 8 bits keep the
    // low weights of a grazing floor apart), and the colour is the last view's within BAKE_KEEP
    // of it. A best so far, not the colour's own weight: a turning camera's views of a texel
    // weaken a little each frame as it leaves the middle of the image, and against its own last
    // weight each would win, down to the frame's border.
    ivec3 at = ivec3(texel, v_tile.z);
    float best = imageLoad(atlas, at).a;
    best *= best;
    best *= best;
    if (w < BAKE_KEEP * best) return;
    float a = max(sqrt(sqrt(max(w, best))), 1.0 / 255.0);
    imageStore(atlas, at, vec4(texture(images, vec3(st, 0.0)).rgb, a));
}
