#version 460
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
layout(set = 0, binding = 0) uniform Guess {
    mat4 view_projection;
    mat4 world_from_camera;
    vec4 origin;
    vec4 intrinsics;     // fx, fy, cx, cy in image pixels
    vec2 image_size;
    vec2 grid_step;      // one grid cell in image uv
    float floor_height;
    float floor_clearance;
    float far;
    float edge;
};
layout(set = 0, binding = 2) uniform sampler2D filled;    // inverse depth, holes filled
layout(set = 0, binding = 3) uniform sampler2D measured;  // weight 1 where the mesh itself is
layout(location = 0) in vec2 uv;                          // over the image, v down
layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec3 v_world;
layout(location = 2) out float v_keep;
layout(location = 3) out float v_measured;

vec3 camera_ray(vec2 st) {  // not normalized: z = -1
    vec2 p = st * image_size - 0.5;
    return vec3((p.x - intrinsics.z) / intrinsics.x, -(p.y - intrinsics.w) / intrinsics.y, -1.0);
}

vec3 ray(vec2 st) {
    return mat3(world_from_camera) * normalize(camera_ray(st));
}

float range_at(vec2 st) {
    vec3 d = ray(st);
    float inv = textureLod(filled, vec2(st.x, 1.0 - st.y), 0.0).r;
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
    v_measured = textureLod(measured, vec2(uv.x, 1.0 - uv.y), 0.0).g;
    v_uv = uv;
    v_world = origin.xyz + ray(uv) * r;
    gl_Position = clip(view_projection * vec4(v_world, 1.0));
}
