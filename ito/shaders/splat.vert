#version 460
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
// One Gaussian splat per instance, a screen-aligned quad around its projected ellipse
// (ito/recon/mast3r_slam/render.py; records as ring.SplatUpdate's).
layout(std430, set = 0, binding = 0) readonly buffer Order { uvec2 order[]; };
layout(std430, set = 0, binding = 1) readonly buffer Scene { vec4 attributes[]; };
layout(set = 0, binding = 2) uniform View {
    mat4 view;
    mat4 projection;
    vec4 eye;       // xyz, and w the scene's time
    vec4 viewport;  // width, height, and z the fade (seconds; 0: none)
    ivec4 layout_;  // x the record stride (vec4s), y the SH degree
};
layout(location = 0) out vec2 gaussian_uv;
layout(location = 1) flat out vec4 color_opacity;

vec4 attribute_at(uint id, uint offset) {
    return attributes[id * uint(layout_.x) + offset];
}
vec3 color(uint id, vec3 direction) {
    float x = direction.x, y = direction.y, z = direction.z;
    float basis[16];
    basis[0] = 0.2820947918;
    basis[1] = -0.4886025119 * y;
    basis[2] = 0.4886025119 * z;
    basis[3] = -0.4886025119 * x;
    basis[4] = 1.0925484306 * x * y;
    basis[5] = -1.0925484306 * y * z;
    basis[6] = 0.3153915653 * (2*z*z - x*x - y*y);
    basis[7] = -1.0925484306 * x * z;
    basis[8] = 0.5462742153 * (x*x - y*y);
    basis[9] = -0.5900435899 * y * (3*x*x - y*y);
    basis[10] = 2.8906114426 * x * y * z;
    basis[11] = -0.4570457995 * y * (4*z*z - x*x - y*y);
    basis[12] = 0.3731763326 * z * (2*z*z - 3*x*x - 3*y*y);
    basis[13] = -0.4570457995 * x * (4*z*z - x*x - y*y);
    basis[14] = 1.4453057213 * z * (x*x - y*y);
    basis[15] = -0.5900435899 * x * (x*x - 3*y*y);
    vec3 rgb = vec3(0.5);
    for (int k = 0; k < (layout_.y + 1)*(layout_.y + 1); ++k)
        rgb += basis[k] * attribute_at(id, uint(3+k)).rgb;
    return max(rgb, vec3(0));
}
void main() {
    uint id = order[gl_InstanceIndex].y;
    vec4 p = attribute_at(id, 0u);
    float scene_time = eye.w, fade_seconds = viewport.z;
    if (fade_seconds > 0)
        p.w *= clamp((attribute_at(id, 1u).w - scene_time) / fade_seconds, 0.0, 1.0);
    vec4 center = view * vec4(p.xyz, 1);
    vec4 clip_space = projection * center;
    gaussian_uv = vec2(0);
    color_opacity = vec4(0);
    gl_Position = vec4(2, 2, 2, 1);
    if (clip_space.w <= 0 || clip_space.z < -clip_space.w || clip_space.z > clip_space.w
        || p.w < 1.0/255.0) return;

    vec3 s = attribute_at(id, 1u).xyz;
    vec4 q = attribute_at(id, 2u); // wxyz
    float w = q.x, x = q.y, y = q.z, z = q.w;
    mat3 rotation = mat3(
        1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y),
        2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x),
        2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y));
    mat3 transform = mat3(view) * rotation * mat3(s.x,0,0, 0,s.y,0, 0,0,s.z);
    mat3 covariance = transform * transpose(transform);
    // Perspective Jacobian in pixels, including asymmetric stereo frusta.
    vec3 row_w = vec3(projection[0][3], projection[1][3], projection[2][3]);
    vec3 row_x = vec3(projection[0][0], projection[1][0], projection[2][0]);
    vec3 row_y = vec3(projection[0][1], projection[1][1], projection[2][1]);
    // Bound the linearization outside the frustum: nearby offscreen splats must
    // not stretch across the entire view as their centers approach the eye plane.
    vec2 ndc = clamp(clip_space.xy / clip_space.w, vec2(-1.3), vec2(1.3));
    vec3 jx = (row_x - ndc.x * row_w) * (viewport.x * 0.5 / clip_space.w);
    vec3 jy = (row_y - ndc.y * row_w) * (viewport.y * 0.5 / clip_space.w);
    float a = dot(jx, covariance * jx) + 0.3;
    float b = dot(jx, covariance * jy);
    float c = dot(jy, covariance * jy) + 0.3;
    float mid = 0.5*(a+c);
    float radius = length(vec2(0.5*(a-c), b));
    float major = sqrt(max(mid + radius, 0.3));
    float minor = sqrt(max(mid - radius, 0.3));
    vec2 axis = abs(b) > 1e-6 ? normalize(vec2(b, mid + radius - a))
                             : (a >= c ? vec2(1,0) : vec2(0,1));
    vec2 corner = vec2((gl_VertexIndex & 1) == 0 ? -1 : 1,
                       (gl_VertexIndex & 2) == 0 ? -1 : 1);
    gaussian_uv = corner * 3.0;
    vec2 offset = 3.0 * (corner.x * major * axis + corner.y * minor * vec2(-axis.y, axis.x));
    gl_Position = clip(clip_space + vec4(offset * 2.0 / viewport.xy * clip_space.w, 0, 0));
    color_opacity = vec4(color(id, normalize(p.xyz - eye.xyz)), p.w);
}
