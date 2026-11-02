#version 460
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
// Guessed depth beyond the live frame (render.MeshRenderer._fill): this eye's ray onto the
// viewer's filled depth map, coloured by the kept view that saw that spot from the most similar
// direction.
layout(set = 0, binding = 0) uniform Fill {
    mat4 clip_to_world;  // this view's
    mat4 fill_clip;      // world to the depth map's clip space
    vec4 eye;            // this view's
    vec4 viewer;         // the depth map's, between the eyes
    vec4 forward;        // the depth map's axis
    vec2 size;
    vec2 image_size;
    float floor_height;
    float floor_clearance;
    float far;
};
layout(set = 0, binding = 1, std430) readonly buffer Projectors { Projector projector[]; };
layout(set = 0, binding = 2) uniform sampler2D surface;  // the viewer's filled inverse depth
layout(set = 0, binding = 3) uniform sampler2DArray images;
layout(set = 0, binding = 4) uniform sampler2D shadows;
// Depth tested before the shader runs (the triangle is at depth 1.0): only empty pixels pay.
layout(early_fragment_tests) in;
layout(location = 0) out vec4 f_colour;

// Whether world point p lies behind the filled surface, as the viewer's map has it there.
bool behind(vec3 p) {
    vec4 c = fill_clip * vec4(p, 1.0);
    if (c.w <= 1e-4) return false;
    float inv = texture(surface, c.xy / c.w * 0.5 + 0.5).r;
    return inv > 1e-4 && dot(p - viewer.xyz, forward.xyz) * inv >= 1.0;
}

void main() {
    vec2 st = gl_FragCoord.xy / size;
    vec4 a = clip_to_world * vec4(st * 2.0 - 1.0, -1.0, 1.0);
    vec4 b = clip_to_world * vec4(st * 2.0 - 1.0, 1.0, 1.0);
    vec3 dir = normalize(b.xyz / b.w - a.xyz / a.w);
    // This eye's ray to the first point behind the viewer's filled surface: the nearest
    // surface it meets, as with any geometry (a surface the viewer sees in front hides the ones
    // behind it from both eyes alike).
    float r = far;
    float step = pow(far / 0.1, 1.0 / float(STEPS));
    float near = 0.1;
    for (int i = 1; i <= STEPS; ++i) {
        float t = 0.1 * pow(step, float(i));
        if (behind(eye.xyz + dir * t)) {
            for (int k = 0; k < 5; ++k) {
                float m = 0.5 * (near + t);
                if (behind(eye.xyz + dir * m)) t = m; else near = m;
            }
            r = t;
            break;
        }
        near = t;
    }
    float height = eye.y - floor_height;
    if (height > floor_clearance && dir.y < -1e-3) r = min(r, height / -dir.y);
    r = clamp(r, 0.1, far);
    vec3 p = eye.xyz + dir * r;
    vec3 sum = vec3(0.0);
    float total = 0.0;
    for (int k = 0; k < K; ++k) {
        Projector view = projector[k];
        if (view.origin.w <= 0.0) continue;
        vec4 c = view.camera_from_world * vec4(p, 1.0);
        float z = -c.z;
        if (z < 0.05) continue;
        vec4 q = view.intrinsics;
        float u = q.z + q.x * c.x / z;
        float v = q.w - q.y * c.y / z;
        if (u < 0.0 || v < 0.0 || u > image_size.x - 1.0 || v > image_size.y - 1.0) continue;
        vec2 uv = vec2((u + 0.5) / image_size.x, (v + 0.5) / image_size.y);
        float d = length(p - view.origin.xyz);
        // A measured surface in front of the spot hides it from this view.
        float nearest = texture(shadows, vec2((float(k) + uv.x) / float(K), 1.0 - uv.y)).r;
        if (d > nearest + 0.06 + 0.03 * d) continue;
        // The view that looked at the spot most nearly along the viewer's ray: least parallax
        // if the guessed range is wrong.
        float along_ray = max(dot((p - view.origin.xyz) / max(d, 1e-4), normalize(p - viewer.xyz)), 0.0);
        float edge = clamp(min(min(u, image_size.x - u), min(v, image_size.y - v))
                           / (0.06 * image_size.x), 0.0, 1.0);
        float w = max(view.origin.w * along_ray * along_ray * edge, 1e-5);
        float w2 = w * w;
        float w6 = w2 * w2 * w2;
        sum += w6 * texture(images, vec3(uv, float(k))).rgb;
        total += w6;
    }
    if (total <= 0.0) discard;  // no frame ever saw this spot
    f_colour = vec4(sum / total, 1.0);  // depth stays 1.0: the live guess's composite wins
}
