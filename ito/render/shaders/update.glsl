#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 1) buffer Scene { vec4 attributes[]; };
layout(std430, binding = 2) readonly buffer Changes { vec4 changes[]; };
layout(std430, binding = 3) readonly buffer Slots { uint slots[]; };
layout(std430, binding = 4) buffer Previous { vec4 previous[]; };
uniform uint count;
uniform float ease;
uniform float scene_time;

// With ease > 0 a changed splat glides from what is drawn now to its new record over ease
// seconds (previous[] holds the start, its [1].w when it began). A splat taking an empty
// or retiring slot fades in where it is, never flying from the old occupant.
vec4 rotation_from(vec4 from, vec4 to, float a) {
    if (dot(from, to) < 0.0) from = -from;
    return normalize(mix(from, to, a));
}
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= count) return;
    uint s = slots[i] * 4u;
    if (ease > 0.0) {
        vec4 n0 = changes[i * 4u], o0 = attributes[s];
        bool born = n0.w > 0.0 && (o0.w <= 0.0 || (attributes[s + 3u].w == 1.0 && changes[i * 4u + 3u].w != 1.0));
        if (born || n0.w <= 0.0) {
            for (uint j = 0u; j < 4u; ++j) previous[s + j] = changes[i * 4u + j];
            previous[s].w = 0.0;
        } else {
            float a = smoothstep(0.0, 1.0, clamp((scene_time - previous[s + 1u].w) / ease, 0.0, 1.0));
            previous[s] = mix(previous[s], o0, a);
            previous[s + 1u].xyz = mix(previous[s + 1u].xyz, attributes[s + 1u].xyz, a);
            previous[s + 2u] = rotation_from(previous[s + 2u], attributes[s + 2u], a);
            previous[s + 3u] = mix(previous[s + 3u], attributes[s + 3u], a);
        }
        previous[s + 1u].w = scene_time;
    }
    for (uint j = 0u; j < 4u; ++j) attributes[s + j] = changes[i * 4u + j];
}
