#version 460
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
// The flat camera view: a panel ahead of the pilot's head.
layout(push_constant) uniform Panel {
    mat4 clip_from_panel;
    vec2 half_size;
    float distance;
};
layout(location = 0) out vec2 uv;
const vec2 corners[4] = vec2[4](vec2(-1, -1), vec2(1, -1), vec2(-1, 1), vec2(1, 1));
void main() {
    vec2 p = corners[gl_VertexIndex];
    uv = vec2(p.x * .5 + .5, .5 - p.y * .5);
    gl_Position = clip(clip_from_panel * vec4(p * half_size, -distance, 1));
}
