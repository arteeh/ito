#version 460
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
layout(push_constant) uniform Shadow {
    mat4 world_to_clip;
    vec4 origin;
};
layout(location = 0) in vec3 position;
layout(location = 0) out vec3 v_world;
void main() {
    v_world = position;
    gl_Position = clip(world_to_clip * vec4(position, 1.0));
}
