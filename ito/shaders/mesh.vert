#version 460
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
layout(set = 0, binding = 0) uniform View {
    mat4 view_projection;
    vec4 eye;          // the viewer's: between the pilot's eyes, the same for both
    vec2 image_size;
    int atlas_layers;  // 0: no atlas yet
};
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 colour;
layout(location = 3) in vec4 block;
layout(location = 0) out vec3 v_world;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec4 v_colour;
layout(location = 3) out vec3 v_local;
layout(location = 4) flat out int v_slot;
void main() {
    v_world = position;
    v_normal = normal;
    v_colour = colour;
    v_local = block.xyz;
    v_slot = int(block.w + 0.5);
    gl_Position = clip(view_projection * vec4(position, 1.0));
}
