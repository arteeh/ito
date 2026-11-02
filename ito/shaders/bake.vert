#version 460
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 3) in vec4 block;
layout(location = 0) out vec3 g_world;
layout(location = 1) out vec3 g_normal;
layout(location = 2) out vec3 g_local;
layout(location = 3) out int g_slot;
void main() {
    g_world = position;
    g_normal = normal;
    g_local = block.xyz;
    g_slot = int(block.w + 0.5);
}
