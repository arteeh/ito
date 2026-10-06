#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 1) buffer Scene { vec4 attributes[]; };
layout(std430, binding = 2) readonly buffer Changes { vec4 changes[]; };
layout(std430, binding = 3) readonly buffer Slots { uint slots[]; };
uniform uint count;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= count) return;
    for (uint j = 0u; j < 4u; ++j) attributes[slots[i] * 4u + j] = changes[i * 4u + j];
}
