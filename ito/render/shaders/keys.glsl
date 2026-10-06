#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 1) readonly buffer Scene { vec4 attributes[]; };
uniform uint count;
uniform uint capacity;
uniform uint stride;
uniform mat4 view;
layout(std430, binding = 0) writeonly buffer Order { uvec2 order[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= capacity) return;
    if (i >= count) { order[i] = uvec2(0, 0xffffffffu); return; }
    vec3 center = attributes[i * stride].xyz;
    float depth = max(0.0, -(view * vec4(center, 1)).z);
    order[i] = uvec2(floatBitsToUint(depth), i);
}
