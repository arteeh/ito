#version 430
layout(local_size_x = 256) in;
uniform sampler2D gaussians;
uniform uint count;
uniform uint capacity;
uniform uint stride;
uniform mat4 view;
layout(std430, binding = 0) writeonly buffer Order { uvec2 order[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= capacity) return;
    if (i >= count) { order[i] = uvec2(0, 0xffffffffu); return; }
    uint address = i * stride;
    uint width = uint(textureSize(gaussians, 0).x);
    vec3 center = texelFetch(gaussians, ivec2(address % width, address / width), 0).xyz;
    float depth = max(0.0, -(view * vec4(center, 1)).z);
    order[i] = uvec2(floatBitsToUint(depth), i);
}
