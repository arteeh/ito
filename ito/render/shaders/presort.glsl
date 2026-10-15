#version 430
// Build and sort each 1024-key tile in one dispatch, before the global merges.
layout(local_size_x = 1024) in;
layout(std430, binding = 1) readonly buffer Scene { vec4 attributes[]; };
layout(std430, binding = 0) writeonly buffer Order { uvec2 order[]; };
layout(std430, binding = 4) readonly buffer Selection { uint selected[]; };
uniform bool use_selection;
uniform uint id_bits;
uniform uint count;
uniform uint stride;
uniform mat4 view;
shared uvec2 tile[1024];
bool before(uvec2 a, uvec2 b) {
    return a.x > b.x || (a.x == b.x && a.y < b.y);
}
void main() {
    uint i = gl_GlobalInvocationID.x;
    uint lane = gl_LocalInvocationID.x;
    uvec2 key = uvec2(0, 0xffffffffu);
    uint id = i;
    bool valid = i < count;
    if (use_selection) {
        uint entry = valid ? selected[i] : 0xffffffffu;
        valid = entry != 0xffffffffu;
        id = entry & ((1u << id_bits)-1u);
    }
    if (valid) {
        float depth = max(0.0, -(view * vec4(attributes[id * stride].xyz, 1)).z);
        key = uvec2(floatBitsToUint(depth), id);
    }
    tile[lane] = key;
    barrier();
    for (uint stage = 2u; stage <= 1024u; stage <<= 1u) {
        for (uint d = stage >> 1u; d > 0u; d >>= 1u) {
            uvec2 a = tile[lane], b = tile[lane ^ d];
            bool descending = (i & stage) == 0u;
            bool first = (lane & d) == 0u;
            bool swap = first ? (before(b, a) == descending) : (before(a, b) == descending);
            barrier();
            if (swap) tile[lane] = b;
            barrier();
        }
    }
    order[i] = tile[lane];
}
