#version 430
// Bitonic merges: distant pairs use SSBOs, each final 256-key merge stays local.
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Order { uvec2 order[]; };
uniform uint capacity;
uniform uint stage;
uniform uint distance;
uniform bool local_merge;
shared uvec2 tile[256];
bool before(uvec2 a, uvec2 b) {
    return a.x > b.x || (a.x == b.x && a.y < b.y);
}
void main() {
    uint i = gl_GlobalInvocationID.x;
    uint lane = gl_LocalInvocationID.x;
    if (local_merge) {
        tile[lane] = order[i];
        barrier();
        for (uint d = distance; d > 0u; d >>= 1u) {
            uint partner = lane ^ d;
            uvec2 a = tile[lane], b = tile[partner];
            bool descending = (i & stage) == 0u;
            bool first = (lane & d) == 0u;
            bool swap = first ? (before(b, a) == descending) : (before(a, b) == descending);
            barrier();
            if (swap) tile[lane] = b;
            barrier();
        }
        order[i] = tile[lane];
    } else {
        uint partner = i ^ distance;
        if (partner <= i || partner >= capacity) return;
        uvec2 a = order[i], b = order[partner];
        if (before(b, a) == ((i & stage) == 0u)) {
            order[i] = b;
            order[partner] = a;
        }
    }
}
