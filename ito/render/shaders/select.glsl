#version 430
// Screen-space LOD, shared by both eyes. Reduce bounds per workgroup: a global
// atomic from every visible splat serializes the scan on just four addresses.
layout(local_size_x = 256) in;
layout(std430, binding = 1) readonly buffer Scene { vec4 attributes[]; };
layout(std430, binding = 4) buffer Selection { uint selected[]; };
uniform uint id_bits;
uniform uint count;
uniform uint stride;
uniform uint cells_x;
uniform uint cells_y;
uniform mat4 view;
uniform mat4 projection;
uniform int phase;
shared uvec4 bounds[256];

bool project_point(uint id, out vec4 camera, out vec2 ndc) {
    camera = vec4(0);
    ndc = vec2(0);
    if (id >= count) return false;
    vec4 p = attributes[id * stride];
    if (p.w < 1.0/255.0) return false;
    camera = view * vec4(p.xyz, 1);
    vec4 clip = projection * camera;
    if (clip.w < 0.01 || clip.z < -clip.w || clip.z > clip.w) return false;
    ndc = clip.xy / clip.w / 1.15;
    return !any(greaterThan(abs(ndc), vec2(1)));
}

void main() {
    uint id = gl_GlobalInvocationID.x;
    uint cells = cells_x * cells_y;
    if (phase == 0) {
        if (id < 2u*cells+4u) selected[id] = 0xffffffffu;
        return;
    }
    vec4 camera;
    vec2 ndc;
    bool visible = project_point(id, camera, ndc);
    if (phase == 1) {
        uint lane = gl_LocalInvocationID.x;
        bounds[lane] = visible ? uvec4(floatBitsToUint(ndc+1.0), floatBitsToUint(1.0-ndc))
                               : uvec4(0xffffffffu);
        barrier();
        for (uint d=128u; d>0u; d>>=1u) {
            if (lane < d) bounds[lane] = min(bounds[lane], bounds[lane+d]);
            barrier();
        }
        if (lane == 0u && bounds[0].x != 0xffffffffu) {
            for (uint k=0u; k<4u; ++k) atomicMin(selected[2u*cells+k], bounds[0][k]);
        }
        return;
    }
    if (!visible) return;
    vec2 lower = vec2(uintBitsToFloat(selected[2u*cells]), uintBitsToFloat(selected[2u*cells+1u]))-1.0;
    vec2 upper = 1.0-vec2(uintBitsToFloat(selected[2u*cells+2u]), uintBitsToFloat(selected[2u*cells+3u]));
    vec2 cell_uv = clamp((ndc-lower)/max(upper-lower,vec2(0.001)), 0.0, 0.999999);
    uvec2 pixel = uvec2(cell_uv*vec2(cells_x,cells_y));
    uint cell = pixel.y*cells_x+pixel.x;
    uint depth = uint(clamp(log2(1.0-camera.z)*float((1u << (32u-id_bits))-2u)/16.0, 0.0, float((1u << (32u-id_bits))-2u)));
    uint key = (depth << id_bits) | id;
    atomicMin(selected[cell + (id & 1u)*cells], key);
}
