#version 430
// Screen-space LOD: one near and one alternate representative per tile.
// Both eyes share this selection; the expanded frustum includes foreground parallax.
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
void main() {
    uint id = gl_GlobalInvocationID.x;
    uint cells = cells_x * cells_y;
    if (phase == 0) {
        if (id < 2u*cells+4u) selected[id] = 0xffffffffu;
        return;
    }
    if (id >= count) return;
    vec4 p = attributes[id * stride];
    if (p.w < 1.0/255.0) return;
    vec4 camera = view * vec4(p.xyz, 1);
    vec4 clip = projection * camera;
    if (clip.w < 0.01 || clip.z < -clip.w || clip.z > clip.w) return;
    vec2 ndc = clip.xy / clip.w / 1.15;
    if (any(greaterThan(abs(ndc), vec2(1)))) return;
    if (phase == 1) {
        atomicMin(selected[2u*cells], floatBitsToUint(ndc.x+1.0));
        atomicMin(selected[2u*cells+1u], floatBitsToUint(ndc.y+1.0));
        atomicMin(selected[2u*cells+2u], floatBitsToUint(1.0-ndc.x));
        atomicMin(selected[2u*cells+3u], floatBitsToUint(1.0-ndc.y));
        return;
    }
    vec2 lower = vec2(uintBitsToFloat(selected[2u*cells]), uintBitsToFloat(selected[2u*cells+1u]))-1.0;
    vec2 upper = 1.0-vec2(uintBitsToFloat(selected[2u*cells+2u]), uintBitsToFloat(selected[2u*cells+3u]));
    vec2 cell_uv = clamp((ndc-lower)/max(upper-lower,vec2(0.001)), 0.0, 0.999999);
    uvec2 pixel = uvec2(cell_uv*vec2(cells_x,cells_y));
    uint cell = pixel.y*cells_x+pixel.x;
    // Separate parity reservoirs retain two distinct overlapping footprints per tile.
    uint depth = uint(clamp(log2(1.0-camera.z)*float((1u << (32u-id_bits))-2u)/16.0, 0.0, float((1u << (32u-id_bits))-2u)));
    uint key = (depth << id_bits) | id;
    atomicMin(selected[cell + (id & 1u)*cells], key);
}
