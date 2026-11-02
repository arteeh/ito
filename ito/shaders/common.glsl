// Shared by Ito's shaders. The constants are specialization constants: Python owns their values
// (ito/render/vk.py SPECIALIZATION), every pipeline gets them all.

layout(constant_id = 0) const int K = 1;            // projector slots (fusion.PROJECTORS)
layout(constant_id = 1) const int TILE = 64;        // atlas tile side, texels
layout(constant_id = 2) const int LAYER_SIDE = 64;  // tiles per atlas layer side
layout(constant_id = 3) const float TILE_SPAN = 10.0;
layout(constant_id = 4) const float DRAW_FACING = 0.25;
layout(constant_id = 5) const float BAKE_FACING = 0.2;
layout(constant_id = 6) const float BAKE_KEEP = 0.85;
layout(constant_id = 7) const int LAYER = 4096;     // atlas layer side, texels
layout(constant_id = 8) const int STEPS = 24;       // the fill's ray march steps

// Ito's matrices are OpenGL's (clip z from -w to w); Vulkan's depth runs from 0 to w. Its y is
// left as it is: every offscreen image then holds what OpenGL's would, row 0 at clip y = -1, and
// only the final pass onto a display image turns it upright (present.frag).
vec4 clip(vec4 p) {
    p.z = (p.z + p.w) * 0.5;
    return p;
}

// A projector: a camera frame placed in the world (render.MeshRenderer).
struct Projector {
    mat4 camera_from_world;
    vec4 intrinsics;  // fx, fy, cx, cy in image pixels
    vec4 origin;      // xyz, and w the boost: 0 is an empty slot
};

// A tile's texel coordinates (in its atlas layer) of a point at `local` voxels of its block, for
// direction d (axis d / 2, + for even d), and the tile's layer and corner.
ivec3 tile_corner(int slot, int d) {
    int t = slot * 6 + d;
    int within = t % (LAYER_SIDE * LAYER_SIDE);
    return ivec3((within % LAYER_SIDE) * TILE, (within / LAYER_SIDE) * TILE,
                 t / (LAYER_SIDE * LAYER_SIDE));
}

vec2 tile_uv(vec3 local, int d) {
    int a = d / 2;
    vec2 p = a == 0 ? local.yz : a == 1 ? local.zx : local.xy;
    return (p + 1.0) / TILE_SPAN * float(TILE);
}

float along(vec3 n, int d) {
    return (d % 2 == 0 ? 1.0 : -1.0) * n[d / 2];
}
