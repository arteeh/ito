#version 460
layout(push_constant) uniform Shadow {
    mat4 world_to_clip;
    vec4 origin;
};
layout(location = 0) in vec3 v_world;
layout(location = 0) out float range_out;
void main() { range_out = length(v_world - origin.xyz); }
