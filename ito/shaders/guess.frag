#version 460
layout(set = 0, binding = 1) uniform sampler2DArray images;
layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec3 v_world;
layout(location = 2) in float v_keep;
layout(location = 3) in float v_measured;
layout(location = 0) out vec4 f_colour;
void main() {
    if (v_keep < 0.999 || v_measured > 0.999) discard;
    // Alpha 1: this pixel holds a guess (the composite reads it).
    f_colour = vec4(texture(images, vec3(v_uv, 0.0)).rgb, 1.0);
}
