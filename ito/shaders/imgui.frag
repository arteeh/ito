#version 460
layout(set = 0, binding = 0) uniform sampler2D atlas;
layout(location = 0) in vec4 v_colour;
layout(location = 1) in vec2 v_uv;
layout(location = 0) out vec4 colour;
void main() { colour = v_colour * texture(atlas, v_uv); }
