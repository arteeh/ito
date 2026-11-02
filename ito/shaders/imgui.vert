#version 460
layout(push_constant) uniform Ui {
    vec2 scale;
    vec2 translate;
};
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texcoord;
layout(location = 2) in vec4 colour;
layout(location = 0) out vec4 v_colour;
layout(location = 1) out vec2 v_uv;
void main() {
    v_colour = colour;
    v_uv = texcoord;
    gl_Position = vec4(position * scale + translate, 0, 1);
}
