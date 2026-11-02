#version 460
layout(set = 0, binding = 0) uniform sampler2D camera;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
void main() { color = vec4(texture(camera, uv).rgb, 1); }
