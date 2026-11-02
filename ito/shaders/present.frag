#version 460
// An offscreen image onto a display image (the window's or a headset's): turned upright when
// it holds an OpenGL-ordered view (row 0 at the bottom: common.glsl), and, onto an sRGB image,
// decoded so its stored values are the ones drawn (the camera's own, already sRGB): a display
// image shows the same colours whatever its format. Alpha is premultiplied.
layout(push_constant) uniform Present {
    int upright;
    int srgb;
};
layout(set = 0, binding = 0) uniform sampler2D source;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
vec3 decode(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}
void main() {
    color = texture(source, upright == 1 ? vec2(uv.x, 1.0 - uv.y) : uv);
    if (srgb == 1)
        color.rgb = color.a > 0.0 ? decode(clamp(color.rgb / color.a, 0.0, 1.0)) * color.a : vec3(0.0);
}
