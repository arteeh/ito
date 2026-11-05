#version 460
layout(location = 0) in vec2 gaussian_uv;
layout(location = 1) flat in vec4 color_opacity;
layout(location = 0) out vec4 frag_color;
void main() {
    float radius2 = dot(gaussian_uv, gaussian_uv);
    if (radius2 > 9.0) discard;
    float alpha = min(0.99, color_opacity.a * exp(-0.5 * radius2));
    if (alpha < 1.0/255.0) discard;
    frag_color = vec4(color_opacity.rgb * alpha, alpha);
}
