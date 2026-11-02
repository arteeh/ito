#version 460
// The guess layer into the view, only where the layer holds a guess.
layout(set = 0, binding = 0) uniform sampler2D layer;
layout(location = 0) out vec4 f_colour;
void main() {
    vec4 c = texelFetch(layer, ivec2(gl_FragCoord.xy), 0);
    if (c.a < 0.5) discard;
    f_colour = vec4(c.rgb, 1.0);
    gl_FragDepth = 1.0;
}
