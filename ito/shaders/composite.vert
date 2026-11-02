#version 460
// A triangle over the whole target at depth 1.0 (the clear value): with the depth test <= it
// reaches only pixels nothing was drawn in.
void main() {
    vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 1.0, 1.0);
}
