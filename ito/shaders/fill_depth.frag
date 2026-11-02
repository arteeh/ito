#version 460
layout(location = 0) out vec2 result;
void main() { result = vec2(gl_FragCoord.w, 1.0); }  // 1 / depth along the view axis
