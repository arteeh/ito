#version 430
uniform sampler2D gaussians;
uniform uint stride;
uniform int sh_degree;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 eye;
uniform vec2 viewport;
layout(std430, binding = 0) readonly buffer Order { uvec2 order[]; };
out vec2 gaussian_uv;
flat out vec4 color_opacity;

vec4 attribute_at(uint id, uint offset) {
    uint address = id * stride + offset;
    uint width = uint(textureSize(gaussians, 0).x);
    return texelFetch(gaussians, ivec2(address % width, address / width), 0);
}
vec3 color(uint id, vec3 direction) {
    float x = direction.x, y = direction.y, z = direction.z;
    float basis[16];
    basis[0] = 0.2820947918;
    basis[1] = -0.4886025119 * y;
    basis[2] = 0.4886025119 * z;
    basis[3] = -0.4886025119 * x;
    basis[4] = 1.0925484306 * x * y;
    basis[5] = -1.0925484306 * y * z;
    basis[6] = 0.3153915653 * (2*z*z - x*x - y*y);
    basis[7] = -1.0925484306 * x * z;
    basis[8] = 0.5462742153 * (x*x - y*y);
    basis[9] = -0.5900435899 * y * (3*x*x - y*y);
    basis[10] = 2.8906114426 * x * y * z;
    basis[11] = -0.4570457995 * y * (4*z*z - x*x - y*y);
    basis[12] = 0.3731763326 * z * (2*z*z - 3*x*x - 3*y*y);
    basis[13] = -0.4570457995 * x * (4*z*z - x*x - y*y);
    basis[14] = 1.4453057213 * z * (x*x - y*y);
    basis[15] = -0.5900435899 * x * (x*x - 3*y*y);
    vec3 rgb = vec3(0.5);
    for (int k = 0; k < (sh_degree + 1)*(sh_degree + 1); ++k)
        rgb += basis[k] * attribute_at(id, uint(3+k)).rgb;
    return max(rgb, vec3(0));
}
void main() {
    uint id = order[gl_InstanceID].y;
    vec4 p = attribute_at(id, 0u);
    vec4 center = view * vec4(p.xyz, 1);
    vec4 clip = projection * center;
    gaussian_uv = vec2(0);
    color_opacity = vec4(0);
    gl_Position = vec4(2, 2, 2, 1);
    if (clip.w <= 0 || clip.z < -clip.w || clip.z > clip.w || p.w < 1.0/255.0) return;

    vec3 s = attribute_at(id, 1u).xyz;
    vec4 q = attribute_at(id, 2u); // wxyz
    float w = q.x, x = q.y, y = q.z, z = q.w;
    mat3 rotation = mat3(
        1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y),
        2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x),
        2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y));
    mat3 transform = mat3(view) * rotation * mat3(s.x,0,0, 0,s.y,0, 0,0,s.z);
    mat3 covariance = transform * transpose(transform);
    // Perspective Jacobian in pixels, including asymmetric stereo frusta.
    vec3 row_w = vec3(projection[0][3], projection[1][3], projection[2][3]);
    vec3 row_x = vec3(projection[0][0], projection[1][0], projection[2][0]);
    vec3 row_y = vec3(projection[0][1], projection[1][1], projection[2][1]);
    vec3 jx = (row_x - clip.x / clip.w * row_w) * (viewport.x * 0.5 / clip.w);
    vec3 jy = (row_y - clip.y / clip.w * row_w) * (viewport.y * 0.5 / clip.w);
    float a = dot(jx, covariance * jx) + 0.3;
    float b = dot(jx, covariance * jy);
    float c = dot(jy, covariance * jy) + 0.3;
    float mid = 0.5*(a+c);
    float radius = length(vec2(0.5*(a-c), b));
    float major = sqrt(max(mid + radius, 0.3));
    float minor = sqrt(max(mid - radius, 0.3));
    vec2 axis = abs(b) > 1e-6 ? normalize(vec2(b, mid + radius - a))
                             : (a >= c ? vec2(1,0) : vec2(0,1));
    vec2 corner = vec2((gl_VertexID & 1) == 0 ? -1 : 1,
                       (gl_VertexID & 2) == 0 ? -1 : 1);
    gaussian_uv = corner * 3.0;
    vec2 offset = 3.0 * (corner.x * major * axis + corner.y * minor * vec2(-axis.y, axis.x));
    gl_Position = clip + vec4(offset * 2.0 / viewport * clip.w, 0, 0);
    color_opacity = vec4(color(id, normalize(p.xyz - eye)), p.w);
}
