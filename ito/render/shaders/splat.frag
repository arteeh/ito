#version 430
noperspective in vec2 gaussian_uv;
flat in vec4 color_opacity;
flat in float optical_density;
out vec4 frag_color;
void main() {
    float radius2 = dot(gaussian_uv, gaussian_uv);
    if (radius2 > 9.0) discard;
    float gaussian = exp(-0.5 * radius2);
    float alpha = color_opacity.a * gaussian;
    if (optical_density > 0) {
        // Surfel samples describe opaque surfaces. Integrate Gaussian optical
        // density (Beer-Lambert), so overlapping samples close coverage without
        // inflating their measured footprint. Taper to zero at the support edge.
        float density = max(0.0, (gaussian - exp(-4.5)) / (1.0 - exp(-4.5)));
        alpha = 1.0 - exp(-optical_density * density);
    }
    alpha = min(0.99, alpha);
    if (alpha < 1.0/255.0) discard;
    frag_color = vec4(color_opacity.rgb * alpha, alpha);
}
