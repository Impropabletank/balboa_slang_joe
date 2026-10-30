[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 sp0, // vertex at screen space
                  uniform float2 sp1,
                  uniform float2 sp2,
                  uniform float inv_depth0, // inverse depth in camera space
                  uniform float inv_depth1,
                  uniform float inv_depth2,
                  uniform int2 aabb_min,
                  uniform float3 c0, // vertex color
                  uniform float3 c1,
                  uniform float3 c2,
                  uniform float s,
                  uniform float a,
                  RWTexture2D<float4> color_image,
                  RWTexture2D<float> depth_image) {
    int2 ipixel = int2(tid) + int2(aabb_min);
    uint width, height;
    color_image.GetDimensions(width, height);
    if (ipixel.x < 0 || ipixel.y < 0 || ipixel.x >= width || ipixel.y >= height) {
        return;
    }
    



    // TODO: HW 2.1

    float2 p = float2(ipixel) + float2(0.5, 0.5);

    // Edge functions (same as 2.1)
    float2 e01 = sp1 - sp0;
    float2 e12 = sp2 - sp1;
    float2 e20 = sp0 - sp2;
    float2 n01 = float2(e01.y, -e01.x);
    float2 n12 = float2(e12.y, -e12.x);
    float2 n20 = float2(e20.y, -e20.x);

    float E01 = dot(p - sp0, n01);
    float E12 = dot(p - sp1, n12);
    float E20 = dot(p - sp2, n20);

    bool inside = (E01 >= 0.0 && E12 >= 0.0 && E20 >= 0.0) ||
                  (E01 <= 0.0 && E12 <= 0.0 && E20 <= 0.0);
    if (!inside) return;

    // Projected barycentric coordinates: b_i' = area(p, v_{i+1}, v_{i+2}) / area(v0, v1, v2)
    // Using edge functions directly: E_ij(p) = (p - v_i) . n_ij, and the signed area
    //   of the triangle is E01(sp2) = E12(sp0) = E20(sp1).
    // So b0' = E12(p) / E12(sp0), b1' = E20(p) / E20(sp1), b2' = E01(p) / E01(sp2).
    float area01 = dot(sp2 - sp0, n01);
    float area12 = dot(sp0 - sp1, n12);
    float area20 = dot(sp1 - sp2, n20);

    float b0 = E12 / area12;
    float b1 = E20 / area20;
    float b2 = E01 / area01;

    // Perspective-correct interpolation (Eq. 15):
    //   b_i = (b_i' / z_i) / sum_j(b_j' / z_j)
    // where 1/z_i = inv_depth_i.
    float denom = b0 * inv_depth0 + b1 * inv_depth1 + b2 * inv_depth2;
    if (denom <= 0.0) return;
    float w0 = b0 * inv_depth0 / denom;
    float w1 = b1 * inv_depth1 / denom;
    float w2 = b2 * inv_depth2 / denom;

    // Depth (positive; smaller = closer). depth = 1 / denom.
    float depth = 1.0 / denom;

    // Depth test against existing depth (init to +inf)
    float old = depth_image[ipixel];
    bool unwritten = (old == 0.0) || isinf(old);
    if (!unwritten && depth >= old) return;

    // Interpolate color and write both
    float3 color = w0 * c0 + w1 * c1 + w2 * c2;
    color_image[ipixel] = float4(color, 1.0);
    depth_image[ipixel] = depth;

    //color_image[ipixel] = float4(float3(0.0), 1.0);
}
