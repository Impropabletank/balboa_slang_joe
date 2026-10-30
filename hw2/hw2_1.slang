[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 sp0,
                  uniform float2 sp1,
                  uniform float2 sp2,
                  uniform int2 aabb_min,
                  uniform float3 color,
                  RWTexture2D<float4> image) {
    int2 ipixel = int2(tid) + int2(aabb_min);
    uint width, height;
    image.GetDimensions(width, height);
    if (ipixel.x < 0 || ipixel.y < 0 || ipixel.x >= width || ipixel.y >= height) {
        return;
    }

    // TODO: HW 2.1
    float2 pixel_center = float2(ipixel) + float2(0.5, 0.5);

    // triangle edges ex. e01 is from 0 to 1
    float2 e01 = sp1 - sp0;
    float2 e12 = sp2 - sp1;
    float2 e20 = sp0 - sp2;

    float2 n01 = float2(e01.y, -e01.x);
    float2 n12 = float2(e12.y, -e12.x);
    float2 n20 = float2(e20.y, -e20.x);

    float E01 = dot(pixel_center - sp0, n01);
    float E12 = dot(pixel_center - sp1, n12);
    float E20 = dot(pixel_center - sp2, n20);

    // if all three have the same signage (pos or neg)
    // this handles both winding orders
    bool inside = (E01 >= 0.0 && E12 >= 0.0 && E20 >= 0.0) ||
                  (E01 <= 0.0 && E12 <= 0.0 && E20 <= 0.0);

    if (inside) {
        image[ipixel] = float4(color, 1.0);
    }

    // will make image black
    //image[ipixel] = float4(float3(0.0), 1.0);
}
