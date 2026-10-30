[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 center,
                  uniform float radius,
                  uniform float3 color,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    image[tid] = float4(1.0);
    // TODO: your code here

    //center in screen 
    float2 pixel_center = float2(tid.x + 0.5, tid.y + 0.5);

    float2 d = pixel_center - center;
    float dist_sq = dot(d, d);

    float3 background = float3(0.5, 0.5, 0.5);

    float3 result;
    if (dist_sq <= radius * radius) {
        result = color; //inside circle
    } else {
        result = background; //outside → gray background
    }

    image[tid] = float4(result, 1.0);
}
