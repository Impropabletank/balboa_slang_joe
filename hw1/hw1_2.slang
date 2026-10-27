import hw1_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  StructuredBuffer<float2> polyline,
                  uniform int num_polyline,
                  uniform bool is_closed,
                  uniform bool use_fill_color,
                  uniform float3 fill_color,
                  uniform bool use_stroke_color,
                  uniform float3 stroke_color,
                  uniform float stroke_width,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    image[tid] = float4(1.0);
    // TODO: your code here

    float2 p = float2(tid.x + 0.5, tid.y + 0.5);
    float3 background = float3(0.5, 0.5, 0.5);

    // rmember nonzero winding rule for later
    bool inside_fill = false;
    if (use_fill_color && is_closed) {
        int winding = 0;
        int n = num_polyline;
        for (int i = 0; i < n; i++) {
            int j = (i + 1) % n;
            float2 a = polyline[i];
            float2 b = polyline[j];
            if (a.y <= p.y) {
                if (b.y > p.y && (b.x - a.x) * (p.y - a.y) - (p.x - a.x) * (b.y - a.y) > 0.0) {
                    winding++;
                }
            } else {
                if (b.y <= p.y && (b.x - a.x) * (p.y - a.y) - (p.x - a.x) * (b.y - a.y) < 0.0) {
                    winding--;
                }
            }
        }
        inside_fill = (winding != 0);
    }

    // stroke is distance to nearest segment
    bool inside_stroke = false;
    if (use_stroke_color) {
        float half_w = stroke_width * 0.5;
        int seg_count = is_closed ? num_polyline : num_polyline - 1;
        for (int i = 0; i < seg_count; i++) {

            float2 a = polyline[i];
            float2 b = polyline[(i + 1) % num_polyline];
            float2 ab = b - a;

            float t = clamp(dot(p - a, ab) / dot(ab, ab), 0.0, 1.0);
            float2 closest = a + t * ab;

            if (dot(p - closest, p - closest) <= half_w * half_w) {
                inside_stroke = true;
                break;
            }
        }
    }

    float3 result = background;
    if (inside_fill) result = fill_color;
    if (inside_stroke) result = stroke_color;

    image[tid] = float4(result, 1.0);
}
