import hw1_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  uniform int /*samples_per_axis*/, // ignore for now
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Circle> circles,
                  StructuredBuffer<Polyline> polylines,
                  StructuredBuffer<QuadraticBezier> /*quadratic_beziers*/, // ignore for now
                  StructuredBuffer<float2> points,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    image[tid] = float4(1.0);
    // TODO: your code here

    float2 p_world = float2(tid.x + 0.5, tid.y + 0.5);
    float3 result = background;

    for (int s = 0; s < num_shapes; s++) {
        Shape shape = shapes[s];

        float3 p_h = mul(shape.world_to_obj, float3(p_world, 1.0));
        float2 p = p_h.xy;

        bool hit_fill = false;
        bool hit_stroke = false;
        float3 fill_col = shape.fill_rgba.rgb;
        float3 stroke_col = shape.stroke_rgba.rgb;

        if (shape.type == ShapeType.Circle) {
            Circle c = circles[shape.index];
            float2 d = p - c.center;
            float dist_sq = dot(d, d);
            // SVG: stroke straddles the radius
            float r_out = c.radius + shape.stroke_width * 0.5;
            float r_in  = c.radius - shape.stroke_width * 0.5;
            hit_fill   = dist_sq <= c.radius * c.radius;
            hit_stroke = (dist_sq <= r_out * r_out) && (dist_sq >= r_in * r_in);
        } else if (shape.type == ShapeType.Polyline) {
            Polyline pl = polylines[shape.index];
            int n = pl.pts_end - pl.pts_begin;

            if (shape.use_fill && pl.closed && n >= 3) {
                int winding = 0;
                for (int i = 0; i < n; i++) {
                    int j = (i + 1) % n;
                    float2 a = points[pl.pts_begin + i];
                    float2 b = points[pl.pts_begin + j];
                    if (a.y <= p.y) {
                        if (b.y > p.y && (b.x - a.x) * (p.y - a.y) - (p.x - a.x) * (b.y - a.y) > 0.0)
                            winding++;
                    } else {
                        if (b.y <= p.y && (b.x - a.x) * (p.y - a.y) - (p.x - a.x) * (b.y - a.y) < 0.0)
                            winding--;
                    }
                }
                hit_fill = (winding != 0);
            }

            if (shape.use_stroke && n >= 2) {
                float half_w = shape.stroke_width * 0.5;
                int seg_count = pl.closed ? n : n - 1;
                for (int i = 0; i < seg_count; i++) {
                    float2 a = points[pl.pts_begin + i];
                    float2 b = points[pl.pts_begin + ((i + 1) % n)];
                    float2 ab = b - a;
                    float t = clamp(dot(p - a, ab) / dot(ab, ab), 0.0, 1.0);
                    float2 closest = a + t * ab;
                    if (dot(p - closest, p - closest) <= half_w * half_w) {
                        hit_stroke = true;
                        break;
                    }
                }
            }
        }

        if (hit_fill && shape.use_fill)     result = fill_col;
        if (hit_stroke && shape.use_stroke) result = stroke_col;
    }

    image[tid] = float4(result, 1.0);
}
