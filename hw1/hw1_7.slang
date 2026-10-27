import hw1_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  uniform int samples_per_axis,
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

    float3 result = background;
    float3 accum = float3(0.0);
    int N = samples_per_axis;
    float step = 1.0 / float(N);

    for (int sy = 0; sy < N; sy++) {
        for (int sx = 0; sx < N; sx++) {
            float2 offset = float2((float(sx) + 0.5) * step,
                                   (float(sy) + 0.5) * step);
            float2 p_world = float2(tid.x, tid.y) + offset;
            float3 sub = background;
            float transmittance = 1.0;

            for (int s = 0; s < num_shapes; s++) {
                Shape shape = shapes[s];
                float3 p_h = mul(shape.world_to_obj, float3(p_world, 1.0));
                float2 p = p_h.xy;

                bool hit_fill = false;
                bool hit_stroke = false;

                if (shape.type == ShapeType.Circle) {
                    Circle c = circles[shape.index];
                    float2 d = p - c.center;
                    float dist_sq = dot(d, d);
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

                // fill first, stroke on top
                if (hit_fill && shape.use_fill) {
                    float a = shape.fill_rgba.a;
                    sub += transmittance * a * shape.fill_rgba.rgb;
                    transmittance *= (1.0 - a);
                }
                if (hit_stroke && shape.use_stroke) {
                    float a = shape.stroke_rgba.a;
                    sub += transmittance * a * shape.stroke_rgba.rgb;
                    transmittance *= (1.0 - a);
                }
            }

            // background fills light
            sub += transmittance * background;
            accum += sub;
        }
    }

    result = accum / float(N * N);
    image[tid] = float4(result, 1.0);
}
