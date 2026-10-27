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
                  StructuredBuffer<QuadraticBezier> quadratic_beziers,
                  StructuredBuffer<float2> points,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    image[tid] = float4(1.0);
    // TODO: your code here
    float step = 1.0 / float(samples_per_axis);
    float3 accum = float3(0.0);

    for (int sy = 0; sy < samples_per_axis; sy++) {
        for (int sx = 0; sx < samples_per_axis; sx++) {
            float2 offset = float2((float(sx) + 0.5) * step,
                                   (float(sy) + 0.5) * step);
            float2 p_world = float2(tid.x, tid.y) + offset;

            float3 sub = float3(0.0);
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

                } else if (shape.type == ShapeType.QuadraticBezier) {
                    QuadraticBezier qb = quadratic_beziers[shape.index];
                    int n = qb.pts_end - qb.pts_begin;

                    // Curve k uses (p[2k], p[2k+1], p[2k+2]).

                    int num_curves = (n - 1) / 2; //num curves
                    if (qb.closed) {
                        num_curves = n / 2;
                    }

                    // horizontal ray for winding number
                    if (shape.use_fill && qb.closed) {
                        int winding = 0;
                        for (int k = 0; k < num_curves; k++) {
                            float2 p0 = points[qb.pts_begin + (2 * k) % n];
                            float2 p1 = points[qb.pts_begin + (2 * k + 1) % n];
                            float2 p2 = points[qb.pts_begin + (2 * k + 2) % n];

                            // q_y(t) - p.y = 0 → quadratic in t
                            float a = p0.y - 2.0 * p1.y + p2.y;
                            float b = -2.0 * p0.y + 2.0 * p1.y;
                            float c = p0.y - p.y;

                            float2 roots;
                            if (solve_quadratic(a, b, c, roots)) {
                                for (int r = 0; r < 2; r++) {
                                    float t = roots[r];
                                    if (t > 0.0 && t < 1.0) {
                                        float omt = 1.0 - t;
                                        float xt = omt * omt * p0.x + 2.0 * omt * t * p1.x + t * t * p2.x;
                                        if (xt > p.x) {
                                            float dydt = 2.0 * omt * (p1.y - p0.y) + 2.0 * t * (p2.y - p1.y);
                                            if (dydt > 0.0) winding++;
                                            else if (dydt < 0.0) winding--;
                                        }
                                    }
                                }
                            }
                        }
                        hit_fill = (winding != 0);
                    }

                    if (shape.use_stroke) {
                        float half_w = shape.stroke_width * 0.5;
                        for (int k = 0; k < num_curves; k++) {
                            float2 p0 = points[qb.pts_begin + (2 * k) % n];
                            float2 p1 = points[qb.pts_begin + (2 * k + 1) % n];
                            float2 p2 = points[qb.pts_begin + (2 * k + 2) % n];

                            float2 A = p0 - 2.0 * p1 + p2;
                            float2 B = -2.0 * p0 + 2.0 * p1;
                            float2 CmP = p0 - p;

                            float c3 = 2.0 * dot(A, A);
                            float c2 = 3.0 * dot(A, B);
                            float c1 = dot(B, B) + 2.0 * dot(A, CmP);
                            float c0 = dot(B, CmP);

                            float3 roots;
                            int nr = solve_cubic(c3, c2, c1, c0, roots);
                            for (int r = 0; r < nr; r++) {
                                float t = roots[r];
                                if (t >= 0.0 && t <= 1.0) {
                                    float omt = 1.0 - t;
                                    float2 qt = omt * omt * p0 + 2.0 * omt * t * p1 + t * t * p2;
                                    if (dot(qt - p, qt - p) <= half_w * half_w) {
                                        hit_stroke = true;
                                        break;
                                    }
                                }
                            }
                            if (hit_stroke) break;

                            // check the endpoints for caps
                            if (dot(p0 - p, p0 - p) <= half_w * half_w || dot(p2 - p, p2 - p) <= half_w * half_w) {
                                hit_stroke = true;
                                break;
                            }
                        }
                    }
                }

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

            sub += transmittance * background;
            accum += sub;
        }
    }

    image[tid] = float4(accum / float(samples_per_axis * samples_per_axis), 1.0);
}
