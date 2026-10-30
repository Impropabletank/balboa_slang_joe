[shader("compute")]
[numthreads(64, 1, 1)]
void clear_buffers(uint tid : SV_DispatchThreadID,
                   uniform uint num_pixels,
                   RWStructuredBuffer<uint> depth_buffer,
                   RWStructuredBuffer<uint> triangle_id_buffer) {
    if (tid >= num_pixels)
        return;

    depth_buffer[tid] = 0xffffffff;
    triangle_id_buffer[tid] = 0xffffffff;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void transform_vertices(
    uint tid : SV_DispatchThreadID,
    uniform uint num_vertices,
    uniform uint vertex_offset,
    uniform float4x4 MVP,
    StructuredBuffer<float3> vertices,
    RWStructuredBuffer<float3> projected_vertices)
{
    if (tid >= num_vertices)
        return;

    uint i = vertex_offset + tid;
    float4 p = mul(MVP, float4(vertices[i], 1.0));
    projected_vertices[i] = p.xyz / p.w;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_depth(
        uint tid : SV_DispatchThreadID,
        uniform uint num_triangles,
        uniform uint width,
        uniform uint height,
        uniform float s,
        uniform float a,
        StructuredBuffer<uint3> faces,
        // A projected vertex store the screen space coordinates
        // in xy, and the inverse depth in z.
        StructuredBuffer<float3> projected_vertices,
        RWStructuredBuffer<uint> depth_buffer) 
{
    // HW 2.5 TODO
    uint face_id = tid;
    if (face_id >= num_triangles) return;

    uint3 face = faces[face_id];
    float3 v0 = projected_vertices[face.x];
    float3 v1 = projected_vertices[face.y];
    float3 v2 = projected_vertices[face.z];

    if (v0.z <= 0.0 || v1.z <= 0.0 || v2.z <= 0.0) return;

    float2 p0 = v0.xy;
    float2 p1 = v1.xy;
    float2 p2 = v2.xy;

    int2 bb_min = int2(floor(min(min(p0, p1), p2)));
    int2 bb_max = int2(ceil (max(max(p0, p1), p2)));
    bb_min = max(bb_min, int2(0, 0));
    bb_max = min(bb_max, int2(int(width) - 1, int(height) - 1));

    float2 e01 = p1 - p0;
    float2 e12 = p2 - p1;
    float2 e20 = p0 - p2;
    float2 n01 = float2(e01.y, -e01.x);
    float2 n12 = float2(e12.y, -e12.x);
    float2 n20 = float2(e20.y, -e20.x);
    float area = dot(p2 - p0, n01);
    if (abs(area) < 1e-8) return;

    for (int y = bb_min.y; y <= bb_max.y; y++) {
        for (int x = bb_min.x; x <= bb_max.x; x++) {
            float2 p = float2(x, y) + float2(0.5, 0.5);
            float E01 = dot(p - p0, n01);
            float E12 = dot(p - p1, n12);
            float E20 = dot(p - p2, n20);
            bool inside = (E01 >= 0 && E12 >= 0 && E20 >= 0) ||
                          (E01 <= 0 && E12 <= 0 && E20 <= 0);
            if (!inside) continue;

            float b0p = E12 / area;
            float b1p = E20 / area;
            float b2p = E01 / area;
            float denom = b0p * v0.z + b1p * v1.z + b2p * v2.z;
            if (denom <= 0.0) continue;
            float depth = 1.0 / denom;

            uint idx = y * width + x;
            InterlockedMin(depth_buffer[idx], asuint(depth));
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_id(
        uint tid : SV_DispatchThreadID,
        uniform uint num_triangles,
        uniform uint width,
        uniform uint height,
        uniform float s,
        uniform float a,
        StructuredBuffer<uint3> faces,
        StructuredBuffer<float3> projected_vertices,
        RWStructuredBuffer<uint> depth_buffer,
        RWStructuredBuffer<uint> triangle_id_buffer) 
{
    // HW 2.5 TODO
    uint face_id = tid;
    if (face_id >= num_triangles) return;

    uint3 face = faces[face_id];
    float3 v0 = projected_vertices[face.x];
    float3 v1 = projected_vertices[face.y];
    float3 v2 = projected_vertices[face.z];

    if (v0.z <= 0.0 || v1.z <= 0.0 || v2.z <= 0.0) return;

    float2 p0 = v0.xy;
    float2 p1 = v1.xy;
    float2 p2 = v2.xy;

    int2 bb_min = int2(floor(min(min(p0, p1), p2)));
    int2 bb_max = int2(ceil (max(max(p0, p1), p2)));
    bb_min = max(bb_min, int2(0, 0));
    bb_max = min(bb_max, int2(int(width) - 1, int(height) - 1));

    float2 e01 = p1 - p0;
    float2 e12 = p2 - p1;
    float2 e20 = p0 - p2;
    float2 n01 = float2(e01.y, -e01.x);
    float2 n12 = float2(e12.y, -e12.x);
    float2 n20 = float2(e20.y, -e20.x);
    float area = dot(p2 - p0, n01);
    if (abs(area) < 1e-8) return;

    for (int y = bb_min.y; y <= bb_max.y; y++) {
        for (int x = bb_min.x; x <= bb_max.x; x++) {
            float2 p = float2(x, y) + float2(0.5, 0.5);
            float E01 = dot(p - p0, n01);
            float E12 = dot(p - p1, n12);
            float E20 = dot(p - p2, n20);
            bool inside = (E01 >= 0 && E12 >= 0 && E20 >= 0) ||
                          (E01 <= 0 && E12 <= 0 && E20 <= 0);
            if (!inside) continue;

            float b0p = E12 / area;
            float b1p = E20 / area;
            float b2p = E01 / area;
            float denom = b0p * v0.z + b1p * v1.z + b2p * v2.z;
            if (denom <= 0.0) continue;
            float depth = 1.0 / denom;

            uint idx = y * width + x;
            uint old_depth = 0;
            InterlockedAdd(depth_buffer[idx], 0u, old_depth);
            if (asuint(depth) == old_depth) {
                InterlockedMin(triangle_id_buffer[idx], face_id);
            }
        }
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void shade(
        uint2 tid : SV_DispatchThreadID,
        uniform uint width,
        uniform uint height,
        uniform float s,
        uniform float a,
        StructuredBuffer<uint3> faces,
        StructuredBuffer<float3> vertex_colors,
        StructuredBuffer<float3> projected_vertices,
        StructuredBuffer<uint> triangle_id_buffer,
        RWTexture2D<float4> image) {
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    // HW 2.5 TODO
    
    uint idx = tid.y * width + tid.x;
    uint face_id = triangle_id_buffer[idx];
    if (face_id == 0xffffffff) {
        return;
    }

    uint3 face = faces[face_id];
    float3 v0 = projected_vertices[face.x];
    float3 v1 = projected_vertices[face.y];
    float3 v2 = projected_vertices[face.z];
    float2 p0 = v0.xy;
    float2 p1 = v1.xy;
    float2 p2 = v2.xy;

    float2 p = float2(tid) + float2(0.5, 0.5);
    float2 e01 = p1 - p0;
    float2 e12 = p2 - p1;
    float2 e20 = p0 - p2;
    float2 n01 = float2(e01.y, -e01.x);
    float2 n12 = float2(e12.y, -e12.x);
    float2 n20 = float2(e20.y, -e20.x);
    float area = dot(p2 - p0, n01);

    float E01 = dot(p - p0, n01);
    float E12 = dot(p - p1, n12);
    float E20 = dot(p - p2, n20);

    float b0p = E12 / area;
    float b1p = E20 / area;
    float b2p = E01 / area;

    float denom = b0p * v0.z + b1p * v1.z + b2p * v2.z;
    float w0 = b0p * v0.z / denom;
    float w1 = b1p * v1.z / denom;
    float w2 = b2p * v2.z / denom;

    float3 c0 = vertex_colors[face.x];
    float3 c1 = vertex_colors[face.y];
    float3 c2 = vertex_colors[face.z];

    float3 color = w0 * c0 + w1 * c1 + w2 * c2;
    image[tid] = float4(color, 1.0);
}
