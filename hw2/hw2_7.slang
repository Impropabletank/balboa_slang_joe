StructuredBuffer<float3> vertices;
StructuredBuffer<float3> vertex_normals;
StructuredBuffer<float2> vertex_uvs;
uniform float4x4 MVP;
uniform float4x4 model_matrix;
uniform float3x3 normal_matrix; // inverse transpose of the top-left 3x3 above
uniform float3 light_dir;
uniform float3 light_color;
uniform float3 camera_position;
uniform float3 Ks;
uniform float exponent;
Texture2D<float4> Kd_texture;
SamplerState Kd_sampler;

struct VertexOutput {
    float4 position : SV_Position;
    float3 world_position : TEXCOORD0;
    float3 world_normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
};

[shader("vertex")]
VertexOutput vertex_main(uint vid : SV_VertexID) {

    VertexOutput output;

    // HW 2.7 TODO: implement
    
    float3 obj_pos = vertices[vid];
    float3 obj_nrm = vertex_normals[vid];

    float4 world_pos = mul(model_matrix, float4(obj_pos, 1.0));
    float3 world_nrm = normalize(mul(normal_matrix, obj_nrm));

    output.position      = mul(MVP, float4(obj_pos, 1.0));
    output.world_position = world_pos.xyz;
    output.world_normal   = world_nrm;
    output.uv             = vertex_uvs[vid];

    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target {

    // HW 2.7 TODO: implement
    
    float3 n = normalize(input.world_normal);
    float3 l = normalize(light_dir);           // points FROM surface TO light
    float3 v = normalize(camera_position - input.world_position);
    float3 r = reflect(-l, n);

    // Diffuse coefficient Kd from texture (handles both constant and image)
    float3 Kd = Kd_texture.Sample(Kd_sampler, input.uv).rgb;

    // Ambient
    float3 Ca = 0.2 * Kd;

    // Diffuse
    float ndotl = max(0.0, dot(n, l));
    float3 Cd = Kd * light_color * ndotl;

    // Specular
    float3 Cs = float3(0.0);
    if (ndotl > 0.0) {
        float rdotv = max(0.0, dot(r, v));
        Cs = Ks * light_color * pow(rdotv, exponent);
    }

    float3 color = Ca + Cd + Cs;
    return float4(color, 1.0);
}
