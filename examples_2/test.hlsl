struct vs_output {
    float4 position : SV_POSITION;
    float4 world_pos : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float3 tangent : TEXCOORD2;
    float3 bitangent : TEXCOORD3;
    float4 texcoord : TEXCOORD4;
};

struct vs_input {
    float4 position : POSITION;
    float4 normal : TEXCOORD0;
    float4 texcoord : TEXCOORD1;
    float4 tangent : TEXCOORD2;
    float4 bitangent : TEXCOORD3;
};

struct sb_sb {
    float4 data;
};

cbuffer per_pass_vs : register(b1) {
    float4x4 projection_matrix;
    float4   test;
};

SamplerState decl_sampler : register(s0);

ConstantBuffer<sbsb> decl_cbuffer_array[] : register(b2);
ConstantBuffer<sbsb> decl_cbuffer_array_bounded[69] : register(b3, space0);

StructuredBuffer<sb_sb> decl_structured_buffer : register(u0, space0);
RWStructuredBuffer<sb_sb> decl_structured_buffer_rw : register(u1, space0);

Texture1D decl_texture1d : register(t0);
Texture2D decl_texture2d : register(t1);
Texture3D decl_texture3d : register(t2);

void test_func() {
    decl_texture1d;
    decl_cbuffer_array;
    decl_cbuffer_array_bounded;
}

void test_func2() {
    decl_texture1d;
    decl_cbuffer_array;
    decl_cbuffer_array_bounded;
}

vs_output vs_main(vs_input input) {
    test_func();
}

ps_output ps_main() {
    test_func2();
}