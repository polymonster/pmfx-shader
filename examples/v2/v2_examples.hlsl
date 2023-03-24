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

struct instance_input {
    float4 mat0 : TEXCOORD4;
    float4 mat2 : TEXCOORD5;
    float4 mat3 : TEXCOORD6;
};

struct ps_output {
    float4 colour : SV_Target;
};

struct sb_sb {
    float4 data;
};

cbuffer per_pass_vs : register(b1) {
    float4x4 projection_matrix;
    float4   test;
};

struct cbuffer_data {
    float4 data2;
};

SamplerState decl_sampler : register(s0);

ConstantBuffer<sb_sb> decl_cbuffer_array[] : register(b2);
ConstantBuffer<sb_sb> decl_cbuffer_array_bounded[69] : register(b3, space0);

StructuredBuffer<sb_sb> decl_structured_buffer : register(u0, space0);
RWStructuredBuffer<sb_sb> decl_structured_buffer_rw : register(u1, space0);

Texture1D decl_texture1d : register(t0);
Texture2D decl_texture2d : register(t1);
Texture3D decl_texture3d : register(t2);

// alias resource types types on t0
Texture2D textures[] : register(t0);
TextureCube cubemaps[] : register(t0);
Texture2DArray texture_arrays[] : register(t0);
Texture3D volume_textures[] : register(t0);
ConstantBuffer<cbuffer_data> cbuffers[] : register(b0);

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

vs_output vs_main_permutations(vs_input input) {
    if:(SKINNED) {
        test_func();
        return vs_output_default();
    }
    else if:(INSTANCED) {
        return vs_output_default();
    }
    return vs_output_default();
}

vs_output vs_output_default() {
    vs_output output;
    output.position = float4(0.0, 0.0, 0.0, 1.0);
    output.world_pos = float4(0.0, 0.0, 0.0, 1.0);
    output.normal = float4(0.0, 0.0, 0.0, 1.0);
    output.tangent = float4(0.0, 0.0, 0.0, 1.0);
    output.bitangent = float4(0.0, 0.0, 0.0, 1.0);
    output.texcoord = float4(0.0, 0.0, 0.0, 1.0);
    return output;
}

ps_output ps_output_default() {
    ps_output output;
    output.colour = float4(0.0, 0.0, 0.0, 1.0);
    return output;
}

vs_output vs_main(vs_input input, instance_input mat) {
    test_func();
    return vs_output_default();
}

vs_output vs_main_mixed_semantics(vs_input input, uint iid : SV_InstanceID) {
    test_func();
    return vs_output_default();
}

vs_output vs_main_separate_elements(float4 pos : POSITION, float4 tex : TEXCOORD0) {
    test_func();
    return vs_output_default();
}

ps_output ps_main() {
    test_func2();
    return ps_output_default();
}

vs_output vs_test_bindless_aliasing(vs_input input) {
    return vs_output_default();
}

ps_output ps_test_bindless_aliasing() {
    test_func();
    // using this void cast touches the resources so they are detected by pmfx and compiled in
    (textures);
    (cubemaps);
    (texture_arrays);
    (volume_textures);

    // cbuffers go on a different slot
    (cbuffers);
    return ps_output_default();
}