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
    pmfx_touch(decl_texture1d);
    pmfx_touch(decl_cbuffer_array);
    pmfx_touch(decl_cbuffer_array_bounded);
}

void test_func2() {
    pmfx_touch(decl_texture1d);
    pmfx_touch(decl_cbuffer_array);
    pmfx_touch(decl_cbuffer_array_bounded);
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
    output.texcoord = float4(0.0, 0.0, 0.0, 1.0);
    output.normal = float3(0.0, 0.0, 0.0);
    output.tangent = float3(0.0, 0.0, 0.0);
    output.bitangent = float3(0.0, 0.0, 0.0);
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

//
// test that different textures can alias and bind on the same register / 
//

ps_output ps_test_bindless_aliasing() {
    test_func();
    // using this void cast touches the resources so they are detected by pmfx and compiled in
    pmfx_touch(textures);
    pmfx_touch(cubemaps);
    pmfx_touch(texture_arrays);
    pmfx_touch(volume_textures);

    // cbuffers go on a different slot
    pmfx_touch(cbuffers);
    return ps_output_default();
}

//
// test using a raw cbuffer member vs a scoped member and test
// ambiguity 

cbuffer cbuffer_unscoped : register(b0) {
    float4x4 world_matrix;
};

struct cbuffer_struct {
    float4x4 world_matrix;
};

ConstantBuffer<cbuffer_struct> cbuffer_scoped : register(b0);

vs_output vs_test_use_cbuffer_unscoped() {
    vs_output output = vs_output_default();
    output.position = mul(world_matrix, output.position);
    return output;
}

vs_output vs_test_use_cbuffer_scoped() {
    vs_output output = vs_output_default();
    output.position = mul(cbuffer_scoped.world_matrix, output.position);

    // test member 
    float4x4 mat = cbuffer_scoped. world_matrix;

    return output;
}

//
// test nested structures 
//

struct buffer_view {
    uint2 location;
    uint  size;
    uint  strude;
};

struct indirect_args {
    buffer_view ib;
    buffer_view vb;
};

vs_output vs_test_nested_structures() {
    vs_output output = vs_output_default();
    return output;
}

//
// global types / compute sync
//

groupshared uint4 accumulated;

void cs_mip_chain_texture2d(uint2 did: SV_DispatchThreadID) {

    accumulated = uint4(0, 0, 0, 0);

    GroupMemoryBarrierWithGroupSync();

    uint original;
    InterlockedAdd(accumulated.x, 1, original);

    GroupMemoryBarrierWithGroupSync();
}