#include <metal_stdlib>
using namespace metal;
// texture
#define texture2d_rw( name, index ) texture2d<float, access::read_write> name [[texture(index)]]
#define texture2d_r( name, index ) texture2d<float, access::read> name [[texture(index)]]
#define texture2d_w( name, index ) texture2d<float, access::write> name [[texture(index)]]
#define read_texture( name, gid ) name.read(gid)
#define write_texture( name, val, gid ) name.write(val, gid)
#define texture_2d( name, sampler_index ) texture2d<float> name [[texture(sampler_index)]], sampler sampler_##name [[sampler(sampler_index)]]
#define texture_3d( name, sampler_index ) texture3d<float> name [[texture(sampler_index)]], sampler sampler_##name [[sampler(sampler_index)]]
#define texture_2dms( type, samples, name, sampler_index ) texture2d_ms<float> name [[texture(sampler_index)]], sampler sampler_##name [[sampler(sampler_index)]]
#define texture_cube( name, sampler_index ) texturecube<float> name [[texture(sampler_index)]], sampler sampler_##name [[sampler(sampler_index)]]
#define texture_2d_array( name, sampler_index ) texture2d_array<float> name [[texture(sampler_index)]], sampler sampler_##name [[sampler(sampler_index)]]
#define texture_2d_arg(name) thread texture2d<float>& name, thread sampler& sampler_##name
#define texture_3d_arg(name) thread texture3d<float>& name, thread sampler& sampler_##name
#define texture_2dms_arg(name) thread texture2d_ms<float>& name, thread sampler& sampler_##name
#define texture_cube_arg(name) thread texturecube<float>& name, thread sampler& sampler_##name
#define texture_2d_array_arg(name) thread texture2d_array<float>& name, thread sampler& sampler_##name
// structured buffers
#define structured_buffer_rw( type, name, index ) device type* name [[buffer(index)]]
#define structured_buffer_rw_arg( type, name, index ) device type* name [[buffer(index)]]
#define structured_buffer( type, name, index ) constant type& name [[buffer(index)]]
#define structured_buffer_arg( type, name, index ) constant type& name [[buffer(index)]]
// sampler
#define sample_texture( name, tc ) name.sample(sampler_##name, tc)
#define sample_texture_2dms( name, x, y, fragment ) name.read(uint2(x, y), fragment)
#define sample_texture_level( name, tc, l ) name.sample(sampler_##name, tc, level(l))
#define sample_texture_grad( name, tc, vddx, vddy ) name.sample(sampler_##name, tc, gradient3d(vddx, vddy))
#define sample_texture_array( name, tc, a ) name.sample(sampler_##name, tc, uint(a))
#define sample_texture_array_level( name, tc, a, l ) name.sample(sampler_##name, tc, uint(a), level(l))
// matrix
#define to_3x3( M4 ) float3x3(M4[0].xyz, M4[1].xyz, M4[2].xyz)
#define from_columns_3x3(A, B, C) (transpose(float3x3(A, B, C)))
#define from_rows_3x3(A, B, C) (float3x3(A, B, C))
#define mul( A, B ) ((A) * (B))
#define mul_tbn( A, B ) ((B) * (A))
#define unpack_vb_instance_mat( mat, r0, r1, r2, r3 ) mat[0] = r0; mat[1] = r1; mat[2] = r2; mat[3] = r3;
#define to_data_matrix(mat) mat
// clip
#define remap_z_clip_space( d ) (d = d * 0.5 + 0.5)
#define remap_ndc_ray( r ) float2(r.x, r.y)
#define remap_depth( d ) (d)
// defs
#define ddx dfdx
#define ddy dfdy
#define discard discard_fragment
#define lerp mix
#define frac fract
#define mod(x, y) (x - y * floor(x/y))
#define _pmfx_unroll
//GENERIC MACROS
#define chebyshev_normalize( V ) (V.xyz / max( max(abs(V.x), abs(V.y)), abs(V.z) ))
#define max3(v) max(max(v.x, v.y),v.z)
#define max4(v) max(max(max(v.x, v.y),v.z), v.w)
struct light_data
{
    float4 pos_radius;
    float4 dir_cutoff;
    float4 colour;
    float4 data;
};
struct distance_field_shadow
{
    float4x4 world_matrix;
    float4x4 world_matrix_inv;
};
struct area_light_data
{
    float4 corners[4];
    float4 colour;
};
struct c_skinning_info
{
    float4x4 bones[85];
};
struct c_per_pass_view
{
    float4x4 vp_matrix;
    float4x4 view_matrix;
    float4x4 vp_matrix_inverse;
    float4x4 view_matrix_inverse;
    float4 camera_view_pos;
    float4 camera_view_dir;
    float4 viewport_correction;
};
struct c_per_draw_call
{
    float4x4 world_matrix;
    float4 user_data;
    float4 user_data2;
    float4x4 world_matrix_inv_transpose;
};
struct c_material_data
{
    float4 m_albedo;
    float2 m_uv_scale;
    float m_roughness;
    float m_reflectivity;
    float m_surface_offset;
    float3 m_padding;
};
struct packed_vs_input_multi
{
    float4 position [[attribute(0)]];
    float4 normal [[attribute(1)]];
    float4 texcoord [[attribute(2)]];
    float4 tangent [[attribute(3)]];
    float4 bitangent [[attribute(4)]];
    float4 blend_indices [[attribute(5)]];
    float4 blend_weights [[attribute(6)]];
    float4 world_matrix_0 [[attribute(7)]];
    float4 world_matrix_1 [[attribute(8)]];
    float4 world_matrix_2 [[attribute(9)]];
    float4 world_matrix_3 [[attribute(10)]];
    float4 user_data [[attribute(11)]];
    float4 user_data2 [[attribute(12)]];
};
struct vs_input_multi
{
    float4 position;
    float4 normal;
    float4 texcoord;
    float4 tangent;
    float4 bitangent;
    float4 blend_indices;
    float4 blend_weights;
};
struct vs_instance_input
{
    float4 world_matrix_0;
    float4 world_matrix_1;
    float4 world_matrix_2;
    float4 world_matrix_3;
    float4 user_data;
    float4 user_data2;
};
struct vs_output
{
    float4 position [[position]];
    float4 world_pos;
    float3 normal;
    float3 tangent;
    float3 bitangent;
    float4 texcoord;
    float4 colour;
};
float4 skin_pos(constant float4x4* bones,
float4 pos,
float4 weights,
float4 indices)
{
    int bone_indices[4];
    bone_indices[0] = int(indices.x);
    bone_indices[1] = int(indices.y);
    bone_indices[2] = int(indices.z);
    bone_indices[3] = int(indices.w);
    float4 sp = float4( 0.0, 0.0, 0.0, 0.0 );
    float final_weight = 1.0;
    for(int i = 3; i >= 0; --i)
    {
        sp += mul( pos, bones[bone_indices[i]] ) * weights[i];
        final_weight -= weights[i];
    }
    sp += mul( pos, bones[bone_indices[0]] ) * final_weight;
    sp.w = 1.0;
    return sp;
}
void skin_tbn(constant float4x4* bones,
thread float3& t,
thread float3& b,
thread float3& n,
float4 weights,
float4 indices)
{
    int bone_indices[4];
    bone_indices[0] = int(indices.x);
    bone_indices[1] = int(indices.y);
    bone_indices[2] = int(indices.z);
    bone_indices[3] = int(indices.w);
    float3 rt = float3( 0.0, 0.0, 0.0);
    float3 rb = float3( 0.0, 0.0, 0.0);
    float3 rn = float3( 0.0, 0.0, 0.0);
    float final_weight = 1.0;
    for( int i = 0; i < 3; ++i)
    {
        float3x3 rot_mat = to_3x3(bones[bone_indices[i]]);
        rt += mul(t, rot_mat) * weights[i];
        rb += mul(b, rot_mat) * weights[i];
        rn += mul(n, rot_mat) * weights[i];
        final_weight -= weights[i];
    }
    float3x3 rot_mat = to_3x3(bones[bone_indices[3]]);
    rt += mul(t, rot_mat) * final_weight;
    rb += mul(b, rot_mat) * final_weight;
    rn += mul(n, rot_mat) * final_weight;
    t = rt;
    b = rb;
    n = rn;
}
vertex vs_output vs_main(
uint vid [[vertex_id]]
, uint iid [[instance_id]]
, packed_vs_input_multi in_vertex [[stage_in]]
, constant c_skinning_info &skinning_info [[buffer(10)]]
, constant c_per_pass_view &per_pass_view [[buffer(8)]]
, constant c_per_draw_call &per_draw_call [[buffer(9)]]
, constant c_material_data &material_data [[buffer(15)]])
{
    vs_input_multi input;
    vs_instance_input instance_input;
    input.position = float4(in_vertex.position);
    input.normal = float4(in_vertex.normal);
    input.texcoord = float4(in_vertex.texcoord);
    input.tangent = float4(in_vertex.tangent);
    input.bitangent = float4(in_vertex.bitangent);
    input.blend_indices = float4(in_vertex.blend_indices);
    input.blend_weights = float4(in_vertex.blend_weights);
    instance_input.world_matrix_0 = float4(in_vertex.world_matrix_0);
    instance_input.world_matrix_1 = float4(in_vertex.world_matrix_1);
    instance_input.world_matrix_2 = float4(in_vertex.world_matrix_2);
    instance_input.world_matrix_3 = float4(in_vertex.world_matrix_3);
    instance_input.user_data = float4(in_vertex.user_data);
    instance_input.user_data2 = float4(in_vertex.user_data2);
    constant float4x4* bones = &skinning_info.bones[0];
    constant float4x4& vp_matrix = per_pass_view.vp_matrix;
    constant float4x4& world_matrix = per_draw_call.world_matrix;
    constant float4& user_data2 = per_draw_call.user_data2;
    constant float2& m_uv_scale = material_data.m_uv_scale;
    vs_output output;
    float4x4 wvp = mul( world_matrix, vp_matrix );
    float4x4 wm = world_matrix;
    output.texcoord = float4(input.texcoord.x, 1.0 - input.texcoord.y,
    input.texcoord.z, 1.0 - input.texcoord.w );
    float4x4 instance_world_mat;
    unpack_vb_instance_mat(
    instance_world_mat,
    instance_input.world_matrix_0,
    instance_input.world_matrix_1,
    instance_input.world_matrix_2,
    instance_input.world_matrix_3
    );
    wvp = mul( instance_world_mat, vp_matrix );
    wm = instance_world_mat;
    output.colour = instance_input.user_data2;
    float4 sp = skin_pos(bones, input.position, input.blend_weights, input.blend_indices);
    output.tangent = input.tangent.xyz;
    output.bitangent = input.bitangent.xyz;
    output.normal = input.normal.xyz;
    skin_tbn(bones, output.tangent, output.bitangent, output.normal, input.blend_weights, input.blend_indices);
    output.position = mul( sp, vp_matrix );
    output.world_pos = sp;
    float3 scale = float3(length(world_matrix[0].xyz),
    length(world_matrix[1].xyz),
    length(world_matrix[2].xyz));
    float xs = length(input.tangent.xyz * scale);
    float ys = length(input.bitangent.xyz * scale);
    output.texcoord *= float4(m_uv_scale.x * xs, m_uv_scale.y * ys, m_uv_scale.x, m_uv_scale.y);
    return output;
}
