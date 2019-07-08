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
struct packed_vs_input_multi
{
    float4 position [[attribute(0)]];
    float4 normal [[attribute(1)]];
    float4 texcoord [[attribute(2)]];
    float4 tangent [[attribute(3)]];
    float4 bitangent [[attribute(4)]];
    float4 world_matrix_0 [[attribute(5)]];
    float4 world_matrix_1 [[attribute(6)]];
    float4 world_matrix_2 [[attribute(7)]];
    float4 world_matrix_3 [[attribute(8)]];
    float4 user_data [[attribute(9)]];
    float4 user_data2 [[attribute(10)]];
};
struct vs_input_multi
{
    float4 position;
    float4 normal;
    float4 texcoord;
    float4 tangent;
    float4 bitangent;
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
struct vs_output_zonly
{
    float4 position [[position]];
};
vertex vs_output_zonly vs_main(
uint vid [[vertex_id]]
, uint iid [[instance_id]]
, packed_vs_input_multi in_vertex [[stage_in]]
, constant c_per_pass_view &per_pass_view [[buffer(8)]])
{
    vs_input_multi input;
    vs_instance_input instance_input;
    input.position = float4(in_vertex.position);
    input.normal = float4(in_vertex.normal);
    input.texcoord = float4(in_vertex.texcoord);
    input.tangent = float4(in_vertex.tangent);
    input.bitangent = float4(in_vertex.bitangent);
    instance_input.world_matrix_0 = float4(in_vertex.world_matrix_0);
    instance_input.world_matrix_1 = float4(in_vertex.world_matrix_1);
    instance_input.world_matrix_2 = float4(in_vertex.world_matrix_2);
    instance_input.world_matrix_3 = float4(in_vertex.world_matrix_3);
    instance_input.user_data = float4(in_vertex.user_data);
    instance_input.user_data2 = float4(in_vertex.user_data2);
    constant float4x4& vp_matrix = per_pass_view.vp_matrix;
    vs_output_zonly output;
    float4x4 wvp;
    float4x4 instance_world_mat;
    unpack_vb_instance_mat(
    instance_world_mat,
    instance_input.world_matrix_0,
    instance_input.world_matrix_1,
    instance_input.world_matrix_2,
    instance_input.world_matrix_3
    );
    wvp = mul( instance_world_mat, vp_matrix );
    output.position = mul( input.position, wvp );
    output.position /= output.position.w;
    remap_z_clip_space(output.position.z);
    return output;
}
