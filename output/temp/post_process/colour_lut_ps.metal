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
struct vs_output
{
    float4 position;
    float4 texcoord;
};
struct ps_output
{
    float4 colour [[color(0)]];
};
fragment ps_output ps_main(vs_output input [[stage_in]]
,  texture_2d( src_texture_0, 0 )
, texture_2d( lookup_texture, 8 ))
{
    ps_output output;
    float2 tc = input.texcoord.xy;
    float4 input_colour = sample_texture(src_texture_0, tc);
    float input_b = input_colour.b * 63.0;
    float2 quad_1;
    quad_1.y = floor(floor(input_b) / 8.0);
    quad_1.x = floor(input_b) - (quad_1.y * 8.0);
    float2 quad_2;
    quad_2.y = floor(ceil(input_b) / 8.0);
    quad_2.x = ceil(input_b) - (quad_2.y * 8.0);
    float2 tc_1;
    tc_1.x = (quad_1.x * 0.125) + 0.5 / 512.0 + ((0.125 - 1.0 / 512.0) * input_colour.r);
    tc_1.y = (quad_1.y * 0.125) + 0.5 / 512.0 + ((0.125 - 1.0 / 512.0) * input_colour.g);
    float2 tc_2;
    tc_2.x = (quad_2.x * 0.125) + 0.5 / 512.0 + ((0.125 - 1.0 / 512.0) * input_colour.r);
    tc_2.y = (quad_2.y * 0.125) + 0.5 / 512.0 + ((0.125 - 1.0 / 512.0) * input_colour.g);
    float4 remapped_1 = sample_texture_level(lookup_texture, tc_1, 0.0);
    float4 remapped_2 = sample_texture_level(lookup_texture, tc_2, 0.0);
    output.colour = lerp(remapped_1, remapped_2, frac(input_b));
    return output;
}
