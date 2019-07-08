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
struct c_image_ex
{
    float4 colour_mask;
    float4 params;
    float4x4 inverse_wvp;
};
struct vs_output
{
    float4 position;
    float4 colour;
    float2 tex_coord;
};
struct ps_output
{
    float4 colour [[color(0)]];
};
float sd_box(float3 p,
float3 b)
{
    float3 d = abs(p) - b;
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}
void unit_cube_trace(constant float4x4& inverse_wvp,
float2 tc,
thread float3& p,
thread float3& ro,
thread float3& rd)
{
    float2 ndc = tc.xy * float2(2.0, 2.0) - float2(1.0, 1.0);
    ndc = remap_ndc_ray(ndc);
    float4 near = float4(ndc.x, ndc.y, 0.0, 1.0);
    float4 far = float4(ndc.x, ndc.y, 1.0, 1.0);
    float4 wnear = mul(near, inverse_wvp);
    wnear /= wnear.w;
    float4 wfar = mul(far, inverse_wvp);
    wfar /= wfar.w;
    ro = wnear.xyz;
    rd = normalize(wfar.xyz - wnear.xyz);
    p = float3(0.0, 0.0, 0.0);
    for(float t = 0.0; t < 10.0;)
    {
        p = ro + rd * t;
        float d = sd_box(p, float3(1.0, 1.0, 1.0));
        if(d < 0.001)
        break;
        t += d;
    }
}
fragment ps_output ps_main(vs_output input [[stage_in]]
,  texture_3d( tex_3d, 0 )
, constant c_image_ex &image_ex [[buffer(15)]])
{
    constant float4x4& inverse_wvp = image_ex.inverse_wvp;
    ps_output output;
    float3 p, ro, rd;
    unit_cube_trace(inverse_wvp, input.tex_coord, p, ro, rd);
    float3 uvw = p * 0.5 + 0.5;
    float3 vddx = ddx( uvw );
    float3 vddy = ddy( uvw );
    float max_samples = 64.0;
    float d = sample_texture_grad(tex_3d, uvw, vddx, vddy).r;
    float3 ray_pos = p.xyz;
    float taken = 0.0;
    for( int s = 0; s < int(max_samples); ++s )
    {
        taken += 1.0/max_samples;
        d = sample_texture_grad(tex_3d, uvw, vddx, vddy).r;
        float3 step = rd.xyz * float3(d, d, d) * 0.01;
        uvw += step;
        if(uvw.x >= 1.0 || uvw.x <= 0.0)
        discard;
        if(uvw.y >= 1.0 || uvw.y <= 0.0)
        discard;
        if(uvw.z >= 1.0 || uvw.z <= 0.0)
        discard;
        if( d <= 0.01 )
        break;
    }
    float vd = (1.0 - d);
    output.colour.rgb = float3(vd*vd,vd*vd, vd*vd);
    output.colour.rgb = float3(taken, taken, taken);
    output.colour.a = 1.0;
    return output;
}
