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
#define chebyshev_normalize( V ) (V.xyz / max( max(abs(V.x), abs(V.y)), abs(V.z) ))
#define max3(v) max(max(v.x, v.y),v.z)
#define max4(v) max(max(max(v.x, v.y),v.z), v.w)
#define PI 3.14159265358979323846264
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
struct vs_output
{
    float4 position;
    float4 texcoord;
};
struct ps_output_colour_depth
{
    float4 colour [[color(0)]];
    float depth [[depth(any)]];
};
float point_light_attenuation(
float4 light_pos_radius,
float3 world_pos)
{
    float d = length( world_pos.xyz - light_pos_radius.xyz );
    float r = light_pos_radius.w;
    float denom = d/r + 1.0;
    float attenuation = 1.0 / (denom*denom);
    return attenuation;
}
float op_union( float d1,
float d2 )
{
    return min(d1,d2);
}
float op_subtract( float d1,
float d2 )
{
    return max(-d1,d2);
}
float sd_box(float3 p,
float3 b)
{
    float3 d = abs(p) - b;
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}
float sd_cross(float3 p,
float2 s)
{
    float da = sd_box(p.xyz, float3(s.y, s.x, s.x));
    float db = sd_box(p.yzx, float3(s.x, s.y, s.x));
    float dc = sd_box(p.zxy, float3(s.x, s.x, s.y));
    return op_union(da, op_union(db, dc));
}
float map( float3 p )
{
    float scale = 10.0;
    float rep = 30.0;
    float3 q = mod(p, rep) - 0.5 * rep;
    q = q / scale;
    float d = sd_box(q, float3(1.0, 1.0, 1.0));
    float s = 1.0;
    for( int m=0; m<4; m++ )
    {
        float3 a = mod(q * s, float3(2.0, 2.0, 2.0)) - 1.0;
        s *= 3.0;
        float3 r = 1.0 - 3.0 * abs(a);
        float c = sd_cross(r, float2(1.0, 10000.0) ) / s;
        d = op_subtract(-c, d);
    }
    return d * scale;
}
float3 calc_normal(float3 pos)
{
    float3 eps = float3(0.001, 0.0, 0.0);
    float3 nor;
    nor.x = map(pos+eps.xyy) - map(pos-eps.xyy);
    nor.y = map(pos+eps.yxy) - map(pos-eps.yxy);
    nor.z = map(pos+eps.yyx) - map(pos-eps.yyx);
    return normalize(nor);
}
float intersect(float3 ro,
float3 rd,
thread float3& pos)
{
    for(float t = 0.0; t < 150.0;)
    {
        float3 p = ro + rd * t;
        float d = map(p);
        if(d < 0.001)
        {
            pos = p;
            return t;
        }
        t += d;
    }
    return -1.0;
}
fragment ps_output_colour_depth ps_main(vs_output input [[stage_in]]
, constant c_per_pass_view &per_pass_view [[buffer(4)]])
{
    constant float4x4& vp_matrix = per_pass_view.vp_matrix;
    constant float4x4& vp_matrix_inverse = per_pass_view.vp_matrix_inverse;
    constant float4& camera_view_pos = per_pass_view.camera_view_pos;
    ps_output_colour_depth output;
    float2 ndc = input.texcoord.xy * float2(2.0, 2.0) - float2(1.0, 1.0);
    ndc = remap_ndc_ray(ndc);
    float4 near = float4(ndc.x, ndc.y, 0.0, 1.0);
    float4 far = float4(ndc.x, ndc.y, 1.0, 1.0);
    float4 wnear = mul(near, vp_matrix_inverse);
    wnear /= wnear.w;
    float4 wfar = mul(far, vp_matrix_inverse);
    wfar /= wfar.w;
    float4 col = float4(0.0, 0.0, 0.0, 1.0);
    float3 ray_origin = wnear.xyz;
    float3 ray_dir = normalize(wfar.xyz - wnear.xyz);
    float3 world_pos;
    float d = intersect(ray_origin, ray_dir, world_pos);
    float3 grad_a = float3(0.9, 0.5, 0.0);
    float3 grad_b = float3(0.5, 0.0, 1.0);
    float grad_t = ray_dir.y * 0.5 + 0.5;
    float4 sky = float4(lerp(grad_a, grad_b, grad_t), 1.0);
    output.depth = 1.0;
    float4 sd_col = sky;
    float sky_t = 0.0;
    if(d > 0.0)
    {
        float3 n = calc_normal(ray_origin + ray_dir * d);
        float4 lpr = float4(camera_view_pos.xyz, 100.0);
        float3 l = normalize(lpr.xyz - world_pos);
        float ndotl = dot(n, l);
        float a = point_light_attenuation(lpr, world_pos.xyz);
        float3 lc = a * ndotl * float3(0.0, 0.7, 0.9);
        sd_col = float4(lc, 1.0);
        float4 proj = mul(float4(world_pos, 1.0), vp_matrix);
        proj /= proj.w;
        output.depth = proj.z;
        sky_t = smoothstep(150.0, 80.0, length(world_pos - ray_origin));
    }
    output.colour = lerp(sky, sd_col, sky_t);
    return output;
}
