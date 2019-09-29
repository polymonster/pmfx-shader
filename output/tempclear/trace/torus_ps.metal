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
struct c_per_draw_call
{
    float4x4 world_matrix;
    float4 user_data;
    float4 user_data2;
    float4x4 world_matrix_inv_transpose;
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
float sd_torus(float3 p,
float2 t)
{
    float2 q = float2(length(p.xy) - t.x,p.z);
    return length(q)-t.y;
}
float3x3 create_camera( float3 ro,
float3 ta,
float cr )
{
    float3 cw = normalize(ta-ro);
    float3 cp = float3(sin(cr), cos(cr),0.0);
    float3 cu = normalize( cross(cw,cp) );
    float3 cv = cross(cu,cw);
    return from_columns_3x3( cu, cv, cw );
}
float bsin(float v)
{
    return sin(v) * 0.5 + 1.0;
}
float bcos(float v)
{
    return cos(v) * 0.5 + 1.0;
}
float3 irrid(float3 n,
float3 rd)
{
    float nv = dot(n, -rd);
    float3 col = float3(0.0, 0.0, 0.0);
    col += sin(nv * float3(0.0, 1.0, 0.0) * 10.0 * 1.5) * 0.5 + 0.5;
    col += sin(nv * float3(1.0, 0.0, 0.0) * 20.0 * 1.5) * 0.5 + 0.5;
    col += sin(nv * float3(0.0, 0.0, 1.0) * 5.0 * 1.5) * 0.5 + 0.5;
    return clamp(normalize(col), 0.0, 1.0);
}
void cam_anim(float2 uv,
float time,
thread float3& ro,
thread float3& rd)
{
    ro = float3(cos(time) * 10, 0.0, sin(time) * 10);
    float3 ta = float3( -0.5, -0.4, 0.5 );
    float3x3 cam = create_camera( ro, ta, time );
    float2 p = (uv * 2.0) - 1.0;
    rd = mul( normalize( float3(p.x, p.y, 2.0) ), cam);
}
float2 bend_tc(float2 uv)
{
    float2 tc = uv;
    float2 cc = tc - 0.5;
    float dist = dot(cc, cc) * 0.07;
    tc = tc * (tc + cc * (1.0 + dist) * dist) / tc;
    return tc;
}
float3 crt_c(float3 src,
float2 tc)
{
    float2 inv_texel = float2(1.0/640.0, 1.0/480.0);
    float2 ca = float2(inv_texel.x * 2.0, 0.0);
    src.rgb *= saturate(abs(sin(tc.y / inv_texel.y/2.0)) + 0.5);
    return src;
}
float3 sky(float3 v,
float time)
{
    float3 grad_a = float3(0.5, 0.5, 0.0);
    float3 grad_b = float3(0.5, 0.0, 1.0);
    grad_a = float3(bcos(time), 0.2, bcos(-time));
    grad_b = float3(bsin(time), bsin(-time), 0.2);
    float grad_t = v.y * 0.5 + 0.5;
    return lerp(grad_b, grad_a, grad_t);
}
float map_torus(float3 p)
{
    return sd_torus(p, float2(2.5, 1.0));
}
float3 calc_normal_torus(float3 pos)
{
    float3 eps = float3(0.001, 0.0, 0.0);
    float3 nor;
    nor.x = map_torus(pos+eps.xyy) - map_torus(pos-eps.xyy);
    nor.y = map_torus(pos+eps.yxy) - map_torus(pos-eps.yxy);
    nor.z = map_torus(pos+eps.yyx) - map_torus(pos-eps.yyx);
    return normalize(nor);
}
fragment ps_output ps_main(vs_output input [[stage_in]]
, constant c_per_draw_call &per_draw_call [[buffer(5)]])
{
    constant float4& user_data = per_draw_call.user_data;
    float2 uv = bend_tc(input.texcoord.xy);
    float eps = 0.005;
    float iTime = mod(user_data.y * 3, 200);
    float2 iResolution = float2(640.0, 480.0);
    float3 ro;
    float3 rd;
    cam_anim(uv, iTime, ro, rd);
    float d = 10.0;
    float xt = 0.0;
    float3 pp = ro;
    for(float t = 0.0; t < 30.0; ++t)
    {
        pp = ro + rd * xt;
        d = map_torus(pp);
        if(d < eps)
        break;
        xt += d;
    }
    float3 n = calc_normal_torus(pp);
    float3 col = irrid(n, rd);
    float mask = step(d, eps);
    float inv_mask = 1.0 - mask;
    float3 csky = sky(rd, iTime + 10);
    csky = csky.zxy;
    float3 cc = crt_c(csky * inv_mask + col * mask, uv);
    ps_output output;
    output.colour = float4(cc, 1.0);
    return output;
}
