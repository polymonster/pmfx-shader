#version 450 core
#define GLSL
#define BINDING_POINTS
//trace box ps 0
#ifdef GLES
// precision qualifiers
precision highp float;
precision highp sampler2DArray;
#endif
// texture
#ifdef BINDING_POINTS
#define _tex_binding(sampler_index) layout(binding = sampler_index)
#else
#define _tex_binding(sampler_index)
#endif
#define texture_2d( sampler_name, sampler_index ) _tex_binding(sampler_index) uniform sampler2D sampler_name
#define texture_3d( sampler_name, sampler_index ) _tex_binding(sampler_index) uniform sampler3D sampler_name
#define texture_cube( sampler_name, sampler_index ) _tex_binding(sampler_index) uniform samplerCube sampler_name
#define texture_2d_array( sampler_name, sampler_index ) _tex_binding(sampler_index) uniform sampler2DArray sampler_name
#ifdef GLES
#define sample_texture_2dms( sampler_name, x, y, fragment ) texture( sampler_name, vec2(0.0, 0.0) )
#define texture_2dms( type, samples, sampler_name, sampler_index ) uniform sampler2D sampler_name
#else
#define sample_texture_2dms( sampler_name, x, y, fragment ) texelFetch( sampler_name, ivec2( x, y ), fragment )
#define texture_2dms( type, samples, sampler_name, sampler_index ) _tex_binding(sampler_index) uniform sampler2DMS sampler_name
#endif
// sampler
#define sample_texture( sampler_name, V ) texture( sampler_name, V )
#define sample_texture_level( sampler_name, V, l ) textureLod( sampler_name, V, l )
#define sample_texture_grad( sampler_name, V, vddx, vddy ) textureGrad( sampler_name, V, vddx, vddy )
#define sample_texture_array( sampler_name, V, a ) texture( sampler_name, vec3(V, a) )
#define sample_texture_array_level( sampler_name, V, a, l ) textureLod( sampler_name, vec3(V, a), l )
// matrix
#define to_3x3( M4 ) float3x3(M4)
#define from_columns_3x3(A, B, C) (transpose(float3x3(A, B, C)))
#define from_rows_3x3(A, B, C) (float3x3(A, B, C))
#define unpack_vb_instance_mat( mat, r0, r1, r2, r3 ) mat[0] = r0; mat[1] = r1; mat[2] = r2; mat[3] = r3;
#define to_data_matrix(mat) mat
// clip
#define remap_z_clip_space( d ) d // gl clip space is -1 to 1, and this is normalised device coordinate
#define remap_depth( d ) (d = d * 0.5 + 0.5)
#define remap_ndc_ray( r ) float2(r.x, r.y)
#define depth_ps_output gl_FragDepth
// def
#define float4x4 mat4
#define float3x3 mat3
#define float2x2 mat2
#define float4 vec4
#define float3 vec3
#define float2 vec2
#define modf mod
#define frac fract
#define lerp mix
#define mul( A, B ) ((A) * (B))
#define mul_tbn( A, B ) ((B) * (A))
#define saturate( A ) (clamp( A, 0.0, 1.0 ))
#define atan2( A, B ) (atan(A, B))
#define ddx dFdx
#define ddy dFdy
#define _pmfx_unroll
#define chebyshev_normalize( V ) (V.xyz / max( max(abs(V.x), abs(V.y)), abs(V.z) ))
#define max3(v) max(max(v.x, v.y),v.z)
#define max4(v) max(max(max(v.x, v.y),v.z), v.w)
#define PI 3.14159265358979323846264
layout(location = 1) in float4 texcoord_vs_output;
layout(location = 0) out float4 colour_ps_output;
struct vs_output
{
    float4 position;
    float4 texcoord;
};
struct ps_output
{
    float4 colour;
};
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
layout (binding= 1,std140) uniform per_draw_call
{
    float4x4 world_matrix;
    float4 user_data;
    float4 user_data2;
    float4x4 world_matrix_inv_transpose;
};
float sd_box(float3 p, float3 b)
{
    float3 d = abs(p) - b;
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}
float3x3 create_camera( float3 ro, float3 ta, float cr )
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
float3 irrid(float3 n, float3 rd)
{
    float nv = dot(n, -rd);
    float3 col = float3(0.0, 0.0, 0.0);
    col += sin(nv * float3(0.0, 1.0, 0.0) * 10.0 * 1.5) * 0.5 + 0.5;
    col += sin(nv * float3(1.0, 0.0, 0.0) * 20.0 * 1.5) * 0.5 + 0.5;
    col += sin(nv * float3(0.0, 0.0, 1.0) * 5.0 * 1.5) * 0.5 + 0.5;
    return clamp(normalize(col), 0.0, 1.0);
}
void cam_anim(float2 uv, float time, out float3 ro, out float3 rd)
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
float3 crt_c(float3 src, float2 tc)
{
    float2 inv_texel = float2(1.0/640.0, 1.0/480.0);
    float2 ca = float2(inv_texel.x * 2.0, 0.0);
    src.rgb *= saturate(abs(sin(tc.y / inv_texel.y/2.0)) + 0.5);
    return src;
}
float3 sky(float3 v, float time)
{
    float3 grad_a = float3(0.5, 0.5, 0.0);
    float3 grad_b = float3(0.5, 0.0, 1.0);
    grad_a = float3(bcos(time), 0.2, bcos(-time));
    grad_b = float3(bsin(time), bsin(-time), 0.2);
    float grad_t = v.y * 0.5 + 0.5;
    return lerp(grad_b, grad_a, grad_t);
}
float map_box(float3 p)
{
    return sd_box(p, float3(2.5, 2.5, 2.5));
}
float3 calc_normal_box(float3 pos)
{
    float3 eps = float3(0.001, 0.0, 0.0);
    float3 nor;
    nor.x = map_box(pos+eps.xyy) - map_box(pos-eps.xyy);
    nor.y = map_box(pos+eps.yxy) - map_box(pos-eps.yxy);
    nor.z = map_box(pos+eps.yyx) - map_box(pos-eps.yyx);
    return normalize(nor);
}
void main()
{
    //assign vs_output struct from glsl inputs
    vs_output _input;
    _input.texcoord = texcoord_vs_output;
    float2 uv = bend_tc(_input.texcoord.xy);
    float eps = 0.005;
    float iTime = mod(user_data.y * 3, 200);
    float2 iResolution = float2(640.0, 480.0);
    float3 ro;
    float3 rd;
    cam_anim(uv, iTime, ro, rd);
    float d = 10.0;
    float xt = 0.0;
    float3 pp = ro;
    for(float t = 0.0; t < 20.0; ++t)
    {
        pp = ro + rd * xt;
        d = map_box(pp);
        if(d < eps)
        break;
        xt += d;
    }
    float3 n = calc_normal_box(pp);
    float3 col = irrid(n, rd);
    float mask = step(d, eps);
    float inv_mask = 1.0 - mask;
    float3 csky = sky(rd, iTime + 10);
    float3 cc = crt_c(csky * inv_mask + col * mask, uv);
    ps_output _output;
    _output.colour = float4(cc, 1.0);
    //assign glsl global outputs from structs
    colour_ps_output = _output.colour;
}
