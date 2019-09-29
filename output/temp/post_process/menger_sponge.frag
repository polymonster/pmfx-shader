#version 450 core
#define GLSL
#define BINDING_POINTS
//post_process menger_sponge ps 0
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
struct ps_output_colour_depth
{
    float4 colour;
    float depth;
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
layout (binding= 0,std140) uniform per_pass_view
{
    float4x4 vp_matrix;
    float4x4 view_matrix;
    float4x4 vp_matrix_inverse;
    float4x4 view_matrix_inverse;
    float4 camera_view_pos;
    float4 camera_view_dir;
    float4 viewport_correction;
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
float op_union( float d1, float d2 )
{
    return min(d1,d2);
}
float op_subtract( float d1, float d2 )
{
    return max(-d1,d2);
}
float sd_box(float3 p, float3 b)
{
    float3 d = abs(p) - b;
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}
float sd_cross(float3 p, float2 s)
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
float intersect(float3 ro, float3 rd, out float3 pos)
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
void main()
{
    //assign vs_output struct from glsl inputs
    vs_output _input;
    _input.texcoord = texcoord_vs_output;
    ps_output_colour_depth _output;
    float2 ndc = _input.texcoord.xy * float2(2.0, 2.0) - float2(1.0, 1.0);
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
    _output.depth = 1.0;
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
        _output.depth = proj.z;
        sky_t = smoothstep(150.0, 80.0, length(world_pos - ray_origin));
    }
    _output.colour = lerp(sky, sd_col, sky_t);
    //assign glsl global outputs from structs
    colour_ps_output = _output.colour;
    depth_ps_output = _output.depth;
}
