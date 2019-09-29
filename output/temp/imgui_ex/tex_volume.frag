#version 450 core
#define GLSL
#define BINDING_POINTS
//imgui_ex tex_volume ps 0
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
layout(location = 1) in float4 colour_vs_output;
layout(location = 2) in float2 tex_coord_vs_output;
layout(location = 0) out float4 colour_ps_output;
struct vs_output
{
    float4 position;
    float4 colour;
    float2 tex_coord;
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
layout (binding= 7,std140) uniform image_ex
{
    float4 colour_mask;
    float4 params;
    float4x4 inverse_wvp;
};
texture_3d( tex_3d, 0 );
float sd_box(float3 p, float3 b)
{
    float3 d = abs(p) - b;
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}
void unit_cube_trace(float2 tc, out float3 p, out float3 ro, out float3 rd)
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
void main()
{
    //assign vs_output struct from glsl inputs
    vs_output _input;
    _input.colour = colour_vs_output;
    _input.tex_coord = tex_coord_vs_output;
    ps_output _output;
    float3 p, ro, rd;
    unit_cube_trace(_input.tex_coord, p, ro, rd);
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
    _output.colour.rgb = float3(vd*vd,vd*vd, vd*vd);
    _output.colour.rgb = float3(taken, taken, taken);
    _output.colour.a = 1.0;
    //assign glsl global outputs from structs
    colour_ps_output = _output.colour;
}
