#version 450 core
#define GLSL
#define BINDING_POINTS
//post_process sss_reflectance ps 0
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
layout (binding= 10,std140) uniform src_info
{
    float2 inv_texel_size[8];
};
layout (binding= 2,std140) uniform filter_kernel
{
    float4 filter_info;
    float4 filter_offset_weight[16];
};
texture_2d( src_texture_0, 0 );
texture_2d( src_texture_1, 1 );
void main()
{
    //assign vs_output struct from glsl inputs
    vs_output _input;
    _input.texcoord = texcoord_vs_output;
    float4 sss_kernel[25];
    sss_kernel[0] = float4(0.530605, 0.613514, 0.739601, 0);
    sss_kernel[1] = float4(0.000973794, 1.11862e-005, 9.43437e-007, -3);
    sss_kernel[2] = float4(0.00333804, 7.85443e-005, 1.2945e-005, -2.52083);
    sss_kernel[3] = float4(0.00500364, 0.00020094, 5.28848e-005, -2.08333);
    sss_kernel[4] = float4(0.00700976, 0.00049366, 0.000151938, -1.6875);
    sss_kernel[5] = float4(0.0094389, 0.00139119, 0.000416598, -1.33333);
    sss_kernel[6] = float4(0.0128496, 0.00356329, 0.00132016, -1.02083);
    sss_kernel[7] = float4(0.017924, 0.00711691, 0.00347194, -0.75);
    sss_kernel[8] = float4(0.0263642, 0.0119715, 0.00684598, -0.520833);
    sss_kernel[9] = float4(0.0410172, 0.0199899, 0.0118481, -0.333333);
    sss_kernel[10] = float4(0.0493588, 0.0367726, 0.0219485, -0.1875);
    sss_kernel[11] = float4(0.0402784, 0.0657244, 0.04631, -0.0833333);
    sss_kernel[12] = float4(0.0211412, 0.0459286, 0.0378196, -0.0208333);
    sss_kernel[13] = float4(0.0211412, 0.0459286, 0.0378196, 0.0208333);
    sss_kernel[14] = float4(0.0402784, 0.0657244, 0.04631, 0.0833333);
    sss_kernel[15] = float4(0.0493588, 0.0367726, 0.0219485, 0.1875);
    sss_kernel[16] = float4(0.0410172, 0.0199899, 0.0118481, 0.333333);
    sss_kernel[17] = float4(0.0263642, 0.0119715, 0.00684598, 0.520833);
    sss_kernel[18] = float4(0.017924, 0.00711691, 0.00347194, 0.75);
    sss_kernel[19] = float4(0.0128496, 0.00356329, 0.00132016, 1.02083);
    sss_kernel[20] = float4(0.0094389, 0.00139119, 0.000416598, 1.33333);
    sss_kernel[21] = float4(0.00700976, 0.00049366, 0.000151938, 1.6875);
    sss_kernel[22] = float4(0.00500364, 0.00020094, 5.28848e-005, 2.08333);
    sss_kernel[23] = float4(0.00333804, 7.85443e-005, 1.2945e-005, 2.52083);
    sss_kernel[24] = float4(0.000973794, 1.11862e-005, 9.43437e-007, 3);
    ps_output _output;
    float2 tc = _input.texcoord.xy;
    float z = sample_texture(src_texture_1, tc).r;
    float n = camera_view_pos.w;
    float f = camera_view_dir.w;
    float ez = (2 * n * f) / (f + n - z * (f - n));
    float lz = (ez - n) / (f - n);
    float dist = 1.0 / tan(0.5 * 60.0 * 3.14 / 180.0 );
    float scale = dist / lz;
    float w = 1.0;
    float2 final_step = w * scale * filter_info.xy;
    final_step *= 1.0 / 3.0;
    float3 col = sample_texture(src_texture_0, tc).rgb * sss_kernel[0].rgb;
    float2 it = inv_texel_size[0];
    for(int i = 1; i < 25; ++i)
    {
        float2 offset = sss_kernel[i].a * it.xy * final_step;
        col += sample_texture(src_texture_0, (tc + offset)).rgb * sss_kernel[i].rgb;
    }
    _output.colour.rgb = col;
    //assign glsl global outputs from structs
    colour_ps_output = _output.colour;
}
