#version 450 core
#define GLSL
#define BINDING_POINTS
//forward_render gbuffer__2147483650__ ps 2147483650
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
layout(location = 1) in float4 world_pos_vs_output;
layout(location = 2) in float3 normal_vs_output;
layout(location = 3) in float3 tangent_vs_output;
layout(location = 4) in float3 bitangent_vs_output;
layout(location = 5) in float4 texcoord_vs_output;
layout(location = 6) in float4 colour_vs_output;
layout(location = 0) out float4 albedo_ps_output;
layout(location = 1) out float4 normal_ps_output;
layout(location = 2) out float4 world_pos_ps_output;
struct vs_output
{
    float4 position;
    float4 world_pos;
    float3 normal;
    float3 tangent;
    float3 bitangent;
    float4 texcoord;
    float4 colour;
};
struct ps_output_multi
{
    float4 albedo;
    float4 normal;
    float4 world_pos;
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
layout (binding= 7,std140) uniform material_data
{
    float4 m_albedo;
    float2 m_uv_scale;
    float m_roughness;
    float m_reflectivity;
};
texture_2d( diffuse_texture, 0 );
texture_2d( normal_texture, 1 );
texture_2d( specular_texture, 2 );
float3 transform_ts_normal( float3 t, float3 b, float3 n, float3 ts_normal )
{
    float3x3 tbn;
    tbn[0] = float3(t.x, b.x, n.x);
    tbn[1] = float3(t.y, b.y, n.y);
    tbn[2] = float3(t.z, b.z, n.z);
    return normalize( mul_tbn( tbn, ts_normal ) );
}
void main()
{
    //assign vs_output struct from glsl inputs
    vs_output _input;
    _input.world_pos = world_pos_vs_output;
    _input.normal = normal_vs_output;
    _input.tangent = tangent_vs_output;
    _input.bitangent = bitangent_vs_output;
    _input.texcoord = texcoord_vs_output;
    _input.colour = colour_vs_output;
    ps_output_multi _output;
    float4 albedo = sample_texture(diffuse_texture, _input.texcoord.xy);
    float4 metalness = sample_texture(specular_texture, _input.texcoord.xy);
    float3 normal_sample = sample_texture( normal_texture, _input.texcoord.xy ).rgb;
    normal_sample = normal_sample * 2.0 - 1.0;
    float4 ro_sample = sample_texture( specular_texture, _input.texcoord.xy );
    float3 n = transform_ts_normal(
    _input.tangent,
    _input.bitangent,
    _input.normal,
    normal_sample );
    float roughness = ro_sample.x;
    float reflectivity = m_reflectivity;
    _output.albedo = float4(albedo.rgb * _input.colour.rgb, roughness);
    _output.normal = float4(n, reflectivity);
    _output.world_pos = float4(_input.world_pos.xyz, metalness.r);
    //assign glsl global outputs from structs
    albedo_ps_output = _output.albedo;
    normal_ps_output = _output.normal;
    world_pos_ps_output = _output.world_pos;
}
