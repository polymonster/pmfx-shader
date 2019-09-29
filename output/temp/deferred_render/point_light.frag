#version 450 core
#define GLSL
#define BINDING_POINTS
//deferred_render point_light ps 0
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
layout(location = 1) in float4 screen_coord_vs_output;
layout(location = 2) in float4 light_pos_radius_vs_output;
layout(location = 3) in float4 light_dir_cutoff_vs_output;
layout(location = 4) in float4 light_colour_vs_output;
layout(location = 5) in float4 light_data_vs_output;
layout(location = 0) out float4 colour_ps_output;
struct vs_output
{
    float4 position;
    float4 screen_coord;
    float4 light_pos_radius;
    float4 light_dir_cutoff;
    float4 light_colour;
    float4 light_data;
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
texture_2d( gbuffer_albedo, 0 );
texture_2d( gbuffer_normals, 1 );
texture_2d( gbuffer_world_pos, 2 );
float3 cook_torrence(
float4 light_pos_radius,
float3 light_colour,
float3 n,
float3 world_pos,
float3 view_pos,
float3 albedo,
float3 metalness,
float roughness,
float reflectivity
)
{
    float3 l = normalize( light_pos_radius.xyz - world_pos.xyz );
    float n_dot_l = dot( n, l );
    if( n_dot_l > 0.0f )
    {
        float roughness_sq = roughness * roughness;
        float k = reflectivity;
        float3 v_view = normalize( (view_pos.xyz - world_pos.xyz) );
        float3 hv = normalize( v_view + l );
        float n_dot_v = dot( n, v_view );
        float n_dot_h = dot( n, hv );
        float v_dot_h = dot( v_view, hv );
        float n_dot_h_2 = 2.0f * n_dot_h;
        float g1 = (n_dot_h_2 * n_dot_v) / v_dot_h;
        float g2 = (n_dot_h_2 * n_dot_l) / v_dot_h;
        float geom_atten = min(1.0, min(g1, g2));
        float r1 = 1.0f / ( 4.0f * roughness_sq * pow(n_dot_h, 4.0f));
        float r2 = (n_dot_h * n_dot_h - 1.0) / (roughness_sq * n_dot_h * n_dot_h);
        float roughness_atten = r1 * exp(r2);
        float fresnel = pow(1.0 - v_dot_h, 5.0);
        fresnel *= roughness;
        fresnel += reflectivity;
        float specular = (fresnel * geom_atten * roughness_atten) / (n_dot_v * n_dot_l * 3.1419);
        float3 lit_colour = metalness * light_colour * n_dot_l * ( k + specular * ( 1.0 - k ) );
        return saturate(lit_colour);
    }
    return float3( 0.0, 0.0, 0.0 );
}
float3 oren_nayar(
float4 light_pos_radius,
float3 light_colour,
float3 n,
float3 world_pos,
float3 view_pos,
float roughness,
float3 albedo)
{
    float3 v = normalize(view_pos-world_pos);
    float3 l = normalize(light_pos_radius.xyz-world_pos);
    float l_dot_v = dot(l, v);
    float n_dot_l = dot(n, l);
    float n_dot_v = dot(n, v);
    float s = l_dot_v - n_dot_l * n_dot_v;
    float t = lerp(1.0, max(n_dot_l, n_dot_v), step(0.0, s));
    float lum = length( albedo );
    float sigma2 = roughness * roughness;
    float A = 1.0 + sigma2 * (lum / (sigma2 + 0.13) + 0.5 / (sigma2 + 0.33));
    float B = 0.45 * sigma2 / (sigma2 + 0.09);
    return ( albedo * light_colour * max(0.0, n_dot_l) * (A + B * s / t) / 3.14159265 );
}
float point_light_attenuation_cutoff(
float4 light_pos_radius,
float3 world_pos)
{
    float r = light_pos_radius.w;
    float d = length(world_pos.xyz - light_pos_radius.xyz);
    d = max(d - r, 0.0);
    float denom = d/r + 1.0;
    float attenuation = 1.0 / (denom*denom);
    float cutoff = 0.2;
    attenuation = (attenuation - cutoff) / (1.0 - cutoff);
    attenuation = max(attenuation, 0.0);
    return attenuation;
}
void main()
{
    //assign vs_output struct from glsl inputs
    vs_output _input;
    _input.screen_coord = screen_coord_vs_output;
    _input.light_pos_radius = light_pos_radius_vs_output;
    _input.light_dir_cutoff = light_dir_cutoff_vs_output;
    _input.light_colour = light_colour_vs_output;
    _input.light_data = light_data_vs_output;
    ps_output _output;
    float2 sc = _input.screen_coord.xy;
    float x = _input.screen_coord.x / _input.screen_coord.w;
    float y = _input.screen_coord.y / _input.screen_coord.w;
    sc = float2(x * 0.5 + 0.5, -y * 0.5 + 0.5);
    float3 final_light_col = float3(0.0, 0.0, 0.0);
    int samples = 1;
    _pmfx_unroll
    for(int i = 0; i < samples; ++i)
    {
        float4 g_albedo = sample_texture(gbuffer_albedo, sc);
        float4 g_normals = sample_texture(gbuffer_normals, sc);
        float4 g_world_pos = sample_texture(gbuffer_world_pos, sc);
        float3 albedo = g_albedo.rgb;
        float3 n = normalize(g_normals.rgb);
        float3 world_pos = g_world_pos.rgb;
        float metalness = g_world_pos.a;
        float roughness = g_albedo.a;
        float reflectivity = g_normals.a;
        float3 light_col = cook_torrence(
        _input.light_pos_radius,
        _input.light_colour.rgb,
        n,
        world_pos,
        camera_view_pos.xyz,
        albedo,
        float3(0.5, 0.5, 0.5),
        roughness,
        reflectivity);
        light_col += oren_nayar(
        _input.light_pos_radius,
        _input.light_colour.rgb,
        n,
        world_pos,
        camera_view_pos.xyz,
        roughness,
        albedo.rgb
        );
        light_col *= point_light_attenuation_cutoff(_input.light_pos_radius, world_pos);
        final_light_col += light_col;
    }
    _output.colour.rgb = final_light_col / float(samples);
    //assign glsl global outputs from structs
    colour_ps_output = _output.colour;
}
