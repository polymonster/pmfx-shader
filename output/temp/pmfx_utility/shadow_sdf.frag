#version 450 core
#define GLSL
#define BINDING_POINTS
//pmfx_utility shadow_sdf ps 0
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
layout(location = 0) out float4 colour_ps_output;
struct vs_output
{
    float4 position;
    float4 world_pos;
    float3 normal;
    float3 tangent;
    float3 bitangent;
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
layout (binding= 3,std140) uniform per_pass_lights
{
    float4 light_info;
    light_data lights[100];
};
layout (binding= 5,std140) uniform per_pass_shadow_distance_fields
{
    distance_field_shadow sdf_shadow;
};
texture_3d( sdf_volume, 14 );
float3 lambert(
float4 light_pos_radius,
float3 light_colour,
float3 n,
float3 world_pos,
float3 albedo
)
{
    float3 l = normalize( light_pos_radius.xyz - world_pos.xyz );
    float n_dot_l = max( dot( n, l ), 0.0 );
    float3 lit_colour = light_colour * n_dot_l * albedo.rgb;
    return lit_colour;
}
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
bool ray_vs_aabb(float3 emin, float3 emax, float3 r1, float3 rv, out float3 intersection)
{
    float3 dirfrac = float3(1.0, 1.0, 1.0) / rv;
    float t1 = (emin.x - r1.x)*dirfrac.x;
    float t2 = (emax.x - r1.x)*dirfrac.x;
    float t3 = (emin.y - r1.y)*dirfrac.y;
    float t4 = (emax.y - r1.y)*dirfrac.y;
    float t5 = (emin.z - r1.z)*dirfrac.z;
    float t6 = (emax.z - r1.z)*dirfrac.z;
    float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
    float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));
    float t = 0.0f;
    if (tmax < 0)
    {
        t = tmax;
        return false;
    }
    if (tmin > tmax)
    {
        t = tmax;
        return false;
    }
    t = tmin;
    intersection = r1 + rv * t;
    return true;
}
float sdf_shadow_trace(float max_samples, float3 light_pos, float3 world_pos, float3 scale, float3 ray_origin, float4x4 inv_mat, float3x3 inv_rot)
{
    float3 ray_dir = normalize(light_pos - world_pos);
    ray_dir = normalize( mul( ray_dir, inv_rot ) );
    float closest = 1.0;
    float3 uvw = ray_origin;
    if(abs(uvw.x) >= 1.0 || abs(uvw.y) >= 1.0 || abs(uvw.z) >= 1.0)
    {
        float3 emin = float3(-1.0, -1.0, -1.0);
        float3 emax = float3(1.0, 1.0, 1.0);
        float3 ip = float3(0.0, 0.0, 0.0);
        bool hit = ray_vs_aabb( emin, emax, uvw, ray_dir, ip);
        uvw = ip;
        if(!hit)
        {
            return closest;
        }
    }
    float3 light_uvw = mul( float4(light_pos, 1.0), inv_mat ).xyz * 0.5 + 0.5;
    uvw = uvw * 0.5 + 0.5;
    float3 v1 = normalize(light_uvw - uvw);
    for( int s = 0; s < int(max_samples); ++s )
    {
        float d = sample_texture_level( sdf_volume, uvw, 0.0 ).r;
        closest = min(d, closest);
        ray_dir = normalize(light_uvw - uvw);
        float3 step = ray_dir.xyz * float3(d, d, d) / scale * 0.7;
        uvw += step;
        if( d <= 0.0 )
        {
            closest = max( d, 0.0 );
            break;
        }
        if(uvw.x >= 1.0 || uvw.x < 0.0)
        break;
        if(uvw.y >= 1.0 || uvw.y < 0.0)
        break;
        if(uvw.z >= 1.0 || uvw.z < 0.0)
        break;
    }
    return closest;
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
    ps_output _output;
    _output.colour = float4(0.0, 0.0, 0.0, 1.0);
    float3 albedo = float3(1.0, 1.0, 1.0);
    float max_samples = 128.0;
    float3x3 inv_rot = to_3x3(sdf_shadow.world_matrix_inv);
    float3 r1 = _input.world_pos.xyz + _input.normal.xyz * 0.3;
    float3 tr1 = mul( float4(r1, 1.0), sdf_shadow.world_matrix_inv ).xyz;
    float3 scale = float3(length(sdf_shadow.world_matrix[0].xyz), length(sdf_shadow.world_matrix[1].xyz), length(sdf_shadow.world_matrix[2].xyz)) * 2.0;
    float3 vddx = ddx( r1 );
    float3 vddy = ddy( r1 );
    float3 v1;
    int point_start = int(light_info.x);
    int point_end = int(light_info.x) + int(light_info.y);
    for( int i = point_start; i < point_end; ++i )
    {
        float3 light_col = float3( 0.0, 0.0, 0.0 );
        light_col += lambert(lights[i].pos_radius,
        lights[i].colour.rgb,
        _input.normal.xyz,
        _input.world_pos.xyz,
        albedo.rgb);
        if(length(light_col) <= 0.0)
        continue;
        float atten = point_light_attenuation(lights[i].pos_radius , _input.world_pos.xyz);
        light_col *= atten;
        float closest = sdf_shadow_trace(max_samples, lights[i].pos_radius.xyz, _input.world_pos.xyz, scale, tr1, sdf_shadow.world_matrix_inv, inv_rot);
        light_col *= smoothstep( 0.0, 0.1, closest);
        _output.colour.rgb += light_col;
    }
    //assign glsl global outputs from structs
    colour_ps_output = _output.colour;
}
