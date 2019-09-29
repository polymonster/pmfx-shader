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
struct c_cbuffer_single_light
{
    light_data single_light;
};
struct c_material_data
{
    float4 m_albedo;
    float m_roughness;
    float m_reflectivity;
    float2 m_padding;
};
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
struct ps_output
{
    float4 colour [[color(0)]];
};
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
float3 transform_ts_normal( float3 t,
float3 b,
float3 n,
float3 ts_normal )
{
    float3x3 tbn;
    tbn[0] = float3(t.x, b.x, n.x);
    tbn[1] = float3(t.y, b.y, n.y);
    tbn[2] = float3(t.z, b.z, n.z);
    return normalize( mul_tbn( tbn, ts_normal ) );
}
fragment ps_output ps_main(vs_output input [[stage_in]]
,  texture_2d( diffuse_texture, 0 )
,  texture_2d( normal_texture, 1 )
,  texture_2d( specular_texture, 2 )
, constant c_per_pass_view &per_pass_view [[buffer(4)]]
, constant c_cbuffer_single_light &cbuffer_single_light [[buffer(14)]]
, constant c_material_data &material_data [[buffer(11)]])
{
    constant float4& camera_view_pos = per_pass_view.camera_view_pos;
    constant light_data& single_light = cbuffer_single_light.single_light;
    constant float& m_reflectivity = material_data.m_reflectivity;
    float4 albedo = sample_texture( diffuse_texture, input.texcoord.xy );
    float3 normal_sample = sample_texture( normal_texture, input.texcoord.xy ).rgb;
    float4 ro_sample = sample_texture( specular_texture, input.texcoord.xy );
    float4 specular_sample = float4(1.0, 1.0, 1.0, 1.0);
    float reflectivity = m_reflectivity;
    float roughness = ro_sample.x;
    normal_sample = normal_sample * 2.0 - 1.0;
    float3 n = transform_ts_normal(
    input.tangent,
    input.bitangent,
    input.normal,
    normal_sample );
    float3 lit_colour = float3( 0.0, 0.0, 0.0 );
    lit_colour += cook_torrence(
    single_light.pos_radius,
    single_light.colour.rgb,
    n,
    input.world_pos.xyz,
    camera_view_pos.xyz,
    albedo.rgb,
    specular_sample.rgb,
    roughness,
    reflectivity);
    lit_colour += oren_nayar(
    single_light.pos_radius,
    single_light.colour.rgb,
    n,
    input.world_pos.xyz,
    camera_view_pos.xyz,
    roughness,
    albedo.rgb);
    ps_output output;
    output.colour.rgb = lit_colour;
    output.colour.a = albedo.a;
    return output;
}
