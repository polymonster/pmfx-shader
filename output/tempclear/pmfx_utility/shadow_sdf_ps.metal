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
struct c_per_pass_lights
{
    float4 light_info;
    light_data lights[100];
};
struct c_per_pass_shadow_distance_fields
{
    distance_field_shadow sdf_shadow;
};
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
    float4 colour [[color(0)]];
};
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
bool ray_vs_aabb(float3 emin,
float3 emax,
float3 r1,
float3 rv,
thread float3& intersection)
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
float sdf_shadow_trace(
texture_3d_arg(sdf_volume),
float max_samples,
float3 light_pos,
float3 world_pos,
float3 scale,
float3 ray_origin,
float4x4 inv_mat,
float3x3 inv_rot)
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
fragment ps_output ps_main(vs_output input [[stage_in]]
,  texture_3d( sdf_volume, 14 )
, constant c_per_draw_call &per_draw_call [[buffer(5)]]
, constant c_per_pass_lights &per_pass_lights [[buffer(7)]]
, constant c_per_pass_shadow_distance_fields &per_pass_shadow_distance_fields [[buffer(9)]])
{
    constant float4x4& world_matrix = per_draw_call.world_matrix;
    constant float4& light_info = per_pass_lights.light_info;
    constant light_data* lights = &per_pass_lights.lights[0];
    constant distance_field_shadow& sdf_shadow = per_pass_shadow_distance_fields.sdf_shadow;
    ps_output output;
    output.colour = float4(0.0, 0.0, 0.0, 1.0);
    float3 albedo = float3(1.0, 1.0, 1.0);
    float max_samples = 128.0;
    float3x3 inv_rot = to_3x3(sdf_shadow.world_matrix_inv);
    float3 r1 = input.world_pos.xyz + input.normal.xyz * 0.3;
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
        input.normal.xyz,
        input.world_pos.xyz,
        albedo.rgb);
        if(length(light_col) <= 0.0)
        continue;
        float atten = point_light_attenuation(lights[i].pos_radius , input.world_pos.xyz);
        light_col *= atten;
        float closest = sdf_shadow_trace(sdf_volume, sampler_sdf_volume, max_samples, lights[i].pos_radius.xyz, input.world_pos.xyz, scale, tr1, sdf_shadow.world_matrix_inv, inv_rot);
        light_col *= smoothstep( 0.0, 0.1, closest);
        output.colour.rgb += light_col;
    }
    return output;
}
