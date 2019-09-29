#version 450 core
#define GLSL
#define BINDING_POINTS
//forward_render forward_lit__1073741836__ ps 1073741836
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
layout(location = 0) out float4 colour_ps_output;
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
layout (binding= 4,std140) uniform per_pass_shadow
{
    float4x4 shadow_matrix[100];
};
layout (binding= 5,std140) uniform per_pass_shadow_distance_fields
{
    distance_field_shadow sdf_shadow;
};
layout (binding= 6,std140) uniform per_pass_area_lights
{
    float4 area_light_info;
    area_light_data area_lights[10];
};
layout (binding= 7,std140) uniform material_data
{
    float4 m_albedo;
    float m_roughness;
    float m_reflectivity;
    float m_sss_scale;
    float m_surface_offset;
};
texture_2d( diffuse_texture, 0 );
texture_2d( normal_texture, 1 );
texture_2d( specular_texture, 2 );
texture_3d( sdf_volume, 14 );
texture_2d( ltc_mat, 13 );
texture_2d( ltc_mag, 12 );
texture_2d_array( shadowmap_texture, 15 );
texture_2d_array( area_light_textures, 11 );
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
float spot_light_attenuation(
float4 light_pos_radius,
float4 light_dir_cutoff,
float falloff,
float3 world_pos)
{
    float co = light_dir_cutoff.w;
    float3 vl = normalize(world_pos.xyz - light_pos_radius.xyz);
    float3 sd = normalize(light_dir_cutoff.xyz);
    float dp = (1.0 - dot(vl, sd));
    return smoothstep(co, co - falloff, dp);
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
float integrate_edge(float3 v1, float3 v2)
{
    float cos_theta = dot(v1, v2);
    float theta = acos(cos_theta);
    float res = cross(v1, v2).z * ((theta > 0.001) ? theta/sin(theta) : 1.0);
    return res;
}
void clip_quad_to_horizon(inout float3 L[5], out int n)
{
    int config = 0;
    if (L[0].z > 0.0) config += 1;
    if (L[1].z > 0.0) config += 2;
    if (L[2].z > 0.0) config += 4;
    if (L[3].z > 0.0) config += 8;
    n = 0;
    if (config == 0)
    {
    }
    else if (config == 1)
    {
        n = 3;
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
        L[2] = -L[3].z * L[0] + L[0].z * L[3];
    }
    else if (config == 2)
    {
        n = 3;
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
    }
    else if (config == 3)
    {
        n = 4;
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
        L[3] = -L[3].z * L[0] + L[0].z * L[3];
    }
    else if (config == 4)
    {
        n = 3;
        L[0] = -L[3].z * L[2] + L[2].z * L[3];
        L[1] = -L[1].z * L[2] + L[2].z * L[1];
    }
    else if (config == 5)
    {
        n = 0;
    }
    else if (config == 6)
    {
        n = 4;
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
        L[3] = -L[3].z * L[2] + L[2].z * L[3];
    }
    else if (config == 7)
    {
        n = 5;
        L[4] = -L[3].z * L[0] + L[0].z * L[3];
        L[3] = -L[3].z * L[2] + L[2].z * L[3];
    }
    else if (config == 8)
    {
        n = 3;
        L[0] = -L[0].z * L[3] + L[3].z * L[0];
        L[1] = -L[2].z * L[3] + L[3].z * L[2];
        L[2] = L[3];
    }
    else if (config == 9)
    {
        n = 4;
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
        L[2] = -L[2].z * L[3] + L[3].z * L[2];
    }
    else if (config == 10)
    {
        n = 0;
    }
    else if (config == 11)
    {
        n = 5;
        L[4] = L[3];
        L[3] = -L[2].z * L[3] + L[3].z * L[2];
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
    }
    else if (config == 12)
    {
        n = 4;
        L[1] = -L[1].z * L[2] + L[2].z * L[1];
        L[0] = -L[0].z * L[3] + L[3].z * L[0];
    }
    else if (config == 13)
    {
        n = 5;
        L[4] = L[3];
        L[3] = L[2];
        L[2] = -L[1].z * L[2] + L[2].z * L[1];
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
    }
    else if (config == 14)
    {
        n = 5;
        L[4] = -L[0].z * L[3] + L[3].z * L[0];
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
    }
    else if (config == 15)
    {
        n = 4;
    }
    if (n == 3)
    L[3] = L[0];
    if (n == 4)
    L[4] = L[0];
}
float3 ltc_uv_coord(float3 p[4])
{
    float3 v1 = p[1] - p[0];
    float3 v2 = p[3] - p[0];
    float3 plane_ortho = (cross(v1, v2));
    float plane_area_squared = dot(plane_ortho, plane_ortho);
    float plane_distx_plane_area = dot(plane_ortho, p[0]);
    float3 pp = plane_distx_plane_area * plane_ortho / plane_area_squared - p[0];
    float v1_dot_v2 = dot(v1, v2);
    float inv_v1_dot_v1 = 1.0 / dot(v1, v1);
    float3 vv2 = v2 - v1 * v1_dot_v2 * inv_v1_dot_v1;
    float2 puv;
    puv.y = dot(vv2, pp) / dot(vv2, vv2);
    puv.x = dot(v1, pp) * inv_v1_dot_v1 - v1_dot_v2 * inv_v1_dot_v1 * puv.y;
    float d = abs(plane_distx_plane_area) / pow(plane_area_squared, 0.75);
    return float3(puv, d);
}
float4 ltc_evaluate(
float3 n,
float3 v,
float3 p,
float3x3 minv,
float3 points[4],
bool two_sided)
{
    float3 t1, t2;
    t1 = normalize(v - n * dot(v, n));
    t2 = cross(n, t1);
    float3x3 ttn = from_columns_3x3(t1, t2, n);
    minv = mul(minv, ttn);
    float3 l[5];
    l[0] = mul(minv, points[0] - p);
    l[1] = mul(minv, points[1] - p);
    l[2] = mul(minv, points[2] - p);
    l[3] = mul(minv, points[3] - p);
    l[4] = l[3];
    float3 ll[4];
    ll[0] = l[0];
    ll[1] = l[1];
    ll[2] = l[2];
    ll[3] = l[3];
    float3 uvl = ltc_uv_coord(ll);
    int nc;
    clip_quad_to_horizon(l, nc);
    if (nc == 0)
    return float4(0, 0, 0, 0.0);
    l[0] = normalize(l[0]);
    l[1] = normalize(l[1]);
    l[2] = normalize(l[2]);
    l[3] = normalize(l[3]);
    l[4] = normalize(l[4]);
    float sum = 0.0;
    sum += integrate_edge(l[0], l[1]);
    sum += integrate_edge(l[1], l[2]);
    sum += integrate_edge(l[2], l[3]);
    if (nc >= 4)
    sum += integrate_edge(l[3], l[4]);
    if (nc == 5)
    sum += integrate_edge(l[4], l[0]);
    sum = two_sided ? abs(sum) : max(0.0, sum);
    float3 lo_i = float3(sum, sum, sum);
    return float4(uvl.x, uvl.y, uvl.z, sum);
}
float ltc_evaluate_cc(
float3 n,
float3 v,
float3 p,
float3x3 minv,
float3 points[4],
bool two_sided)
{
    float3 t1, t2;
    t1 = normalize(v - n * dot(v, n));
    t2 = cross(n, t1);
    float3x3 ttn = from_columns_3x3(t1, t2, n);
    minv = mul(minv, ttn);
    float3 l[5];
    for(int i = 0; i < 4; ++i)
    l[i] = mul(minv, points[i] - p);
    l[4] = l[3];
    int nc;
    clip_quad_to_horizon(l, nc);
    if (nc == 0)
    return 0.0;
    for(int i = 0; i < 5; ++i)
    l[i] = normalize(l[i]);
    float sum = 0.0;
    sum += integrate_edge(l[0], l[1]);
    sum += integrate_edge(l[1], l[2]);
    sum += integrate_edge(l[2], l[3]);
    if (nc >= 4)
    sum += integrate_edge(l[3], l[4]);
    if (nc == 5)
    sum += integrate_edge(l[4], l[0]);
    sum = two_sided ? abs(sum) : max(0.0, sum);
    return sum;
}
float4 area_light_specular_uv(
float3 points[4],
float3 pos,
float roughness,
float3 n,
float3 v)
{
    float pi = 3.14159265359;
    float lut_size = 64.0;
    float lut_scale = (lut_size - 1.0)/lut_size;
    float lut_bias = 0.5/lut_size;
    float theta = acos(dot(n, v));
    float2 uv = float2(roughness, theta / (0.5 * pi));
    uv = uv * lut_scale + lut_bias;
    float4 mat = sample_texture(ltc_mat, uv);
    float mag = sample_texture(ltc_mag, uv).w;
    float3x3 minv = from_rows_3x3(
    float3(1.0, 0.0, mat.y),
    float3(0.0, mat.z, 0.0),
    float3(mat.w, 0.0, mat.x)
    );
    float4 spec = ltc_evaluate(n, v, pos, minv, points, true);
    return spec;
}
float area_light_specular(
float3 points[4],
float3 pos,
float roughness,
float3 n,
float3 v)
{
    float pi = 3.14159265359;
    float lut_size = 64.0;
    float lut_scale = (lut_size - 1.0)/lut_size;
    float lut_bias = 0.5/lut_size;
    float theta = acos(dot(n, v));
    float2 uv = float2(roughness, theta / (0.5 * pi));
    uv = uv * lut_scale + lut_bias;
    float4 mat = sample_texture(ltc_mat, uv);
    float mag = sample_texture(ltc_mag, uv).w;
    float3x3 minv = from_rows_3x3(
    float3(1.0, 0.0, mat.y),
    float3(0.0, mat.z, 0.0),
    float3(mat.w, 0.0, mat.x)
    );
    float spec = ltc_evaluate_cc(n, v, pos, minv, points, true);
    return spec;
}
float4 area_light_diffuse_uv(
float3 points[4],
float3 pos,
float3 n,
float3 v)
{
    float3x3 difv = float3x3(
    float3(1.0, 0.0, 0.0),
    float3(0.0, 1.0, 0.0),
    float3(0.0, 0.0, 1.0)
    );
    float4 diff = ltc_evaluate(n, v, pos, difv, points, true);
    return diff;
}
float area_light_diffuse(
float3 points[4],
float3 pos,
float3 n,
float3 v)
{
    float3x3 difv = float3x3(
    float3(1.0, 0.0, 0.0),
    float3(0.0, 1.0, 0.0),
    float3(0.0, 0.0, 1.0)
    );
    float diff = ltc_evaluate_cc(n, v, pos, difv, points, true);
    return diff;
}
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
    ps_output _output;
    float4 albedo = sample_texture( diffuse_texture, _input.texcoord.xy );
    float3 normal_sample = sample_texture( normal_texture, _input.texcoord.xy ).rgb;
    float4 ro_sample = sample_texture( specular_texture, _input.texcoord.xy );
    float4 specular_sample = float4(1.0, 1.0, 1.0, 1.0);
    normal_sample = normal_sample * 2.0 - 1.0;
    float3 n = transform_ts_normal(
    _input.tangent,
    _input.bitangent,
    _input.normal,
    normal_sample );
    albedo *= _input.colour;
    float3 lit_colour = float3( 0.0, 0.0, 0.0 );
    float reflectivity = saturate(user_data.z);
    float roughness = saturate(user_data.y);
    reflectivity = m_reflectivity;
    roughness = ro_sample.x;
    roughness = _input.colour.a;
    n = _input.normal.rgb;
    roughness = m_roughness;
    float max_samples = 128.0;
    float3x3 inv_rot = to_3x3(sdf_shadow.world_matrix_inv);
    float3 r1 = _input.world_pos.xyz + _input.normal.xyz * m_surface_offset;
    float3 tr1 = mul( float4(r1, 1.0), sdf_shadow.world_matrix_inv ).xyz;
    float3 scale = float3(length(sdf_shadow.world_matrix[0].xyz), length(sdf_shadow.world_matrix[1].xyz), length(sdf_shadow.world_matrix[2].xyz)) * 2.0;
    float3 vddx = ddx( r1 );
    float3 vddy = ddy( r1 );
    float t = 1.0;
    float3 lll = float3(0.0, 0.0, 0.0);
    for( int i = 0; i < int(light_info.x); ++i )
    {
        float3 light_col = float3( 0.0, 0.0, 0.0 );
        light_col += cook_torrence(
        lights[i].pos_radius,
        lights[i].colour.rgb,
        n,
        _input.world_pos.xyz,
        camera_view_pos.xyz,
        albedo.rgb,
        specular_sample.rgb,
        roughness,
        reflectivity
        );
        light_col += oren_nayar(
        lights[i].pos_radius,
        lights[i].colour.rgb,
        n,
        _input.world_pos.xyz,
        camera_view_pos.xyz,
        roughness,
        albedo.rgb
        );
        float s = sdf_shadow_trace(max_samples, lights[i].pos_radius.xyz, _input.world_pos.xyz, scale, tr1, sdf_shadow.world_matrix_inv, inv_rot);
        light_col *= smoothstep( 0.0, 0.1, s);
        if( lights[i].colour.a == 0.0 )
        {
            lit_colour += light_col;
            continue;
        }
        else
        {
            float shadow = 1.0;
            float d = 1.0;
            float4 offset_pos = float4(_input.world_pos.xyz + n.xyz * 0.01, 1.0);
            float4 sp = mul( offset_pos, shadow_matrix[i] );
            sp.xyz /= sp.w;
            sp.y *= -1.0;
            sp.xyz = sp.xyz * 0.5 + 0.5;
            float4 sm = sample_texture_array_level( shadowmap_texture, sp.xy, float(i), 0 );
            d = sm.r;
            shadow = sp.z < d ? 1.0 : 0.0;
            lit_colour += light_col * shadow;
            float sss_offset = 0.1;
            float4 shrink_pos = float4(_input.world_pos.xyz - _input.normal.xyz * sss_offset, 1.0);
            sp = mul( shrink_pos, shadow_matrix[i] );
            sp.xyz /= sp.w;
            sp.y *= -1.0;
            sp.xyz = sp.xyz * 0.5 + 0.5;
            sm = sample_texture_array_level( shadowmap_texture, sp.xy, float(i), 0 );
            d = sm.r;
            float fp = 1.0;
            float3 light_dir = _input.world_pos.xyz - lights[i].pos_radius.xyz;
            float d1 = d;
            float d2 = sp.z;
            d1 *= fp;
            float sssd = m_sss_scale * abs(d1 - d2) * 0.5 + 0.5;
            float dd = -(sssd * sssd);
            float3 profile = float3(0.233, 0.455, 0.649) * exp(dd / 0.0064) +
            float3(0.1, 0.336, 0.344) * exp(dd / 0.0484) +
            float3(0.118, 0.198, 0.0) * exp(dd / 0.187) +
            float3(0.113, 0.007, 0.007) * exp(dd / 0.567) +
            float3(0.358, 0.004, 0.0) * exp(dd / 1.99) +
            float3(0.078, 0.0, 0.0) * exp(dd / 7.41);
            float sss = saturate(0.5 + dot(light_dir, n));
            profile *= albedo.rgb;
            lit_colour.rgb += profile * sss;
            lit_colour += albedo.rgb * 0.15;
        }
    }
    int point_start = int(light_info.x);
    int point_end = int(light_info.x) + int(light_info.y);
    for( int i = point_start; i < point_end; ++i )
    {
        float3 light_col = float3( 0.0, 0.0, 0.0 );
        light_col += cook_torrence(
        lights[i].pos_radius,
        lights[i].colour.rgb,
        n,
        _input.world_pos.xyz,
        camera_view_pos.xyz,
        albedo.rgb,
        specular_sample.rgb,
        roughness,
        reflectivity
        );
        light_col += oren_nayar(
        lights[i].pos_radius,
        lights[i].colour.rgb,
        n,
        _input.world_pos.xyz,
        camera_view_pos.xyz,
        roughness,
        albedo.rgb
        );
        float a = point_light_attenuation_cutoff( lights[i].pos_radius, _input.world_pos.xyz );
        light_col *= a;
        float s = sdf_shadow_trace(max_samples, lights[i].pos_radius.xyz, _input.world_pos.xyz, scale, tr1, sdf_shadow.world_matrix_inv, inv_rot);
        light_col *= smoothstep( 0.0, 0.1, s);
        lit_colour += light_col;
    }
    int spot_start = point_end;
    int spot_end = int(light_info.y) + int(light_info.z);
    for(int i = spot_start; i < spot_end; ++i )
    {
        float3 light_col = float3( 0.0, 0.0, 0.0 );
        light_col += cook_torrence(
        lights[i].pos_radius,
        lights[i].colour.rgb,
        n,
        _input.world_pos.xyz,
        camera_view_pos.xyz,
        albedo.rgb,
        specular_sample.rgb,
        roughness,
        reflectivity
        );
        light_col += oren_nayar(
        lights[i].pos_radius,
        lights[i].colour.rgb,
        n,
        _input.world_pos.xyz,
        camera_view_pos.xyz,
        roughness,
        albedo.rgb
        );
        float a = spot_light_attenuation(lights[i].pos_radius,
        lights[i].dir_cutoff,
        lights[i].data.x,
        _input.world_pos.xyz );
        light_col *= a;
        float s = sdf_shadow_trace(max_samples, lights[i].pos_radius.xyz, _input.world_pos.xyz, scale, tr1, sdf_shadow.world_matrix_inv, inv_rot);
        light_col *= smoothstep( 0.0, 0.1, s);
        lit_colour += light_col;
    }
    float pi = 3.14159265359;
    int num_area_lights = int(area_light_info.x);
    for(int i = 0; i < num_area_lights; ++i)
    {
        float3 v = -normalize(_input.world_pos.xyz - camera_view_pos.xyz);
        float3 pos = _input.world_pos.xyz;
        float3 points[4];
        for(int j = 0; j < 4; ++j)
        points[j] = area_lights[i].corners[j].xyz;
        float diff_sum = area_light_diffuse(points, pos, n, v);
        float3 diff = area_lights[i].colour.rgb * diff_sum;
        float spec_sum = area_light_specular(points, pos, ro_sample.x, n, v);
        float3 spec = area_lights[i].colour.rgb * spec_sum;
        float3 light_col = (spec.rgb + diff.rgb) / (2.0 * pi);
        lit_colour += light_col;
    }
    int ts = num_area_lights;
    int num_area_lights_textured = int(area_light_info.y);
    for(int i = ts; i < ts + num_area_lights_textured; ++i)
    {
        float slice = area_lights[i].colour.w;
        float levels = 8.0;
        float2 inv_texel = float2(1.0/640.0, 1.0/480.0);
        float2 inv_texel_x = float2(1.0, 1.0) - inv_texel;
        float3 points[4];
        for(int j = 0; j < 4; ++j)
        points[j] = area_lights[i].corners[j].xyz;
        float3 v = -normalize(_input.world_pos.xyz - camera_view_pos.xyz);
        float3 pos = _input.world_pos.xyz;
        float4 diff_uv = area_light_diffuse_uv(points, pos, n, v);
        float2 duv = clamp(diff_uv.xy, inv_texel, inv_texel_x);
        float3 diff = sample_texture_array_level( area_light_textures, duv, slice, diff_uv.z * levels).rgb * diff_uv.w;
        float4 spec_uv = area_light_specular_uv(points, pos, ro_sample.x, n, v);
        float2 suv = clamp(spec_uv.xy, inv_texel, inv_texel_x);
        float3 spec = sample_texture_array_level(area_light_textures, suv, slice, spec_uv.z * levels).rgb * spec_uv.w;
        float3 light_col = (spec.rgb + diff.rgb) / (2.0 * pi);
        lit_colour += light_col;
    }
    _output.colour.rgb = lit_colour.rgb;
    _output.colour.a = albedo.a;
    //assign glsl global outputs from structs
    colour_ps_output = _output.colour;
}
