// texture
#define texture2d_rw( name, index ) RWTexture2D<float4> name : register(u##index)
#define texture2d_r( name, index ) Texture2D<float4> name : register(t##index)
#define texture2d_w( name, index ) texture2d_rw( name, index )
#define texture3d_rw( name, index ) RWTexture3D<float4> name : register(u##index)
#define texture3d_r( name, index ) Texture3D<float4> name : register(t##index)
#define texture3d_w( name, index ) texture3d_rw( name, index )
#define read_texture( name, gid ) name[gid]
#define write_texture( name, val, gid ) name[gid] = val
#define texture_2d( name, sampler_index ) Texture2D name : register(t##sampler_index); ; SamplerState sampler_##name : register(s##sampler_index); 
#define texture_3d( name, sampler_index ) Texture3D name : register(t##sampler_index); ; SamplerState sampler_##name : register(s##sampler_index); 
#define texture_2dms( type, samples, name, sampler_index ) Texture2DMS<type, samples> name : register(t##sampler_index); ; SamplerState sampler_##name : register(s##sampler_index); 
#define texture_cube( name, sampler_index )    TextureCube name : register(t##sampler_index); ; SamplerState sampler_##name : register(s##sampler_index); 
#define texture_2d_array( name, sampler_index ) Texture2DArray name : register(t##sampler_index); ; SamplerState sampler_##name : register(s##sampler_index); 
#define texture_cube_array( name, sampler_index ) TextureCubeArray name : register(t##sampler_index); ; SamplerState sampler_##name : register(s##sampler_index); 
// structured buffer
#define structured_buffer_rw( type, name, index ) RWStructuredBuffer<type> name : register(u##index);
#define structured_buffer( type, name, index ) StructuredBuffer<type> name : register(t##index);
// sampler
#define sample_texture_2dms( name, x, y, fragment ) name.Load( int2(x, y), int(fragment) )
#define sample_texture( name, V ) name.Sample(sampler_##name, V)
#define sample_texture_level( name, V, l ) name.SampleLevel(sampler_##name, V, l)
#define sample_texture_grad( name, V, vddx, vddy ) name.SampleGrad(sampler_##name, V, vddx, vddy )
#define sample_texture_array( name, V, a ) name.Sample(sampler_##name, float3(V.xy, a) )
#define sample_texture_array_level( name, V, a, l ) name.SampleLevel(sampler_##name, float3(V.xy, a), l)
#define sample_texture_cube_array( name, V, a ) name.Sample(sampler_##name, float4(V.xyz, a) )
#define sample_texture_cube_array_level( name, V, a, l ) name.SampleLevel(sampler_##name, float4(V.xyz, a), l)
// matrix
#define to_3x3( M4 ) (float3x3)M4
#define from_columns_3x3(A, B, C) (float3x3(A, B, C))
#define from_rows_3x3(A, B, C) (transpose(float3x3(A, B, C)))
#define mul_tbn( A, B ) mul(A, B)
#define unpack_vb_instance_mat( mat, r0, r1, r2, r3 ) mat[0] = r0; mat[1] = r1; mat[2] = r2; mat[3] = r3; mat = transpose(mat)
#define to_data_matrix(mat) transpose(mat)
// clip
#define remap_z_clip_space( d ) (d = d * 0.5 + 0.5)
#define remap_depth( d ) (d)
#define remap_ndc_ray( r ) float2(r.x, r.y * -1.0)
// defs
#define mod(x, y) (x - y * floor(x/y))
#define fract frac
#define _pmfx_unroll [unroll]
#define _pmfx_loop [loop]


