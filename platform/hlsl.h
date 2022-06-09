// texture
#define texture_2d_rw( name, index ) RWTexture2D<float4> name : register(u##index)
#define texture_2d_r( name, index ) Texture2D<float4> name : register(t##index)
#define texture_2d_w( name, index ) texture_2d_rw( name, index )
#define texture_3d_rw( name, index ) RWTexture3D<float4> name : register(u##index)
#define texture_3d_r( name, index ) Texture3D<float4> name : register(t##index)
#define texture_3d_w( name, index ) texture_3d_rw( name, index )
#define texture_2d_array_rw( name, index ) RWTexture2DArray<float4> : register(u##index)
#define texture_2d_array_r( name, index ) Texture2DArray<float4> name : register(t##index)
#define texture_2d_array_w( name, index ) texture_2d_array_rw(name, index)
#define read_texture( name, gid ) name[gid]
#define write_texture( name, val, gid ) name[gid] = val
#define read_texture_array( name, gid, slice ) name[uint3(gid.xy, slice)]
#define write_texture_array( name, val, gid, slice ) name[uint3(gid.xy, slice)] = val
#define texture_2d( name, sampler_index ) Texture2D name : register(t##sampler_index); ; SamplerState sampler_##name : register(s##sampler_index)
#define texture_3d( name, sampler_index ) Texture3D name : register(t##sampler_index); ; SamplerState sampler_##name : register(s##sampler_index)
#define texture_2dms( type, samples, name, sampler_index ) Texture2DMS<type, samples> name : register(t##sampler_index); ; SamplerState sampler_##name : register(s##sampler_index)
#define texture_cube( name, sampler_index )    TextureCube name : register(t##sampler_index); ; SamplerState sampler_##name : register(s##sampler_index)
#define texture_2d_array( name, sampler_index ) Texture2DArray name : register(t##sampler_index); ; SamplerState sampler_##name : register(s##sampler_index)
#define texture_cube_array( name, sampler_index ) TextureCubeArray name : register(t##sampler_index); ; SamplerState sampler_##name : register(s##sampler_index)
#define texture_2d_external( name, sampler_index ) texture_2d( name, sampler_index )

// sampler
#define sampler_state(name, sampler_index) SamplerState name : register(s##sampler_index)
#define sampler_state_table(name, dimension, sampler_index) SamplerState name##dimension : register(s##sampler_index)

// bindless resources
#define texture2d_table(name, type, dimension, register_index, space_index) Texture2D<type> name##dimension : register(t##register_index, space##space_index)
#define texture2d_rw_table(name, type, dimension, register_index, space_index) RWTexture2D<type> name##dimension : register(t##register_index, space##space_index)
#define cbuffer_table(name, type, dimension, register_index, space_index) ConstantBuffer<type> name##dimension : register(b##register_index, space##space_index)

// depth texture (required for gl and metal)
#define depth_2d( name, sampler_index ) Texture2D name : register(t##sampler_index); ; SamplerComparisonState sampler_##name : register(s##sampler_index)
#define depth_2d_array( name, sampler_index ) Texture2DArray name : register(t##sampler_index); ; SamplerComparisonState sampler_##name : register(s##sampler_index)
#define depth_cube( name, sampler_index ) TextureCube name : register(t##sampler_index); ; SamplerComparisonState sampler_##name : register(s##sampler_index)
#define depth_cube_array( name, sampler_index ) TextureCubeArray name : register(t##sampler_index); ; SamplerComparisonState sampler_##name : register(s##sampler_index)

// structured buffer
#define structured_buffer_rw( type, name, index ) RWStructuredBuffer<type> name : register(u##index)
#define structured_buffer( type, name, index ) StructuredBuffer<type> name : register(t##index)

// combined texture samplers
#define sample_texture_2dms( name, x, y, fragment ) name.Load( int2(x, y), int(fragment) )
#define sample_texture( name, V ) name.Sample(sampler_##name, V)
#define sample_texture_level( name, V, l ) name.SampleLevel(sampler_##name, V, l)
#define sample_texture_grad( name, V, vddx, vddy ) name.SampleGrad(sampler_##name, V, vddx, vddy )
#define sample_texture_array( name, V, a ) name.Sample(sampler_##name, float3(V.xy, a) )
#define sample_texture_array_level( name, V, a, l ) name.SampleLevel(sampler_##name, float3(V.xy, a), l)
#define sample_texture_cube_array( name, V, a ) name.Sample(sampler_##name, float4(V.xyz, a) )
#define sample_texture_cube_array_level( name, V, a, l ) name.SampleLevel(sampler_##name, float4(V.xyz, a), l)

// separate sampler / textures
#define texture_sample(texture, sampler, coord) texture.Sample(sampler, coord)

// gather / compare
#define sample_depth_compare( name, tc, compare_value ) saturate(name.SampleCmp(sampler_##name, tc, compare_value))
#define sample_depth_compare_array( name, tc, a, compare_value ) saturate(name.SampleCmp(sampler_##name, float3(tc.xy, a), compare_value))
#define sample_depth_compare_cube( name, tc, compare_value ) saturate(name.SampleCmp(sampler_##name, tc, compare_value))
#define sample_depth_compare_cube_array( name, tc, a, compare_value ) saturate(name.SampleCmp(sampler_##name, float4(tc.x, tc.y, tc.z, a), compare_value))

// matrix
#define to_3x3( M4 ) ((float3x3)M4)
#define from_columns_2x2(A, B) (float2x2(A, B))
#define from_rows_2x2(A, B) (transpose(float2x2(A, B)))
#define from_columns_3x3(A, B, C) (float3x3(A, B, C))
#define from_rows_3x3(A, B, C) (transpose(float3x3(A, B, C)))
#define from_columns_4x4(A, B, C, D) (float4x4(A, B, C, D))
#define from_rows_4x4(A, B, C, D) (transpose(float4x4(A, B, C, D)))
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
#define mix lerp
#pragma warning( disable : 3078) // 'i': loop control variable conflicts with a previous declaration
#pragma warning( disable : 4000) // use of potentially uninitialized variable (from function return)
#define	read3 uint3
#define read2 uint2

// atomics
#define atomic_uint uint
#define atomic_int int
#define atomic_counter(name, index) RWStructuredBuffer<uint> name : register(u##index)
#define atomic_load(atomic) atomic
#define atomic_store(atomic, value) atomic = value
#define atomic_increment(atomic, original) InterlockedAdd(atomic, 1, original)
#define atomic_decrement(atomic, original) InterlockedAdd(atomic, (int)-1, original)
#define atomic_add(atomic, value, original) InterlockedAdd(atomic, value, original)
#define atomic_subtract(atomic, value, original) InterlockedAdd(atomic, value, original)
#define atomic_min(atomic, value, original) InterlockedAdd(atomic, (int)value, original)
#define atomic_max(atomic, value, original) InterlockedMin(atomic, value, original)
#define atomic_and(atomic, value, original) InterlockedMax(atomic, value, original)
#define atomic_or(atomic, value, original) InterlockedOr(atomic, value, original)
#define atomic_xor(atomic, value, original) InterlockedXor(atomic, value, original)
#define atomic_exchange(atomic, value, original) InterlockedExchange(atomic, value, original)
#define threadgroup_barrier GroupMemoryBarrierWithGroupSync
#define device_barrier DeviceMemoryBarrierWithGroupSync
