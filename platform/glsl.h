// precision qualifiers
#ifdef GLES
precision highp float;
precision highp samplerCube;
precision highp sampler2D;
#endif

#ifdef GLES3
precision highp sampler2DShadow;
precision highp sampler2DArray;
precision highp sampler2DArrayShadow;
precision highp sampler3D;
#endif

#ifdef PMFX_GLES_COMPUTE
precision highp image2D;
#endif

#ifdef PMFX_GL_OES_EGL_image_external
precision highp samplerExternalOES;
#endif

// defs
#define float4x4 mat4
#define float3x3 mat3
#define float2x2 mat2
#define float4 vec4
#define float3 vec3
#define float2 vec2
#define uint4 uvec4
#define uint3 uvec3
#define uint2 uvec2
#define int4 ivec4
#define int3 ivec3
#define int2 ivec2
#define	read3 ivec3
#define read2 ivec2
#define modf mod
#define fmod mod
#define frac fract
#define lerp mix
#define mul( A, B ) ((A) * (B))
#define mul_tbn( A, B ) ((B) * (A))
#define saturate( A ) (clamp( A, 0.0, 1.0 ))
#define atan2( A, B ) (atan(A, B))
#define ddx dFdx
#define ddy dFdy
#define _pmfx_unroll
#define _pmfx_loop
    
// texture location binding is not supported on all glsl version's
#ifdef PMFX_BINDING_POINTS
#if PMFX_TEXTURE_OFFSET
#define _tex_binding(sampler_index) layout(binding = sampler_index + PMFX_TEXTURE_OFFSET)
#define _compute_tex_binding binding(layout_index) = layout_index + PMFX_TEXTURE_OFFSET
#else
#define _tex_binding(sampler_index) layout(binding = sampler_index)
#define _compute_tex_binding(layout_index) binding = layout_index
#endif
#else
#define _tex_binding(sampler_index)  
#endif

// textures
#define texture_2d( sampler_name, sampler_index ) _tex_binding(sampler_index) uniform sampler2D sampler_name
#define texture_cube( sampler_name, sampler_index ) _tex_binding(sampler_index) uniform samplerCube sampler_name

// depth text formats for compare samples
#define depth_2d( sampler_name, sampler_index ) _tex_binding(sampler_index) uniform sampler2DShadow sampler_name
#define depth_cube( sampler_name, sampler_index ) _tex_binding(sampler_index) uniform samplerCubeShadow sampler_name

// non gles2 textures
#ifndef GLES2
#define texture_2d_array( sampler_name, sampler_index ) _tex_binding(sampler_index) uniform sampler2DArray sampler_name
#define texture_3d( sampler_name, sampler_index ) _tex_binding(sampler_index) uniform sampler3D sampler_name
#define depth_2d_array( sampler_name, sampler_index ) _tex_binding(sampler_index) uniform sampler2DArrayShadow sampler_name
#else
#define texture_2d_array( sampler_name, sampler_index ) uniform sampler2D sampler_name
#define texture_3d( sampler_name, sampler_index ) uniform sampler2D sampler_name
#define depth_2d_array( sampler_name, sampler_index ) uniform sampler2D sampler_name
#endif

// multisample texture
#ifdef GLES
#define sample_texture_2dms( sampler_name, x, y, fragment ) texture( sampler_name, vec2(0.0, 0.0) )
#define texture_2dms( type, samples, sampler_name, sampler_index ) uniform sampler2D sampler_name
#define texture_cube_array( sampler_name, sampler_index ) uniform int sampler_name
#define depth_cube_array( sampler_name, sampler_index ) uniform int sampler_name
#else
#define sample_texture_2dms( sampler_name, x, y, fragment ) texelFetch( sampler_name, ivec2( x, y ), fragment )
#define texture_2dms( type, samples, sampler_name, sampler_index ) _tex_binding(sampler_index) uniform sampler2DMS sampler_name
#define texture_cube_array( sampler_name, sampler_index ) _tex_binding(sampler_index) uniform samplerCubeArray sampler_name
#define depth_cube_array( sampler_name, sampler_index ) _tex_binding(sampler_index) uniform samplerCubeArrayShadow sampler_name
#endif

// extension textures
#ifdef PMFX_GL_OES_EGL_image_external
#define texture_2d_external(name, sampler_index) uniform samplerExternalOES name
#else
#define texture_2d_external(name, sampler_index) texture_2d(name, sampler_index)
#endif

// compute
#if !defined(GLES) || defined(PMFX_GLES_COMPUTE)
#define texture_2d_rw( image_name, layout_index ) layout (_compute_tex_binding(layout_index), rgba8) uniform image2D image_name
#define texture_2d_r( image_name, layout_index ) layout (_compute_tex_binding(layout_index), rgba8) uniform readonly image2D image_name
#define texture_2d_w( image_name, layout_index ) layout (_compute_tex_binding(layout_index), rgba8) uniform writeonly image2D image_name
#define texture_3d_rw( image_name, layout_index ) layout (_compute_tex_binding(layout_index), rgba8) uniform image3D image_name
#define texture_3d_r( image_name, layout_index ) layout (_compute_tex_binding(layout_index), rgba8) uniform readonly image3D image_name
#define texture_3d_w( image_name, layout_index ) texture3d_rw(image_name, layout_index)
#define texture_2d_array_rw( image_name, layout_index ) layout (_compute_tex_binding(layout_index), rgba8) uniform image2DArray image_name
#define texture_2d_array_r( image_name, layout_index ) layout (_compute_tex_binding(layout_index), rgba8) uniform readonly image2DArray image_name
#define texture_2d_array_w( image_name, layout_index ) texture2d_array_rw(image_name, layout_index)
#define read_texture( image_name, coord ) imageLoad(image_name, coord)
#define write_texture( image_name, value, coord ) imageStore(image_name, coord, value)
#define read_texture_array( image_name, coord, slice ) imageLoad(image_name, ivec3(coord.xy, slice))
#define write_texture_array( image_name, value, coord, slice ) imageStore(image_name, ivec3(coord.xy, slice), value)

// bindless
#define texture2d_table(name, type, dimension, register_index, space_index) _compute_tex_binding(register_index) uniform sampler2D name##dimension
#define texture2d_rw_table(name, type, dimension, register_index, space_index) layout (_compute_tex_binding(register_index), rgba8) uniform image2D name##dimension
#define cbuffer_table(name, type, dimension, register_index, space_index) layout(std430, binding=register_index) buffer name##_buffer { type name##dimension; }

#endif

#ifndef GLES
#define structured_buffer(type, name, index) layout(std430, binding=index) buffer name##_buffer { type name[]; }
#define structured_buffer_rw(type, name, index) layout(std430, binding=index) buffer name##_buffer { type name[]; }
#define atomic_counter(name, index) structured_buffer_rw(uint, name, index)
#else
#ifdef PMFX_GLES_COMPUTE
#define structured_buffer(type, name, index, buffer_name) layout(std430, binding=index) buffer buffer_name { type name[]; }
#define structured_buffer_rw(type, name, index, buffer_name) layout(std430, binding=index) buffer buffer_name { type name[]; }
#define atomic_counter(name, index, buffer_name) structured_buffer_rw(uint, name, index, buffer_name)
#endif
#endif

// sampler
#define sample_texture( sampler_name, V ) texture( sampler_name, V )
#define sample_texture_level( sampler_name, V, l ) textureLod( sampler_name, V, l )
#define sample_texture_grad( sampler_name, V, vddx, vddy ) textureGrad( sampler_name, V, vddx, vddy )
#define sample_texture_array( sampler_name, V, a ) texture( sampler_name, vec3(V, a) )
#define sample_texture_array_level( sampler_name, V, a, l ) textureLod( sampler_name, vec3(V, a), l )
#define sample_depth_compare( name, tc, compare_value ) texture( name, vec3(tc.xy, compare_value) )
#define sample_depth_compare_array( name, tc, a, compare_value ) texture( name, vec4(tc.xy, a, compare_value) )
#define sample_depth_compare_cube( name, tc, compare_value ) texture( name, vec4(tc.xyz, compare_value) )

// cube arrays are not supoorted on webgl / gles 3.0-
#ifdef PMFX_TEXTURE_CUBE_ARRAY
#define sample_texture_cube_array_level( sampler_name, V, a, l ) textureLod( sampler_name, vec4(V, a), l )
#define sample_texture_cube_array( sampler_name, V, a ) texture( sampler_name, vec4(V, a))
#define sample_depth_compare_cube_array( name, V, a, compare_value ) texture( name, vec4(V.xyz, a), compare_value )
#else
#define sample_texture_cube_array_level( sampler_name, V, a, l ) vec4(0.0, 0.0, 0.0, 0.0)
#define sample_texture_cube_array( sampler_name, V, a ) vec4(0.0, 0.0, 0.0, 0.0)
#define sample_depth_compare_cube_array( name, V, a, compare_value ) vec4(0.0, 0.0, 0.0, 0.0)
#endif

// glsl 200 / gles 1.0 texture sample macros
#define sample_texture_2d( sampler_name, V ) texture2D( sampler_name, V )
#define sample_texture_array_2d( sampler_name, V, a ) texture2D( sampler_name, V ) // not supported
#define sample_texture_level_2d( sampler_name, V, l ) texture2DLod( sampler_name, V, l )
#define sample_texture_grad_2d( sampler_name, V, vddx, vddy ) texture2DLod( sampler_name, V, 0.0 )
#define sample_texture_cube( sampler_name, V ) textureCube( sampler_name, V )
#define sample_texture_level_cube( sampler_name, V, l ) textureCubeLod( sampler_name, V, l )
#define sample_texture_grad_cube( sampler_name, V, vddx, vddy ) textureCubeLod( sampler_name, V, 0.0 )

// matrix
#define to_3x3( M4 ) float3x3(M4)
#define from_columns_2x2(A, B) (transpose(float2x2(A, B)))
#define from_rows_2x2(A, B) (float2x2(A, B))
#define from_columns_3x3(A, B, C) (transpose(float3x3(A, B, C)))
#define from_rows_3x3(A, B, C) (float3x3(A, B, C))
#define from_columns_4x4(A, B, C, D) (transpose(float4x4(A, B, C, D)))
#define from_rows_4x4(A, B, C, D) (float4x4(A, B, C, D))
#define unpack_vb_instance_mat( mat, r0, r1, r2, r3 ) mat[0] = r0; mat[1] = r1; mat[2] = r2; mat[3] = r3;
#define to_data_matrix(mat) mat

// clip
#define remap_z_clip_space( d ) d // gl clip space is -1 to 1, and this is normalised device coordinate
#define remap_depth( d ) (d = d * 0.5 + 0.5)
#define remap_ndc_ray( r ) float2(r.x, r.y)  
#define depth_ps_output gl_FragDepth

// atomics
#if !defined(GLES) || defined(PMFX_GLES_COMPUTE)
#define atomic_store(atomic, value) atomicExchange(atomic, value)
#define atomic_increment(atomic, original) original = atomicAdd(atomic, 1u)
#define atomic_decrement(atomic, original) original = atomicExchange(atomic, atomic - 1u)
#define atomic_add(atomic, value, original) original = atomicAdd(atomic, value)
#define atomic_subtract(atomic, value, original) original = atomicExchange(atomic, atomic - value)
#define atomic_min(atomic, value, original) original = atomicMin(atomic, value)
#define atomic_max(atomic, value, original) original = atomicMax(atomic, value)
#define atomic_and(atomic, value, original) original = atomicAnd(atomic, value)
#define atomic_or(atomic, value, original) original = atomicOr(atomic, value)
#define atomic_xor(atomic, value, original) original = atomicXor(atomic, value)
#define atomic_exchange(atomic, value, original) original = atomicExchange(atomic, value)
#define threadgroup_barrier() barrier()
#define device_barrier() barrier()
#endif
