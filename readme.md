# pmfx-shader
[![Build Status](https://travis-ci.org/polymonster/pmfx-shader.svg?branch=master)](https://travis-ci.org/polymonster/pmfx-shader) [![Build status](https://ci.appveyor.com/api/projects/status/wohe0i5v0hvnjnfb?svg=true)](https://ci.appveyor.com/project/polymonster/pmfx-shader)

Cross platform shader compilation, with output reflection info, c++ header with shader structs, fx-like techniques and compile time branch evaluation via (uber-shader) "permutations". 

A single file does all the shader parsing and code generation. Simple syntax changes are handled through macros and defines found in [platform](https://github.com/polymonster/pmfx-shader/tree/master/platform), so it is simple to add new features or change things to behave how you like. More complex differences between shader languages (such as Metals lack of global textures / buffers) are handled through code-generation. 

pmfx currently supports a subset of shader functionality with features added on an as-needed basis, it has been used in a number of personal projects as well as some upcoming commercial projects so the feature set is fairly compehensive but by no-means complete.

This is a small part of the larger [pmfx](https://github.com/polymonster/pmtech/wiki/Pmfx) system found in [pmtech](https://github.com/polymonster/pmtech), it has been moved into a separate repository to be used with other projects, if you are interested to see how pmfx shaders are integrated please take a look [here](https://github.com/polymonster/pmtech).

## Supported Targets

- HLSL Shader Model 3+
- GLSL 330+
- SPIR-V. (Vulkan, OpenGL)
- Metal 1.0+ (macOS, iOS, tvOS) 
- PSSL (wip)

## Dependencies
Windows users need [vcredist 2013](https://www.microsoft.com/en-us/download/confirmation.aspx?id=40784) for the glsl/spirv validator.

## Usage

```
python3 build_pmfx.py -help

--------------------------------------------------------------------------------
pmfx shader (v3) ---------------------------------------------------------------
--------------------------------------------------------------------------------
commandline arguments:
    -shader_platform <hlsl, glsl, gles, spirv, metal>
    -shader_version (optional) <shader version unless overriden in technique>
        hlsl: 3_0, 4_0 (default), 5_0
        glsl: 330 (default), 420, 450
        spirv: 420 (default), 450
        metal: 2.0 (default)
    -metal_sdk [metal only] <iphoneos, macosx, appletvos>
    -metal_min_os (optional) <9.0 - 13.0 (ios), 10.11 - 10.15 (macos)>
    -i <list of input files or directories separated by spaces>
    -o <output dir for shaders>
    -t <output dir for temp files>
    -h <output dir header file with shader structs>
    -d (optional) generate debuggable shader
    -root_dir <directory> sets working directory here
    -source (optional) (generates platform source into -o no compilation)
    -stage_in <0, 1> (optional) [metal only] (default 1) 
        uses stage_in for metal vertex buffers, 0 uses raw buffers
    -cbuffer_offset (optional) [metal only] (default 4) 
        specifies an offset applied to cbuffer locations to avoid collisions with vertex buffers
    -texture_offset (optional) [vulkan only] (default 32) 
        specifies an offset applied to texture locations to avoid collisions with buffers
    -v_flip (optional) (inserts glsl uniform to control geometry flipping)

```

## Compiling Examples

#### Metal for macOS

```
python3 build_pmfx.py -shader_platform metal -metal_sdk macosx -metal_min_os 10.14 -shader_version 2.2 -i examples -o output/bin -h output/structs -t output/temp
```

#### Metal for iOS

```
python3 build_pmfx.py -shader_platform metal -metal_sdk iphoneos -metal_min_os 0.9 -shader_version 2.2 -i examples -o output/bin -h output/structs -t output/temp
```

#### SPIR-V for Vulkan

```
python3 build_pmfx.py -shader_platform spirv -i examples -o output/bin -h output/structs -t output/temp
```

#### HLSL for Direct3D11

```
python3 build_pmfx.py -shader_platform hlsl -shader_version 4_0 -i examples -o output/bin -h output/structs -t output/temp
```

#### GLSL

```
python3 build_pmfx.py -shader_platform glsl -shader_version 330 -i examples -o output/bin -h output/structs -t output/temp
```

## Usage

Use HLSL syntax everwhere for shaders, with some small differences:

#### Always use structs for inputs and outputs.

```hlsl
struct vs_input
{
    float4 position : POSITION;
};

struct vs_output
{
    float4 position : SV_POSITION0;
};

vs_output vs_main( vs_input input )
{
    vs_output output;
    
    output.position = input.position;
    
    return output;
}
```

#### Supported semantics and sizes

```hlsl
POSITION     // 32bit float
TEXCOORD     // 32bit float
NORMAL       // 32bit float
TANGENT      // 32bit float
BITANGENT    // 32bit float
BLENDWEIGHTS // 32bit float
COLOR        // 8bit unsigned int
BLENDINDICES // 8bit unsigned int
```

#### Shader resources

Due to fundamental differences accross shader languages, shader resource declarations and access have a syntax unique to pmfx. Define a block of shader_resources to allow global textures or buffers as supported in HLSL and GLSL.

```c
shader_resources
{
    texture_2d( diffuse_texture, 0 );
    texture_2dms( float4, 2, texture_msaa_2, 0 );
};
```

#### Resource types

```c
// texture types
texture_2d( sampler_name, layout_index );
texture_2dms( type, samples, sampler_name, layout_index );
texture_2d_array( sampler_name, layout_index );
texture_cube( sampler_name, layout_index );
texture_cube_array( sampler_name, layout_index ); // requires sm 4+, gles 400+
texture_3d( sampler_name, layout_index );

// depth formats are required for sampler compare ops
depth_2d( sampler_name, layout_index ); 
depth_2d_array( sampler_name, layout_index );

// compute shader texture types
texture_2d_r( image_name, layout_index );
texture_2d_w( image_name, layout_index );
texture_2d_rw( image_name, layout_index );
texture_3d_r( image_name, layout_index );
texture_3d_w( image_name, layout_index );
texture_3d_rw( image_name, layout_index );
texture_2d_array_r( image_name, layout_index );
texture_2d_array_w( image_name, layout_index );
texture_2d_array_rw( image_name, layout_index );

// compute shader buffer types
structured_buffer( type, name, index );
structured_buffer_rw( type, name, index );
```

#### Accessing resources

```c
// sample texture
float4 col = sample_texture( diffuse_texture, texcoord.xy );
float4 cube = sample_texture( cubemap_texture, normal.xyz );
float4 msaa_sample = sample_texture_2dms( msaa_texture, x, y, fragment );
float4 level = sampler_texture_level( texture, texcoord.xy, mip_level);
float4 array = sampler_texture_array( texture, texcoord.xy, array_slice);
float4 array_level = sampler_texture_array_level( texture, texcoord.xy, array_slice, mip_level);

// sample compare
float shadow = sample_depth_compare( shadow_map, texcoord.xy, compare_ref);
float shadow_array = sample_depth_compare_array( shadow_map, texcoord.xy, array_slice, compare_ref);

// compute rw texture
float4 rwtex = read_texture( tex_rw, gid );
write_texture(rwtex, val, gid);

// compute structured buffer
struct val = structured_buffer[gid]; // read
structured_buffer[gid] = val;        // write
```

#### Cbuffers

Cbuffers are a unique kind of resource, this is just because they are so in HLSL. you can use cbuffers as you normally do in HLSL.

```hlsl
cbuffer per_view : register(b0)
{
    float4x4 view_matrix;
};

cbuffer per_draw_call : register(b1)
{
    float4x4 world_matrix;
};

vs_output vs_main( vs_input input )
{
    vs_output output;
    
    float4 world_pos = mul(input.position, world_matrix);
    output.position = mul(world_pos, view_matrix);
    
    return output;
}
```

#### Includes

Include files are supported even though some shader platforms or versions may not support them natively.

```c
#include "libs/lighting.pmfx"
#include "libs/skinning.pmfx"
#include "libs/globals.pmfx"
#include "libs/sdf.pmfx"
#include "libs/area_lights.pmfx"
```

## Unique pmfx features

### cbuffer_offset / texture_offset

HLSL has different registers for textures, vertex buffers, cbuffers and un-ordered access views. Metal and Vulkan have some differences where the register indices are shared across different resource types. To avoid collisions in different API backends you can supply offsets using the following command line options.

Metal: -cbuffer_offset (cbuffers start binding at this offset to allow vertex buffers to be bound to the slots prior to these offsets)

Vulkan: -texture_offset (textures start binding at this point allowing uniform buffers to bind to the prior slots)

### v_flip

OpenGL has different viewport co-ordinates to texture coordinate so when rendering to the backbuffer vs rendering into a render target you can get output results that are flipped in the y-axis, this can propagate it's way far into a code base with conditional "v_flips" happening during different render passes.

To solve this issue in a cross platform way, pmfx will expose a uniform bool called "v_flip" in all gl vertex shaders, this allows you to conditionally flip the y-coordinate when rendering to the backbuffer or not. 

To make this work make sure you also change the winding glFrontFace(GL_CCW) to glFrontFace(GL_CW).

### Techniques

Single .pmfx file can contain multiple shader functions so you can share functionality, you can define a block of [jsn](https://github.com/polymonster/jsn) in the shader to configure techniques. (jsn is a more lenient and user friendly data format similar to json).

Simply specify vs, ps or cs to select which function in the source to use for that shader stage. If no pmfx: json block is found you can still supply vs_main and ps_main which will be output as a technique named "default".

```c
pmfx:
{    
    gbuffer:
    {
        vs: vs_main,
        ps: ps_gbuffer
    },
        
    zonly:
    {
        vs: vs_main_zonly,
        ps: ps_null
    },
}
```

You can also use json to specify technique constants with range and ui type.. so you can later hook them into a gui:

```c
constants:
{
    albedo      : { type: float4, widget: colour, default: [1.0, 1.0, 1.0, 1.0] },
    roughness   : { type: float, widget: slider, min: 0, max: 1, default: 0.5 },
    reflectivity: { type: float, widget: slider, min: 0, max: 1, default: 0.3 },
}
```

![pmfx constants](https://github.com/polymonster/polymonster.github.io/raw/master/assets/wiki/pmfx-material.png)

Access to technique constants is done with m_prefix.

```hlsl
ps_output ps_main(vs_output input)
{
    float4 col = m_albedo;
}
```

### Inherit

You can inherit techniques by using jsn inherit feature.

```c
gbuffer(forward_lit):
{
    vs: vs_main,
    ps: ps_gbuffer,

    permutations:
    {
        SKINNED: [31, [0,1]],
        INSTANCED: [30, [0,1]],
        UV_SCALE: [1, [0,1]]
    }
},
```

gbuffer inherits from forward lit, by putting the base clase inside brackets.

### Permutations

Permutations provide an uber shader style compile time branch evaluation to generate optimal shaders but allowing for flexibility to share code as much as possible. The pmfx block is used here again, you can specify permutations inside a technique.

```c
permutations:
{
    SKINNED: [31, [0,1]],
    INSTANCED: [30, [0,1]],
    UV_SCALE: [1, [0,1]]
}
```

The first parameter is a bit shift that we can check.. so skinned is 1<<31 and uv scale is 1<<1. The second value is number of options, so in the above example we just have on or off, but you could have a quality level 0-5 for instance.

To insert a compile time evaluated branch in code, use a colon after if / else

```c++
if:(SKINNED)
{
    float4 sp = skin_pos(input.position, input.blend_weights, input.blend_indices);
    output.position = mul( sp, vp_matrix );
}
else:
{
    output.position = mul( input.position, wvp );
}
```

For each permutation a shader is generated with the technique plus the permutation id. The id is generated from the values passed in the permutation object.

Adding permutations can cause the number of generated shaders to grow exponentially, pmfx will detect redundant shader combinations using md5 hashing, to re-use duplicate permutation combinations and avoid un-necessary compilation, additional permutations are compiled asyncrounsly to make use of mulit-core cpu's.

### C++ Header

After compilation a header is output for each .pmfx file containing c struct declarations for the cbuffers, technique constant buffers and vertex inputs. It also containts defines for the shader permutation id / flags that you can check and test against.

```c++
namespace debug
{
    struct per_pass_view
    {
        float4x4 view_projection_matrix;
        float4x4 view_matrix;
    };
    struct per_pass_view_2d
    {
        float4x4 projection_matrix;
        float4 user_data;
    };
    #define OMNI_SHADOW_SKINNED 2147483648
    #define OMNI_SHADOW_INSTANCED 1073741824
    #define FORWARD_LIT_SKINNED 2147483648
    #define FORWARD_LIT_INSTANCED 1073741824
    #define FORWARD_LIT_UV_SCALE 2
    #define FORWARD_LIT_SSS 4
    #define FORWARD_LIT_SDF_SHADOW 8
}
```

### JSON Reflection Info

Each .pmfx file comes along with a json file containing reflection info. This info contains the locations textures / buffers are bound to, the size of structs, vertex layout description and more, at this point please remember the output reflection info is fully compliant json, and not lightweight jsn.. this is because of the more widespread support of json.

```json
"texture_sampler_bindings": [
    {
        "name": "gbuffer_albedo",
        "data_type": "float4",
        "fragments": 1,
        "type": "texture_2d",
        "unit": 0
    }]
   
"vs_inputs": [
    {
        "name": "position",
        "semantic_index": 0,
        "semantic_id": 1,
        "size": 16,
        "element_size": 4,
        "num_elements": 4,
        "offset": 0
    }]
```

