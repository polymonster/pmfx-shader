# pmfx-shader
[![tests](https://github.com/polymonster/pmfx-shader/actions/workflows/tests.yaml/badge.svg)](https://github.com/polymonster/pmfx-shader/actions/workflows/tests.yaml)[![release](https://github.com/polymonster/pmfx-shader/actions/workflows/release.yaml/badge.svg)](https://github.com/polymonster/pmfx-shader/actions/workflows/release.yaml)  

A cross platform shader language with multi-threaded offline compilation or platform shader source code generation. Output json reflection info and c++ header with your shaders structs, fx-like techniques and compile time branch evaluation via (uber-shader) "permutations". Version 1.0 is now in maintenence mode and version 2.0 is in progress which aims to offer wider support for more modern GPU features.

## Supported Targets

- HLSL Shader Model 3+
- GLSL 330+
- GLES 300+ (WebGL 2.0)
- GLSL 200 (compatibility)
- GLES (WebGL 1.0) (compatibility)
- SPIR-V. (Vulkan, OpenGL)
- Metal 1.0+ (macOS, iOS, tvOS)
- PSSL
- NVN (Nintendo Switch)

(compatibility) platforms for older hardware might not support all pmfx features and may have missing legacy features.

## Dependencies

Windows users need [vcredist 2013](https://www.microsoft.com/en-us/download/confirmation.aspx?id=40784) for the glsl/spirv validator.

## Console Platforms

Compilation for Orbis and Nvn is possible but you will need the SDK's installed and the environment variables set.

## Usage

You can use from source by cloning this repository and build `pmfx.py`, or install the latest packaged [release](https://github.com/polymonster/pmfx-shader/releases) if you do not need access to the source code.

```text
py -3 build.py -help (windows)
python3 build.py -help (macos/linux)

--------------------------------------------------------------------------------
pmfx shader (v2.0) ------------------------------------------------------------
--------------------------------------------------------------------------------
commandline arguments:
    -v1 compile using pmfx version 1 (legacy) will use v2 otherwise
    -shader_platform <hlsl, glsl, gles, spirv, metal, pssl, nvn>
    -shader_version (optional) <shader version unless overridden in technique>
        hlsl: 3_0, 4_0 (default), 5_0
        glsl: 200, 330 (default), 420, 450
        gles: 100, 300, 310, 350
        spirv: 420 (default), 450
        metal: 2.0 (default)
        nvn: (glsl)
    -metal_sdk [metal only] <iphoneos, macosx, appletvos>
    -metal_min_os (optional) <9.0 - 13.0 (ios), 10.11 - 10.15 (macos)>
    -nvn_exe [nvn only] <path to execulatble that can compile glsl to nvn glslc>
    -extensions (optional) <list of glsl extension strings separated by spaces>
    -i <list of input files or directories separated by spaces>
    -o <output dir for shaders>
    -t <output dir for temp files>
    -h (optional) <output dir header file with shader structs>
    -d (optional) generate debuggable shader
    -root_dir (optional) <directory> sets working directory here
    -source (optional) (generates platform source into -o no compilation)
    -stage_in <0, 1> (optional) [metal only] (default 1)
        uses stage_in for metal vertex buffers, 0 uses raw buffers
    -cbuffer_offset (optional) [metal only] (default 4)
        specifies an offset applied to cbuffer locations to avoid collisions with vertex buffers
    -texture_offset (optional) [vulkan only] (default 32)
        specifies an offset applied to texture locations to avoid collisions with buffers
    -v_flip (optional) (inserts glsl uniform to conditionally flip verts in the y axis)
--------------------------------------------------------------------------------
```

## Version 2 (Experimental)

Version 2 is currently work in progress, documentation will be updated in due course. Version 2 will use Microsoft DXC to cross compile to SPIRV and exposes support to specify entire pipeline state objects using jsn compatible with Vulkan, Direct3D 12 and Metal. Currently only HLSL is the only supported platform, others will become available via SPIRV-cross and DXC.

## Version 1 (Maintenance Mode)

A single file does all the shader parsing and code generation. Simple syntax changes are handled through macros and defines found in [platform](https://github.com/polymonster/pmfx-shader/tree/master/platform), so it is simple to add new features or change things to behave how you like. More complex differences between shader languages are handled through code-generation.  

This is a small part of the larger [pmfx](https://github.com/polymonster/pmtech/wiki/Pmfx) system found in [pmtech](https://github.com/polymonster/pmtech), it has been moved into a separate repository to be used with other projects, if you are interested to see how pmfx shaders are integrated please take a look [here](https://github.com/polymonster/pmtech).

### Compiling Version 1 Examples

```text
// metal macos
python3 pmfx.py -v1 -shader_platform metal -metal_sdk macosx -metal_min_os 10.14 -shader_version 2.2 -i examples/v1 -o output/bin -h output/structs -t

// metal ios
python3 pmfx.py -v1 -shader_platform metal -metal_sdk iphoneos -metal_min_os 0.9 -shader_version 2.2 -i examples/v1 -o output/bin -h output/structs -t 

// spir-v vulkan
python3 pmfx.py -v1 -shader_platform spirv -i examples/v1 -o output/bin -h output/structs -t output/temp

// hlsl d3d11
python3 pmfx.py -v1 -shader_platform hlsl -shader_version 4_0 -i examples/v1 -o output/bin -h output/structs -t output/temp

// glsl
python3 pmfx.py -v1 -shader_platform glsl -shader_version 330 -i examples/v1 -o output/bin -h output/structs -t output/temp

// gles
python3 pmfx.py -v1 -shader_platform gles -shader_version 320 -i examples/v1 -o output/bin -h output/structs -t output/temp
```

### Shader Language 

Use mostly HLSL syntax for shaders, with some small differences:

#### Always use structs for inputs and outputs  

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

pmfx will generate an input layout for you in the json reflection info, containing the stride of the vertex layout and the byte offsets to each of the elements. If you choose to use this, pmfx will assume the following sizes for semantics:

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
texture_2d_external( sampler_name, layout_index ); // gles specific extension

// depth formats are required for sampler compare ops
depth_2d( sampler_name, layout_index ); 
depth_2d_array( sampler_name, layout_index );
depth_cube( sampler_name, layout_index ); 
depth_cube_array( sampler_name, layout_index );

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
atomic_counter(name, index);

// bindless resouce tables
// name, type, dimension, register, space
texture2d_table(texture0, float4, [], 0, 0);
cbuffer_table(constant_buffer0, data, [], 1, 0);
sampler_state_table(sampler0, [], 0);

// smapler type
sampler_state(sampler0, 0);
```

#### Accessing resources

Textures and samplers are combined when using a binding renderer model. a `texture_2d` declares a texture and a sampler on the corresponding texture and sampler register index which is passed into the macro. The `sample_texture` can be used to sample textures of varying dimensions.

```c
// sample texture
float4 col = sample_texture( diffuse_texture, texcoord.xy );
float4 cube = sample_texture( cubemap_texture, normal.xyz );
float4 msaa_sample = sample_texture_2dms( msaa_texture, x, y, fragment );
float4 level = sample_texture_level( texture, texcoord.xy, mip_level);
float4 array = sample_texture_array( texture, texcoord.xy, array_slice);
float4 array_level = sample_texture_array_level( texture, texcoord.xy, array_slice, mip_level);

// sample compare
float shadow = sample_depth_compare( shadow_map, texcoord.xy, compare_ref);
float shadow_array = sample_depth_compare_array( shadow_map, texcoord.xy, array_slice, compare_ref);
float cube_shadow = sample_depth_compare_cube( shadow_map, texcoord.xyz, compare_ref);
float cube_shadow_array = sample_depth_compare_cube_array( shadow_map, texcoord.xyz, array_slice, compare_ref);

// compute rw texture
float4 rwtex = read_texture( tex_rw, gid );
write_texture(rwtex, val, gid);

// compute structured buffer
struct val = structured_buffer[gid]; // read
structured_buffer[gid] = val;        // write

// read type!
// glsl expects ivec (int) to be pass to imageLoad, hlsl and metal require uint... 
// there is a `read` type you can use to be platform safe
read3 read_coord = read3(x, y, z);
read_texture( tex_rw, read_coord );
```

### Bindless resources

Initial implementation of bindless resources is implemented and tested with HLSL, more platforms will follow. This is still work in progress. 

Define resource tables types with `[]` dimensions (you could use multi-dimensional resources `[10][5][2]` for instance). Use `[]` empty square brackets for unbounded sizes. The resources are called `tables` due to ambiguity with using `array` due to `texture_2d_array` and other dimensional array types. 

With bindless rendering, textures and samplers need to be decoupled so to sample a texture you supply both a `texture` and a `sampler` to the `texture_sample` macro, which can be used on textures and tables of varying dimensitonality. Constant buffer tables can be accessed through raw `[]` operator access. Smplers can also be a `table` type.

Reflection info for creating descriptor sets from these resource tables will be generated and output into the `.json` file after compilation.

```hlsl
shader_resources
{
    // resource table types
    texture2d_table(texture0, float4, [6], 0, 0);
    cbuffer_table(constant_buffer0, data, [6], 1, 0);
    sampler_state_table(sampler_table0, [], 0);

    // separate sampler
    sampler_state(sampler0, 0);
};

ps_output ps_main(ps_input input)
{
    ps_output output;

    float4 final = float4(0.0, 0.0, 0.0, 0.0);
    float2 uv = input.colour.rg * float2(1.0, -1.0);

    float4 r0 = texture_sample(texture0[0], sampler0, uv * 2.0);
    float4 r1 = texture_sample(texture0[1], sampler0, (uv * 2.0) + float2(0.0, 1.0));
    float4 r2 = texture_sample(texture0[2], sampler0, (uv * 2.0) + float2(1.0, 1.0));
    float4 r3 = texture_sample(texture0[5], sampler0, (input.colour.rg * 2.0) + float2(1.0, 0.0));
    r3 += texture_sample(texture0[6], sampler0, (input.colour.rg * 2.0) + float2(1.0, 0.0));

    // ..

    final *= constant_buffer0[4].rgba;
}
```

### cbuffers

cbuffers are a unique kind of resource, this is just because they are so in HLSL. you can use cbuffers as you normally do in HLSL.

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

#### GLES 2.0 / GLSL 2.0 cbuffers

cbuffers are emulated for older glsl versions, a cbuffer is packed into a single float4 array. The uniform float4 array (`glUniform4fv`) is named after the cbuffer, you can find the uniform location from this name using `glUniformLocation`. The count of the float4 array is the number of members the cbuffer where float4 and float4x4 are supported and float4x4 count for 4 array elements. You can use the generated c++ structs from pmfx to create a coherent copy of the uniform data on the cpu.

### Atomic Operations

Support for glsl, hlsl and metal compatible atomics and memory barriers is available. The atomic_counter resource type is a RWStructuredBuffer<uint> in hlsl, a atomic_uint read/write buffer in Metal and a uniform atomic_uint in GLSL.

```hlsl
// types
atomic_uint u;
atomic_int i;

// operations
atomic_load(atomic, original)
atomic_store(atomic, value)
atomic_increment(atomic, original)
atomic_decrement(atomic, original)
atomic_add(atomic, value, original)
atomic_subtract(atomic, value, original)
atomic_min(atomic, value, original)
atomic_max(atomic, value, original)
atomic_and(atomic, value, original)
atomic_or(atomic, value, original)
atomic_xor(atomic, value, original)
atomic_exchange(atomic, value, original)
threadgroup_barrier()
device_barrier()

// usage
shader_resources
{
    atomic_counter(counter, 0); // counter bound to index 0
}

// increments counter and stores the original value in 'index'
uint index = 0;
atomic_increment(counter, index);
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

#### Extensions

To enable glsl extensions you can pass a list of strings to the `-extensions` commandline argument. The glsl extension will be inserted to the top of the generated code with `: require` set:

```text
-extensions GL_OES_EGL_image_external GL_OES_get_program_binary
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

### cbuffer padding

HLSL/Direct3D requires cbuffers to be padded to 16 bytes alignment, pmfx allows you to create cbuffers with any size and will pad the rest out for you.

### Techniques

Single .pmfx file can contain multiple shader functions so you can share functionality, you can define a block of [jsn](https://github.com/polymonster/jsn) in the shader to configure techniques. (jsn is a more lenient and user friendly data format similar to json).

Simply specify `vs`, `ps` or `cs` to select which function in the source to use for that shader stage. If no pmfx: json block is found you can still supply `vs_main` and `ps_main` which will be output as a technique named "default".

```yaml
pmfx:
{    
    gbuffer: {
        vs: vs_main
        ps: ps_gbuffer
    }
        
    zonly: {
        vs: vs_main_zonly
        ps: ps_null
    }
}
```

You can also use json to specify technique constants with range and ui type.. so you can later hook them into a gui:

```yaml
constants:
{
    albedo: { 
        type: float4, widget: colour, default: [1.0, 1.0, 1.0, 1.0]
    }
    roughness: { 
        type: float, widget: slider, min: 0, max: 1, default: 0.5
    }
    reflectivity: { 
        type: float, widget: slider, min: 0, max: 1, default: 0.3
    }
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

```yaml
gbuffer(forward_lit):
{
    vs: vs_main
    ps: ps_gbuffer

    permutations:
    {
        SKINNED: [31, [0,1]]
        INSTANCED: [30, [0,1]]
        UV_SCALE: [1, [0,1]]
    }
}
```

gbuffer inherits from forward lit, by putting the base clase inside brackets.

### Permutations

Permutations provide an uber shader style compile time branch evaluation to generate optimal shaders but allowing for flexibility to share code as much as possible. The pmfx block is used here again, you can specify permutations inside a technique.

```yaml
permutations:
{
    SKINNED: [31, [0,1]]
    INSTANCED: [30, [0,1]]
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

Adding permutations can cause the number of generated shaders to grow exponentially, pmfx will detect redundant shader combinations using md5 hashing, to re-use duplicate permutation combinations and avoid un-necessary compilation.

### C++ Header

After compilation a header is output for each .pmfx file containing c struct declarations for the cbuffers, technique constant buffers and vertex inputs. You can use these sturcts to fill buffers in your c++ code and use sizeof for buffer update calls in your graphics api. 

It also contains defines for the shader permutation id / flags that you can check and test against to select the correct shader permutations for a draw call (ie. skinned, instanced, etc).

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