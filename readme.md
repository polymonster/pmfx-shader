# pmfx-shader

Cross platform shader compillation, with outputted reflection info, c++ header with shader structs, techniques and compile time permutation evaluation. 

This is a small part of the larger pmfx system found in pmtech, it has been moved into a separate repository to be used with other projects, if you are interested to see how pmfx shaders are integegrated please take a look here.

## Targets

- glsl 330, 420, 450.
- SPIR-V.
- hlsl sm 3.0, sm 4.0, sm 5.0.
- metal.

## Features

### Simple

A single file does all the shader parsing and code generation. Simple syntax changes are handled through macros and defines found in platform, so it is simple to add new features or change things to behave how you like.

### HLSL Syntax

Use hlsl syntax everwhere for shaders, with some small differences:

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

#### Shader resources declaration differ slightly.
```c
shader_resources
{
    texture_2d( diffuse_texture, 0 );
    texture_2dms( float4, 2, texture_msaa_2, 0 );
    structured_buffer_rw( boids, rw_boids, 12);
    structured_buffer( boids, read_boids, 13);
};
```

### Includes

```c
#include "libs/lighting.pmfx"
#include "libs/skinning.pmfx"
#include "libs/globals.pmfx"
#include "libs/sdf.pmfx"
#include "libs/area_lights.pmfx"
```

### Techniques

Single .pmfx file can contain multiple shader functions so you can share functionality, you can define a block of json in the shader to configure techniques, simply specify vs, ps or cs to select which function in the source to use for that shader stage.

```json
pmfx:
{    
    "single_light_directional":
    {
        "vs": "vs_main",
        "ps": "ps_single_light"
    }
    
    "compute_job"
    {
        "cs": "cs_some_job"
    }
```

You can also use json to specify technique constants with range and ui type.. so you can later hook them into a gui:

```json
"constants":
{
    "albedo"      : { "type": "float4", "widget": "colour", "default": [1.0, 1.0, 1.0, 1.0] },
    "roughness"   : { "type": "float", "widget": "slider", "min": 0, "max": 1, "default": 0.5 },
    "reflectivity": { "type": "float", "widget": "slider", "min": 0, "max": 1, "default": 0.3 },
}
```

Access to technique constants is done with m_prefix.

```hlsl
ps_output ps_main(vs_output input)
{
    float4 col = m_albedo;
}
```

### Inherit

You can inherit techniques or technique constants by simply adding an inherit into the json.

```json
"gbuffer":
{
    "vs": "vs_main",
    "ps": "ps_gbuffer",

    "inherit_constants": ["forward_lit"]
}
```

### Permutations

Permutations provide an uber shader style compile time branch evaluation to generate optimal shaders but allowing for flexibility to share code as much as possible. The pmfx json block is used here again, you can specify permutations inside a technique.

```json
"gbuffer":
{
    "vs": "vs_main",
    "ps": "ps_gbuffer",

    "permutations":
    {
        "SKINNED": [31, [0,1]],
        "INSTANCED": [30, [0,1]],
        "UV_SCALE": [1, [0,1]]
    },

    "inherit_constants": ["forward_lit"]
},
```

For each permutation a shader is generated with the technique plus the permutation id. The id is generated from the values passed in the permutation object.

### C++ Header

After compilation a header is output for each .pmfx file containing c struct declarations for the cbuffers, technique constant buffers and vertex inputs. It also containts defines for the shader permutation id / flags.

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
}
```

### JSON Reflection Info

Each .pmfx file comes along with a json file containing reflection info. This info contains the locations textures / buffers are bound to, the size of structs, vertex layout description and more. 

```json
"texture_sampler_bindings": [
    {
        "name": "gbuffer_albedo",
        "data_type": "float4",
        "fragments": 1,
        "type": "texture_2d",
        "unit": 0
    },
   
"vs_inputs": [
    {
        "name": "position",
        "semantic_index": 0,
        "semantic_id": 1,
        "size": 16,
        "element_size": 4,
        "num_elements": 4,
        "offset": 0
    },
```

