# pmfx-shader

Cross platform shader compillation, with outputted reflection info, c++ header with shader structs, techniques and compile time permutation evaluation.

## Targets

- glsl 330, 420, 450.
- SPIR-V.
- hlsl sm 3.0, sm 4.0, sm 5.0.
- metal.

## Features

### HLSL Syntax

Use hlsl syntax everwhere for shaders, with some small differences:

#### Use structs for inputs and outputs.

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
    structured_buffer_rw( struct, rw_boids, 12);
    structured_buffer( boids, read_boids, 13);
};
```

### Techniques

### Permutations

### C++ Header

### JSON Reflection Info

