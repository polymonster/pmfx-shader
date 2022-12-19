import build_pmfx
import re
import os
import cgu

# separate name (alpha characters) from index (numerical_characters)
def separate_name_index(src):
    name = re.sub(r'[0-9]', '', src)
    index = re.sub(r'[^0-9]','', src)
    if len(index) == 0:
        index = 0
    index = int(index)
    return (name, index)


# parses the register for the resource unit, ie. : register(t0)
def parse_register(type_dict):
    rp = cgu.find_token("register", type_dict["declaration"])
    type_dict["shader_register"] = None
    type_dict["register_type"] = None
    type_dict["register_space"] = 0
    if rp != -1:
        start, end = cgu.enclose_start_end("(", ")", type_dict["declaration"], rp)
        decl = type_dict["declaration"][start:end]
        multi = decl.split(",")
        for r in multi:
            r = r.strip()
            if r.find("space") != -1:
                type_dict["register_space"] = separate_name_index(r)[1]
            else:
                type_dict["register_type"], type_dict["shader_register"] = separate_name_index(r)


# returns info for types, (num_elements, element_size, total_size)
def get_type_size_info(type):
    lookup = {
        "float": (1, 4, 4),
        "float2": (2, 4, 8),
        "float3": (3, 4, 12),
        "float4": (4, 4, 16),
        "float2x2": (8, 4, 32),
        "float3x4": (12, 4, 48),
        "float4x3": (12, 4, 48),
        "float4x4": (16, 4, 64),
    }
    return lookup[type]


# parses a type and generates a vertex layout, array of elements with sizes and offsets
def parse_vertex_layout(type_dict):
    offset = 0
    layout = list()
    for member in type_dict["members"]:
        semantic_name, semantic_index = separate_name_index(member["semantic"])
        num_elems, elem_size, size = get_type_size_info(member["data_type"])
        input = {
            "name": semantic_name,
            "semantic_index": semantic_index,
            "num_elements": num_elems,
            "element_size": elem_size,
            "size": size,
            "offset": offset
        }
        offset += size
        layout.append(input)
    return layout


# shader visibility can be on a single stage or all
def get_shader_visibility(vis):
    if len(vis) > 1:
        return "all"
    elif len(vis) == 1:
        stages = {
            "vs": "vertex",
            "ps": "fragment",
            "cs": "compute"
        }
        return stages[vis[0]]


# return the binding type from hlsl register names
def get_binding_type(register_type):
    type_lookup = {
        "t": "shader_resource",
        "b": "constant_buffer",
        "u": "unordered_access",
        "s": "sampler"
    }
    return type_lookup[register_type]


# builds a descriptor set from resources used in the pipeline
def parse_descriptor_layout(resources):
    bindable_resources = [
        "ConstantBuffer",
        "StructuredBuffer",
        "Texture1D",
        "Texture2D",
        "Texture3D"
    ]
    descriptor_layout = dict()
    descriptor_layout["bindings"] = list()
    for r in resources:
        resource = resources[r]
        if resource["type"] in bindable_resources:
            binding = {
                "shader_register": resource["shader_register"],
                "register_space": resource["register_space"],
                "binding_type": get_binding_type(resource["register_type"]),
                "visibility": get_shader_visibility(resource["visibility"])
            }
            descriptor_layout["bindings"].append(binding)
    return descriptor_layout


# compile a hlsl version 2
def compile_shader_hlsl(src):
    print(src)


# new generation of pmfx
def generate_pmfx(file, root):
    file_and_path = os.path.join(root, file)
    shader_file_text_full, included_files = build_pmfx.create_shader_set(file_and_path, root)
    pmfx_json, shader_source = build_pmfx.find_pmfx_json(shader_file_text_full)

    # pmfx dictionary
    pmfx = dict()
    pmfx["pmfx"] = pmfx_json
    pmfx["source"] = cgu.format_source(shader_source, 4)

    # create build folders
    info = build_pmfx.get_info()
    name = os.path.splitext(file)[0]
    temp_path = os.path.join(info.temp_dir, name)
    output_path = os.path.join(info.output_dir, name)
    os.makedirs(temp_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    print(temp_path)
    
    # functions
    pmfx["functions"] = dict()
    functions, function_names = cgu.find_functions(pmfx["source"])
    for function in functions:
        if function["name"] != "register":
            pmfx["functions"][function["name"]] = function

    # type mappings
    mapping = [
        {"category": "structs", "identifier": "struct"},
        {"category": "cbuffers", "identifier": "cbuffer"},
        {"category": "cbuffers", "identifier": "ConstantBuffer"},
        {"category": "samplers", "identifier": "SamplerState"},
        {"category": "structured_buffers", "identifier": "StructuredBuffer"},
        {"category": "textures", "identifier": "Texture1D"},
        {"category": "textures", "identifier": "Texture2D"},
        {"category": "textures", "identifier": "Texture3D"}
    ]

    # find types
    pmfx["resources"] = dict()
    for map in mapping:
        decls, names = cgu.find_type_declarations(map["identifier"], pmfx["source"])
        if map["category"] not in pmfx["resources"].keys():
            pmfx["resources"][map["category"]] = dict()
        for decl in decls:
            parse_register(decl)
            if decl["name"] in pmfx["resources"][map["category"]]:
                assert(0)
            pmfx["resources"][map["category"]][decl["name"]] = decl

    # for each pipeline generate code and track used resources
    shader_stages = [
        "vs",
        "ps",
        "cs"
    ]

    resource_categories = [
        "structs",
        "cbuffers",
        "structured_buffers",
        "textures"
    ]

    if "pipelines" in pmfx["pmfx"]:
        pipelines = pmfx["pmfx"]["pipelines"]
        for pipeline_key in pipelines:
            pipeline = pipelines[pipeline_key]
            print("processing: {}".format(pipeline_key))
            resources = dict()
            vertex_layout = dict()
            for stage in shader_stages:
                if stage in pipeline:
                    print("{}:".format(stage))
                    # grab entry point
                    added_functions = []
                    entry_point = pipeline[stage]
                    src = pmfx["functions"][entry_point]["source"]
                    complete = False
                    added_functions.append(entry_point)
                    # recursively insert used functions
                    while not complete:
                        complete = True
                        for func in pmfx["functions"]:
                            if func not in added_functions:
                                if cgu.find_token(func, src) != -1:
                                    added_functions.append(func)
                                    src = pmfx["functions"][func]["source"] + "\n" + src
                                    complete = False
                                    break
                    # now add used resource src decls
                    res = ""
                    for category in resource_categories:
                        for resource in pmfx["resources"][category]:
                            if cgu.find_token(resource, src) != -1:
                                res += pmfx["resources"][category][resource]["declaration"] + ";\n"
                                resources[resource] = pmfx["resources"][category][resource]
                                if "visibility" not in resources[resource]:
                                    resources[resource]["visibility"] = list()
                                resources[resource]["visibility"].append(stage)
                    # extract vs_input (input layout)
                    if stage == "vs":
                        for input in pmfx["functions"][entry_point]["args"]:
                            t = input["type"]
                            if t in pmfx["resources"]["structs"]:
                                vertex_layout = parse_vertex_layout(pmfx["resources"]["structs"][t])
                    src = cgu.format_source(res + src, 4)
                    # compile shader source
                    compile_shader_hlsl(src)
            # build descriptor set
            descriptor_layout = parse_descriptor_layout(resources)

            pipeline_json = dict()
            pipeline_json["vertex_layout"] = vertex_layout
            pipeline_json["descriptor_layout"] = descriptor_layout

            # print(json.dumps(pipeline_json, indent=4))


# entry
if __name__ == "__main__":
    build_pmfx.main(generate_pmfx)