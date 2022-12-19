import build_pmfx
import re
import os
import cgu
import json

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
def generate_vertex_layout(type_dict):
    offset = 0
    layout = list()
    for member in type_dict["members"]:
        semantic_name, semantic_index = separate_name_index(member["semantic"])
        num_elems, elem_size, size = get_type_size_info(member["data_type"])
        input = {
            "name": member["name"],
            "semantic": semantic_name,
            "index": semantic_index,
            "format": "Unknown",
            "aligned_byte_offset": offset,
            "input_slot": 0,
            "input_slot_class": "PerVertex",
            "step_rate": 0
        }
        offset += size
        layout.append(input)
    return layout


# shader visibility can be on a single stage or all
def get_shader_visibility(vis):
    if len(vis) == 1:
        stages = {
            "vs": "Vertex",
            "ps": "Fragment",
            "cs": "Compute"
        }
        return stages[vis[0]]
    return "All"


# return the binding type from hlsl register names
def get_binding_type(register_type):
    type_lookup = {
        "t": "ShaderResource",
        "b": "ConstantBuffer",
        "u": "UnorderedAccess",
        "s": "Sampler"
    }
    return type_lookup[register_type]


# builds a descriptor set from resources used in the pipeline
def generate_descriptor_layout(resources):
    bindable_resources = [
        "cbuffer",
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
        resource_type = resource["type"]
        if resource_type in bindable_resources:
            binding = {
                "shader_register": resource["shader_register"],
                "register_space": resource["register_space"],
                "binding_type": get_binding_type(resource["register_type"]),
                "visibility": get_shader_visibility(resource["visibility"])
            }
            descriptor_layout["bindings"].append(binding)
    return descriptor_layout


# compile a hlsl version 2
def compile_shader_hlsl(src, temp_path, output_path, filename):
    open(os.path.join(temp_path, filename), "w+").write(src)


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

    output_json = dict()
    output_json["pipelines"] = dict()

    if "pipelines" in pmfx["pmfx"]:
        pipelines = pmfx["pmfx"]["pipelines"]
        for pipeline_key in pipelines:
            pipeline = pipelines[pipeline_key]
            resources = dict()
            pipeline_json = dict()
            for stage in shader_stages:
                if stage in pipeline:
                    print("processing: {}.{}".format(pipeline_key, stage))
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
                            tokens = [
                                resource
                            ]
                            # cbuffers with inline decl need to check for usage per member
                            if pmfx["resources"][category][resource]["type"] == "cbuffer":
                                for member in pmfx["resources"][category][resource]["members"]:
                                    tokens.append(member["name"])
                            for token in tokens:
                                if cgu.find_token(token, src) != -1:
                                    res += pmfx["resources"][category][resource]["declaration"] + ";\n"
                                    resources[resource] = pmfx["resources"][category][resource]
                                    if "visibility" not in resources[resource]:
                                        resources[resource]["visibility"] = list()
                                    resources[resource]["visibility"].append(stage)
                                    break
                    # extract vs_input (input layout)
                    if stage == "vs":
                        for input in pmfx["functions"][entry_point]["args"]:
                            t = input["type"]
                            if t in pmfx["resources"]["structs"]:
                                pipeline_json["vertex_layout"] = generate_vertex_layout(pmfx["resources"]["structs"][t])
                    src = cgu.format_source(res + src, 4)
                    # compile shader source
                    stage_source_filepath = "{}.{}".format(pipeline_key, stage)
                    pipeline_json[stage] = stage_source_filepath
                    compile_shader_hlsl(src, temp_path, output_path, stage_source_filepath)
            # build descriptor set
            pipeline_json["descriptor_layout"] = generate_descriptor_layout(resources)
            # store info in dict
            output_json["pipelines"][pipeline_key] = pipeline_json

        # write info per pmfx, containing multiple pipelines
        json_filepath = os.path.join(output_path, "{}.json".format(name))
        open(json_filepath, "w+").write(json.dumps(output_json, indent=4))


# entry
if __name__ == "__main__":
    build_pmfx.main(generate_pmfx, "2.0")