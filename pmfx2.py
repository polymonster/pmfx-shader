import build_pmfx
import re
import os
import cgu
import json


# log formatted json
def log_json(j):
    print(json.dumps(j, indent=4), flush=True)


# member wise merge 2 dicts, second will overwrite dest
def merge_dicts(dest, second):
    for k, v in second.items():
        if type(v) == dict:
            if k not in dest or type(dest[k]) != dict:
                dest[k] = dict()
            merge_dicts(dest[k], v)
        else:
            dest[k] = v
    return dest


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


# returnnumber of 32 bit values for a member of a push constants cbuffer
def get_num_32bit_values(type):
    lookup = {
        "float": 1, 
        "float2": 2,
        "float3": 3,
        "float4": 4,
        "float2x2": 8,
        "float3x4": 12,
        "float4x3": 12,
        "float4x4": 16,
    }
    return lookup[type]


# returns a vertex format type from the data type
def vertex_format_from_type(type):
    lookup = {
        "float": "R32f",
        "float2": "RG32f",
        "float3": "RGB32f",
        "float4": "RGBA32f"
    }
    if type in lookup:
        return lookup[type]
    assert(0)
    return "Unknown"


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
            "format": vertex_format_from_type(member["data_type"]),
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


# retuns the array size of a descriptor binding -1 can indicate unbounded, which translates to None
def get_descriptor_array_size(resource):
    if "array_size" in resource:
        if resource["array_size"] == -1:
            return None
        else:
            return resource["array_size"]
    return None


# builds a descriptor set from resources used in the pipeline
def generate_descriptor_layout(pmfx, pmfx_pipeline, resources):
    bindable_resources = [
        "cbuffer",
        "ConstantBuffer",
        "StructuredBuffer",
        "RWStructuredBuffer",
        "Texture1D",
        "Texture2D",
        "Texture3D",
        "RWTexture1D",
        "RWTexture2D",
        "RWTexture3D",
        "SamplerState"
    ]
    descriptor_layout = dict()
    descriptor_layout["bindings"] = list()
    descriptor_layout["push_constants"] = list()
    descriptor_layout["static_samplers"] = list()
    for r in resources:
        resource = resources[r]
        resource_type = resource["type"]
        # check if we have flagged resource as push constants
        if "push_constants" in pmfx_pipeline:
            if r in pmfx_pipeline["push_constants"]:
                # work out how many 32 bit values
                num_values = 0
                for member in resource["members"]:
                    num_values += get_num_32bit_values(member["data_type"])
                # todo: fold
                push_constants = {
                    "shader_register": resource["shader_register"],
                    "register_space": resource["register_space"],
                    "binding_type": get_binding_type(resource["register_type"]),
                    "visibility": get_shader_visibility(resource["visibility"]),
                    "num_values": num_values
                }
                descriptor_layout["push_constants"].append(push_constants)
                continue
        if "static_samplers" in pmfx_pipeline:
            if r in pmfx_pipeline["static_samplers"]:
                lookup = pmfx_pipeline["static_samplers"][r]
                 # todo: fold
                static_sampler = {
                    "shader_register": resource["shader_register"],
                    "register_space": resource["register_space"],
                    "visibility": get_shader_visibility(resource["visibility"]),
                    "sampler_info": pmfx["sampler_states"][lookup]
                }
                descriptor_layout["static_samplers"].append(static_sampler)
                continue
        # fall trhough and add as a bindable resource
        if resource_type in bindable_resources:
             # todo: fold
            binding = {
                "shader_register": resource["shader_register"],
                "register_space": resource["register_space"],
                "binding_type": get_binding_type(resource["register_type"]),
                "visibility": get_shader_visibility(resource["visibility"]),
                "num_descriptors": get_descriptor_array_size(resource)
            }
            descriptor_layout["bindings"].append(binding)
    # sort bindings in index order
    sorted_bindings = list()
    for binding in descriptor_layout["bindings"]:
        insert_pos = 0
        for i in range(0, len(sorted_bindings)):
            if binding["shader_register"] < sorted_bindings[i]["shader_register"]:
                insert_pos = i
        sorted_bindings.insert(insert_pos, binding)
    descriptor_layout["bindings"] = sorted_bindings
    return descriptor_layout


# compile a hlsl version 2
def compile_shader_hlsl(info, src, temp_path, output_path, filename, stage, entry_point):
    exe = os.path.join(info.tools_dir, "bin", "dxc", "dxc")
    temp_filepath = os.path.join(temp_path, filename)
    output_filepath = os.path.join(output_path, filename + "c")
    open(temp_filepath, "w+").write(src)
    cmdline = "{} -T {}_{} -E {} -Fo {} {}".format(exe, stage, info.shader_version, entry_point, output_filepath, temp_filepath)
    error_code, error_list, output_list = build_pmfx.call_wait_subprocess(cmdline)
    if error_code:
        for err in error_list:
            print(err, flush=True)
        for out in output_list:
            print(out, flush=True)


# assign default values to all struct members
def state_with_defaults(state_type, state):
    state_defaults = {
        "depth_stencil_states": {
            "depth_enabled": False,
            "depth_write_mask": "None",
            "depth_func": "Always",
            "stencil_enabled": False,
            "stencil_read_mask": 0,
            "stencil_write_mask": 0,
            "front_face": {
                "fail": "Keep",
                "depth_fail": "Keep",
                "pass": "Keep",
                "func": "Always"
            },
            "back_face": {
                "fail": "Keep",
                "depth_fail": "Keep",
                "pass": "Keep",
                "func": "Always"
            }
        },
        "sampler_states": {
            "filter": "Linear",
            "address_u": "Wrap",
            "address_v": "Wrap",
            "address_w": "Wrap",
            "comparison": None,
            "border_colour": None,
            "mip_lod_bias": 0.0,
            "max_aniso": 0,
            "min_lod": -1.0,
            "max_lod": -1.0
        }
    }
    default = dict(state_defaults[state_type])
    return merge_dicts(default, state)


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
        {"category": "structured_buffers", "identifier": "RWStructuredBuffer"},
        {"category": "textures", "identifier": "Texture1D"},
        {"category": "textures", "identifier": "Texture2D"},
        {"category": "textures", "identifier": "Texture3D"},
        {"category": "textures", "identifier": "RWTexture1D"},
        {"category": "textures", "identifier": "RWTexture2D"},
        {"category": "textures", "identifier": "RWTexture3D"},
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

    # process states (just fill the defaults out)
    states = [
        "depth_stencil_states",
        "sampler_states",
    ]

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
        "textures",
        "samplers"
    ]

    output_json = dict()
    output_json["pipelines"] = dict()

    # fill state default parameters
    for state_type in states:
        if state_type in pmfx["pmfx"]:
            category = pmfx["pmfx"][state_type]
            output_json[state_type] = dict()
            for state in category:
                output_json[state_type][state] = state_with_defaults(state_type, category[state])

    # process pipelines
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
                                    # add attributes
                                    src = pmfx["functions"][func]["source"] + "\n" + src
                                    complete = False
                                    break
                    # now add used resource src decls
                    res = ""
                    for category in resource_categories:
                        for r in pmfx["resources"][category]:
                            tokens = [r]
                            resource = pmfx["resources"][category][r]
                            # cbuffers with inline decl need to check for usage per member
                            if category == "cbuffers":
                                for member in resource["members"]:
                                    tokens.append(member["name"])
                            # types with templates need to include structs
                            if resource["template_type"]:
                                template_typeame = resource["template_type"]
                                if template_typeame in pmfx["resources"]["structs"]:
                                    struct_resource = pmfx["resources"]["structs"][template_typeame]
                                    # todo: fold 
                                    res += struct_resource["declaration"] + ";\n"
                                    resources[template_typeame] = struct_resource
                                    if "visibility" not in resources[template_typeame]:
                                        resources[template_typeame]["visibility"] = list()
                                    resources[template_typeame]["visibility"].append(stage)
                            # add 
                            for token in tokens:
                                if cgu.find_token(token, src) != -1:
                                    # todo: fold 
                                    res += resource["declaration"] + ";\n"
                                    resources[r] = resource
                                    if "visibility" not in resources[r]:
                                        resources[r]["visibility"] = list()
                                    resources[r]["visibility"].append(stage)
                                    break
                    # extract vs_input (input layout)
                    if stage == "vs":
                        for input in pmfx["functions"][entry_point]["args"]:
                            t = input["type"]
                            if t in pmfx["resources"]["structs"]:
                                pipeline_json["vertex_layout"] = generate_vertex_layout(pmfx["resources"]["structs"][t])
                    src = cgu.format_source(res, 4) + "\n" + cgu.format_source(src, 4)
                    # compile shader source
                    stage_source_filepath = "{}.{}".format(pipeline_key, stage)
                    pipeline_json[stage] = stage_source_filepath + "c"
                    compile_shader_hlsl(info, src, temp_path, output_path, stage_source_filepath, stage, entry_point)
            # build descriptor set
            pipeline_json["descriptor_layout"] = generate_descriptor_layout(output_json, pipeline, resources)
            # topology
            if "topology" in pipeline:
                pipeline_json["topology"] = pipeline["topology"]
            # store info in dict
            output_json["pipelines"][pipeline_key] = pipeline_json

        # write info per pmfx, containing multiple pipelines
        json_filepath = os.path.join(output_path, "{}.json".format(name))
        open(json_filepath, "w+").write(json.dumps(output_json, indent=4))


# entry
if __name__ == "__main__":
    build_pmfx.main(generate_pmfx, "2.0")

    # todo:
    # compile shaders only once, not for each technique combination
    # include handling
    # expand permutations
    # timestamps
    # proper error handling
    # allow single file compilation (pmbuild... learn it)
    # automate cargo publish
    # cargo doc options
