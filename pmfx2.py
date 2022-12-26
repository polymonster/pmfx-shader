import build_pmfx
import re
import os
import cgu
import json
import hashlib
from multiprocessing.pool import ThreadPool

# return names of supported shader stages
def get_shader_stages():
    return [
        "vs",
        "ps",
        "cs"
    ]


# return the key of state groups, specified in pmfx
def get_states():
    return [
        "depth_stencil_states",
        "sampler_states",
        "blend_states",
        "raster_states"
    ]


# return a lits of bindabled resource keywords 
def get_bindable_resource_keys():
    return [
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


# resource keyword to categroy mapping
def get_resource_mappings():
    return [
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


# return list of resource categories
def get_resource_categories():
    return [
        "structs",
        "cbuffers",
        "structured_buffers",
        "textures",
        "samplers"
    ]


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
        elif type(v) == list and k in dest and type(dest[k]) == list:
            dest[k].extend(v)
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


# builds a descriptor set from resources used in the pipeline
def generate_descriptor_layout(pmfx, pmfx_pipeline, resources):
    bindable_resources = get_bindable_resource_keys()
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
def compile_shader_hlsl(info, src, stage, entry_point, temp_filepath, output_filepath):
    exe = os.path.join(info.tools_dir, "bin", "dxc", "dxc")
    open(temp_filepath, "w+").write(src)
    cmdline = "{} -T {}_{} -E {} -Fo {} {}".format(exe, stage, info.shader_version, entry_point, output_filepath, temp_filepath)
    error_code, error_list, output_list = build_pmfx.call_wait_subprocess(cmdline)
    output = ""
    if error_code:
        for err in error_list:
            output += "  " + err + "\n"
    for out in output_list:
        output += "  " + out + "\n"
    output =output.strip("\n")
    print("  compiling {}\n{}".format(output_filepath, output), flush=True)
    return error_code


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


# add shader resource for the shader stage
def add_used_shader_resource(resource, stage):
    output = dict(resource)
    if "visibility" not in output:
        output["visibility"] = list()
    output["visibility"].append(stage)
    return output


# given an entry point generate src code and resource meta data for the shader
def generate_shader_info(pmfx, entry_point, stage):
    # resource categories
    resource_categories = get_resource_categories()
    # start with entry point src code
    src = pmfx["functions"][entry_point]["source"]
    resources = dict()
    vertex_layout = None
    # recursively insert used functions
    complete = False
    added_functions = [entry_point]
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
                    res += struct_resource["declaration"] + ";\n"
                    resources[template_typeame] = add_used_shader_resource(struct_resource, stage)
            # add resource and append resource src code
            for token in tokens:
                if cgu.find_token(token, src) != -1:
                    res += resource["declaration"] + ";\n"
                    resources[r] = add_used_shader_resource(resource, stage)
                    break
    # extract vs_input (input layout)
    if stage == "vs":
        for input in pmfx["functions"][entry_point]["args"]:
            t = input["type"]
            if t in pmfx["resources"]["structs"]:
                vertex_layout = generate_vertex_layout(pmfx["resources"]["structs"][t])
    # join resource src and src
    src = cgu.format_source(res, 4) + "\n" + cgu.format_source(src, 4)
    return {
        "src": src,
        "src_hash": hashlib.md5(src.encode("utf-8")).hexdigest(),
        "resources": dict(resources),
        "vertex_layout": vertex_layout
    }


# check if we need to compile shaders
def shader_needs_compiling(pmfx, entry_point, hash, output_filepath):
    if not os.path.exists(output_filepath):
        return True
    if entry_point in pmfx["compiled_shaders"]:
        if pmfx["compiled_shaders"][entry_point] != hash:
            return True
        else:
            return False
    else:
        return True


# generate shader and metadata
def generate_shader(build_info, stage, entry_point, pmfx, temp_path, output_path):
    info = generate_shader_info(pmfx, entry_point, stage)
    filename = entry_point + "." + stage
    output_filepath = os.path.join(output_path, filename + "c")
    if shader_needs_compiling(pmfx, entry_point, info["src_hash"], output_filepath):
        temp_filepath = os.path.join(temp_path, filename)
        info["error_code"] = compile_shader_hlsl(build_info, info["src"], stage, entry_point, temp_filepath, output_filepath)
    return (stage, entry_point, info)


# generate pipeline and metadata
def generate_pipeline(pipeline_name, pipeline, output_pmfx, shaders):
    print("generating pipeline: {}".format(pipeline_name))
    resources = dict()
    output_pipeline = dict(pipeline)
    # lookup info from compiled shaders and combine resources
    for stage in get_shader_stages():
        if stage in pipeline:
            entry_point = pipeline[stage]
            output_pipeline[stage] = entry_point + ".{}{}".format(stage, "c")
            shader = shaders[stage][entry_point]
            resources = merge_dicts(resources, shader["resources"])
            if stage == "vs":
                output_pipeline["vertex_layout"] = shader["vertex_layout"]
            output_pipeline["error_code"] = shaders[stage][entry_point]["error_code"]
    # build descriptor set
    output_pipeline["descriptor_layout"] = generate_descriptor_layout(output_pmfx, pipeline, resources)
    # topology
    if "topology" in pipeline:
        output_pipeline["topology"] = pipeline["topology"]
    return (pipeline_name, output_pipeline)


# new generation of pmfx
def generate_pmfx(file, root):
    input_pmfx_filepath = os.path.join(root, file)
    shader_file_text_full, included_files = build_pmfx.create_shader_set(input_pmfx_filepath, root)
    pmfx_json, shader_source = build_pmfx.find_pmfx_json(shader_file_text_full)

    # src (input) pmfx dictionary
    pmfx = dict()
    pmfx["pmfx"] = pmfx_json
    pmfx["source"] = cgu.format_source(shader_source, 4)

    # output dictionary
    output_pmfx = {
        "pipelines": dict()
    }

    # create build folders
    build_info = build_pmfx.get_info()
    name = os.path.splitext(file)[0]
    temp_path = os.path.join(build_info.temp_dir, name)
    output_path = os.path.join(build_info.output_dir, name)
    json_filepath = os.path.join(output_path, "{}.json".format(name))
    os.makedirs(temp_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    # check deps
    out_of_date = True
    included_files.append(build_info.this_file)
    included_files.append(input_pmfx_filepath)
    if os.path.exists(json_filepath):
        out_of_date = False
        last_built = os.path.getmtime(json_filepath)
        for file in included_files:
            filepath = build_pmfx.sanitize_file_path(os.path.join(build_info.root_dir, file))
            mtime = os.path.getmtime(filepath)
            if mtime > last_built:
                out_of_date = True
                break
        # get hashes from compiled shaders
        existing = json.loads(open(json_filepath, "r").read())
        if "compiled_shaders" in existing:
            pmfx["compiled_shaders"] = existing["compiled_shaders"]
    
    # return if file not out of date
    check_deps = False
    if check_deps:
        if not out_of_date:
            print("{}: up-to-date".format(file))
            return
    print("building: {}".format(file))
    
    # parse functions
    pmfx["functions"] = dict()
    functions, _ = cgu.find_functions(pmfx["source"])
    for function in functions:
        if function["name"] != "register":
            pmfx["functions"][function["name"]] = function

    # parse types
    pmfx["resources"] = dict()
    for map in get_resource_mappings():
        decls, names = cgu.find_type_declarations(map["identifier"], pmfx["source"])
        if map["category"] not in pmfx["resources"].keys():
            pmfx["resources"][map["category"]] = dict()
        for decl in decls:
            parse_register(decl)
            pmfx["resources"][map["category"]][decl["name"]] = decl

    # fill state default parameters
    for state_type in get_states():
        if state_type in pmfx["pmfx"]:
            category = pmfx["pmfx"][state_type]
            output_pmfx[state_type] = dict()
            for state in category:
                output_pmfx[state_type][state] = state_with_defaults(state_type, category[state])

    # thread pool for compiling shaders and pipelines
    pool = ThreadPool(processes=16)

    # gather shader list
    shader_list = list()
    if "pipelines" in pmfx["pmfx"]:
        pipelines = pmfx["pmfx"]["pipelines"]
        for pipeline_key in pipelines:
            pipeline = pipelines[pipeline_key]
            for stage in get_shader_stages():
                if stage in pipeline:
                    stage_shader = (stage, pipeline[stage])
                    if stage_shader not in shader_list:
                        shader_list.append(stage_shader)

    # compile individual used entry points async
    compile_jobs = []
    for shader in shader_list:
        compile_jobs.append(
            pool.apply_async(generate_shader, (build_info, shader[0], shader[1], pmfx, temp_path, output_path)))

    # wait for compilation to complete, gather results into shaders
    shaders = dict()
    for stage in get_shader_stages():
        shaders[stage] = dict()
    output_pmfx["compiled_shaders"] = dict()
    for job in compile_jobs:
        (stage, entry_point, info) = job.get()
        shaders[stage][entry_point] = info
        output_pmfx["compiled_shaders"][entry_point] = info["src_hash"]

    # generate pipeline reflection info
    pipeline_jobs = []
    if "pipelines" in pmfx["pmfx"]:
        pipelines = pmfx["pmfx"]["pipelines"]
        for pipeline_key in pipelines:
            pipeline_jobs.append(
                pool.apply_async(generate_pipeline, (pipeline_key, pipelines[pipeline_key], output_pmfx, shaders)))

    # wait on pipeline jobs and gather results
    for job in pipeline_jobs:
        (pipeline_name, pipeline_info) = job.get()
        output_pmfx[pipeline_name] = pipeline_info
    
    # write info per pmfx, containing multiple pipelines
    open(json_filepath, "w+").write(json.dumps(output_pmfx, indent=4))

    # return errors
    for pipeline in output_pmfx:
        if "error_code" in output_pmfx[pipeline]:
            return output_pmfx[pipeline]["error_code"]

    return 0


# entry
if __name__ == "__main__":
    build_pmfx.main(generate_pmfx, "2.0")

    # todo:
    # x include handling
    # x multi-threading

    # - proper error handling

    # x timestamps
    # x hashes

    # - blend state
    # - raster state
    # - vertex buffer override
    # - vertex step rate
    
    # - expand permutations

    # - automate cargo publish
    # x cargo doc options
