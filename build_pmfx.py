import os
import sys
import json
import jsn
import re
import math
import subprocess
import platform
import copy
import threading
import cgu
import hashlib


# paths and info for current build environment
class BuildInfo:
    shader_platform = ""                                                # hlsl, glsl, metal, spir-v, pssl
    shader_sub_platform = ""                                            # gles
    shader_version = "0"                                                # 4_0, 5_0 (hlsl), 330, 420 (glsl), 1.1, 2.0 (metal)
    metal_sdk = ""                                                      # macosx, iphoneos, appletvos
    metal_min_os = ""                                                   # iOS (9.0 - 13.0), macOS (10.11 - 10.15)
    debug = False                                                       # generate shader with debug info
    inputs = []                                                         # array of input files or directories
    extensions = []                                                     # array of shader extension currently for glsl
    root_dir = ""                                                       # cwd dir to run from
    build_config = ""                                                   # json contents of build_config.json
    pmfx_dir = ""                                                       # location of pmfx
    tools_dir = ""                                                      # location of pmtech/tools
    output_dir = ""                                                     # dir to build shader binaries
    struct_dir = ""                                                     # dir to output the shader structs
    temp_dir = ""                                                       # dir to put temp shaders
    this_file = ""                                                      # the file u are reading
    macros_file = ""                                                    # pmfx.h
    platform_macros_file = ""                                           # glsl.h, hlsl.h, metal.h
    macros_source = ""                                                  # source code inside _shader_macros.h
    error_code = 0                                                      # non-zero if any shaders failed to build
    nvn_exe = ""                                                        # optional executable path for nvn
    cmdline_string = ""                                                 # stores the full cmdline passed


# info and contents of a .pmfx file
class PmfxInfo:
    includes = ""                                                       # list of included files
    json = ""                                                           # json object containing techniques
    json_text = ""                                                      # json as text to reload mutable dictionary
    source = ""                                                         # source code of the entire shader +includes


# info of pmfx technique permutation which is a combination of vs, ps or cs
class TechniquePermutationInfo:
    pmfx_name = ""                                                      # name of the .pmfx shader containing technique
    technique_name = ""                                                 # name of technique
    technique = ""                                                      # technique / permutation json
    permutation = ""                                                    # permutation options
    shader_version = "0"                                                # shader version to compile with
    source = ""                                                         # conditioned source code for permute
    id = ""                                                             # permutation id
    cbuffers = []                                                       # list of cbuffers source code
    functions = []                                                      # list of functions source code
    textures = []                                                       # technique / permutation textures
    shader = []                                                         # list of shaders, vs, ps or cs
    resource_decl = []                                                  # list of shader resources (textures / buffers)
    threads = []                                                        # number of compute threads, x, y, z
    error_code = 0                                                      # return value from compilation
    error_list = []                                                     # list of errors / warnings from compilation
    output_list = []                                                    # list of output from compilation


# info about a single vs, ps, or cs
class SingleShaderInfo:
    shader_type = ""                                                    # ie. vs (vertex), ps (pixel), cs (compute)
    main_func_name = ""                                                 # entry point ie. vs_main
    functions_source = ""                                               # source code of all used functions
    main_func_source = ""                                               # source code of main function
    input_struct_name = ""                                              # name of input to shader ie. vs_input
    instance_input_struct_name = ""                                     # name of instance input to vertex shader
    output_struct_name = ""                                             # name of output from shader ie. vs_output
    input_decl = ""                                                     # struct decl of input struct
    instance_input_decl = ""                                            # struct decl of instance input struct
    output_decl = ""                                                    # struct decl of shader output
    struct_decls = ""                                                   # decls of all generic structs
    resource_decl = []                                                  # decl of only used resources by shader
    cbuffers = []                                                       # array of cbuffer decls used by shader
    sv_semantics = []                                                   # array of tuple [(semantic, variable name), ..]
    duplicate = False


# used for eval to allow undefined variables
class Reflector(object):
    def __getitem__(self, name):
        return 0


# parse command line args passed in
def parse_args():
    global _info
    # set defaults
    _info.compiled = True
    _info.cbuffer_offset = 4
    _info.texture_offset = 32
    _info.stage_in = 1
    _info.v_flip = False
    _info.debug = False
    if len(sys.argv) == 1:
        display_help()
    for arg in sys.argv:
        _info.cmdline_string += arg + " "
    for i in range(1, len(sys.argv)):
        if "-help" in sys.argv[i]:
            display_help()
        if "-root_dir" in sys.argv[i]:
            os.chdir(sys.argv[i + 1])
        if "-shader_platform" in sys.argv[i]:
            _info.shader_platform = sys.argv[i + 1]
        if "-shader_version" in sys.argv[i]:
            _info.shader_version = sys.argv[i + 1]
        if sys.argv[i] == "-i":
            j = i + 1
            while j < len(sys.argv) and sys.argv[j][0] != '-':
                _info.inputs.append(sys.argv[j])
                j = j + 1
            i = j
        if sys.argv[i] == "-o":
            _info.output_dir = sys.argv[i + 1]
        if sys.argv[i] == "-h":
            _info.struct_dir = sys.argv[i + 1]
        if sys.argv[i] == "-t":
            _info.temp_dir = sys.argv[i + 1]
            pass
        if sys.argv[i] == "-source":
            _info.compiled = False
        if sys.argv[i] == "-cbuffer_offset":
            _info.cbuffer_offset = sys.argv[i + 1]
        if sys.argv[i] == "-texture_offset":
            _info.cbuffer_offset = sys.argv[i + 1]
        if sys.argv[i] == "-stage_in":
            _info.stage_in = sys.argv[i + 1]
        if sys.argv[i] == "-v_flip":
            _info.v_flip = True
        if sys.argv[i] == "-d":
            _info.debug = False
        if sys.argv[i] == "-metal_min_os":
            _info.metal_min_os = sys.argv[i+1]
        if sys.argv[i] == "-metal_sdk":
            _info.metal_sdk = sys.argv[i+1]
        if sys.argv[i] == "-nvn_exe":
            _info.nvn_exe = sys.argv[i+1]
        if sys.argv[i] == "-extensions":
            j = i + 1
            while j < len(sys.argv) and sys.argv[j][0] != '-':
                _info.extensions.append(sys.argv[j])
                j = j + 1
            i = j


# display help for args
def display_help():
    print("commandline arguments:")
    print("    -shader_platform <hlsl, glsl, gles, spirv, metal, pssl, nvn>")
    print("    -shader_version (optional) <shader version unless overridden in technique>")
    print("        hlsl: 3_0, 4_0 (default), 5_0")
    print("        glsl: 200, 330 (default), 420, 450")
    print("        gles: 100, 300, 310, 350")
    print("        spirv: 420 (default), 450")
    print("        metal: 2.0 (default)")
    print("        nvn: (glsl)")
    print("    -metal_sdk [metal only] <iphoneos, macosx, appletvos>")
    print("    -metal_min_os (optional) <9.0 - 13.0 (ios), 10.11 - 10.15 (macos)>")
    print("    -nvn_exe [nvn only] <path to execulatble that can compile glsl to nvn glslc>")
    print("    -extensions <list of glsl extension strings separated by spaces>")
    print("    -i <list of input files or directories separated by spaces>")
    print("    -o <output dir for shaders>")
    print("    -t <output dir for temp files>")
    print("    -h <output dir header file with shader structs>")
    print("    -d (optional) generate debuggable shader")
    print("    -root_dir <directory> sets working directory here")
    print("    -source (optional) (generates platform source into -o no compilation)")
    print("    -stage_in <0, 1> (optional) [metal only] (default 1) ")
    print("        uses stage_in for metal vertex buffers, 0 uses raw buffers")
    print("    -cbuffer_offset (optional) [metal only] (default 4) ")
    print("        specifies an offset applied to cbuffer locations to avoid collisions with vertex buffers")
    print("    -texture_offset (optional) [vulkan only] (default 32) ")
    print("        specifies an offset applied to texture locations to avoid collisions with buffers")
    print("    -v_flip (optional) (inserts glsl uniform to conditionally flip verts in the y axis)") 
    sys.stdout.flush()
    sys.exit(0)


# duplicated from pmtech/tools/scripts/util
def get_platform_name():
    plat = "win64"
    if os.name == "posix":
        plat = "osx"
        if platform.system() == "Linux":
            plat = "linux"
    return plat


# gets shader sub platform name, gles (glsl) spirv (glsl)
def shader_sub_platform():
    sub_platforms = ["gles", "spirv"]
    if _info.shader_sub_platform in sub_platforms:
        return _info.shader_sub_platform
    return _info.shader_platform


# get extension for windows
def get_platform_exe():
    if get_platform_name() == "win64":
        return ".exe"
    return ""


def sanitize_file_path(path):
    path = path.replace("/", os.sep)
    path = path.replace("\\", os.sep)
    path = path.replace("@", ":")
    return path


# duplicated from pmtech/tools/scripts/dependencies
def unstrict_json_safe_filename(file):
    file = file.replace("\\", '/')
    file = file.replace(":", "@")
    return file


def create_dependency(file):
    file = sanitize_file_path(file)
    modified_time = os.path.getmtime(file)
    return {"name": file, "timestamp": float(modified_time)}


# wrap a string in quotes
def in_quotes(string):
    return "\"" + string + "\""


# convert signed to unsigned integer in a c like manner for comparisons
def us(v):
    if v == -1:
        return sys.maxsize
    return v


# calls subprocess, waits and gets output errors
def call_wait_subprocess(cmdline):
    exclude_output = [
        "Microsoft (R)",
        "Copyright (C)",
        "compilation object save succeeded;"
    ]
    p = subprocess.Popen(cmdline, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    error_code = p.wait()
    output, err = p.communicate()
    err_str = err.decode('utf-8')
    err_str = err_str.strip(" ")
    err_list = err_str.split("\n")
    out_str = output.decode('utf-8')
    out_str = out_str.strip(" ")
    out_list = out_str.split("\n")

    clean_err = []
    for e in err_list:
        if len(e) > 0:
            clean_err.append(e.strip())

    clean_out = []
    for o in out_list:
        o = o.strip()
        exclude = False
        for ex in exclude_output:
            if o.startswith(ex):
                exclude = True
                break
        if len(o) > 0 and not exclude:
            clean_out.append(o)

    return error_code, clean_err, clean_out


# recursively merge members of 2 json objects
def member_wise_merge(j1, j2):
    for key in j2.keys():
        if key not in j1.keys():
            j1[key] = j2[key]
        elif type(j1[key]) is dict:
            j1[key] = member_wise_merge(j1[key], j2[key])
    return j1


# remove comments, taken from stub_format.py ()
def remove_comments(file_data):
    lines = file_data.split("\n")
    inside_block = False
    conditioned = ""
    for line in lines:
        if inside_block:
            ecpos = line.find("*/")
            if ecpos != -1:
                inside_block = False
                line = line[ecpos+2:]
            else:
                continue
        cpos = line.find("//")
        mcpos = line.find("/*")
        if cpos != -1:
            conditioned += line[:cpos] + "\n"
        elif mcpos != -1:
            conditioned += line[:mcpos] + "\n"
            inside_block = True
        else:
            conditioned += line + "\n"
    return conditioned


# tidy shader source with consistent spaces, remove tabs and comments to make subsequent operations easier
def sanitize_shader_source(shader_source):
    # replace tabs with spaces
    shader_source = shader_source.replace("\t", " ")
    # replace all spaces with single space
    shader_source = re.sub(' +', ' ', shader_source)
    # remove comments
    shader_source = remove_comments(shader_source)
    return shader_source


# parse and split into an array, from a list of textures or cbuffers etc
def parse_and_split_block(code_block):
    start = code_block.find("{") + 1
    end = code_block.find("};")
    block_conditioned = code_block[start:end].replace(";", " ")
    block_conditioned = block_conditioned.replace(":", " ")
    block_conditioned = block_conditioned.replace("(", " ")
    block_conditioned = block_conditioned.replace(")", " ")
    block_conditioned = block_conditioned.replace(",", " ")
    block_conditioned = re.sub(' +', ' ', block_conditioned)
    return block_conditioned.split()


# find the end of a body text enclosed in brackets
def enclose_brackets(text):
    body_pos = text.find("{")
    bracket_stack = ["{"]
    text_len = len(text)
    while len(bracket_stack) > 0 and body_pos < text_len:
        body_pos += 1
        character = text[body_pos:body_pos+1]
        if character == "{":
            bracket_stack.insert(0, "{")
        if character == "}" and bracket_stack[0] == "{":
            bracket_stack.pop(0)
            body_pos += 1
    return body_pos


# replace all "input" and "output" tokens to "_input" and "_ouput" to avoid glsl keywords
# todo: this should be replaced with "replace_token"
def replace_io_tokens(text):
    token_io = ["input", "output"]
    token_io_replace = ["_input", "_output"]
    token_post_delimiters = ['.', ';', ' ', '(', ')', ',', '-', '+', '*', '/']
    token_pre_delimiters = [' ', '\t', '\n', '(', ')', ',', '-', '+', '*', '/']
    split = text.split(' ')
    split_replace = []
    for token in split:
        for i in range(0, len(token_io)):
            if token_io[i] in token:
                last_char = len(token_io[i])
                first_char = token.find(token_io[i])
                t = token[first_char:first_char+last_char+1]
                l = len(t)
                if first_char > 0 and token[first_char-1] not in token_pre_delimiters:
                    continue
                if l > last_char:
                    c = t[last_char]
                    if c in token_post_delimiters:
                        token = token.replace(token_io[i], token_io_replace[i])
                        continue
                elif l == last_char:
                    token = token.replace(token_io[i], token_io_replace[i])
                    continue
        split_replace.append(token)
    replaced_text = ""
    for token in split_replace:
        replaced_text += token + " "
    return replaced_text


# get info filename for dependency checking
def get_resource_info_filename(filename, build_dir):
    global _info
    base_filename = os.path.basename(filename)
    dir_path = os.path.dirname(filename)
    info_filename = os.path.join(_info.output_dir, os.path.splitext(base_filename)[0], "info.json")
    return info_filename, base_filename, dir_path


# check file time stamps and build times to determine if rebuild needs to happen
# returns true if the file does not need re-building, false if a file/dependency is out of date or input has changed
def check_dependencies(filename, included_files):
    global _info
    # look for .json file
    file_list = list()
    file_list.append(sanitize_file_path(os.path.join(_info.root_dir, filename)))
    file_list.append(sanitize_file_path(_info.this_file))
    file_list.append(sanitize_file_path(_info.macros_file))
    file_list.append(sanitize_file_path(_info.platform_macros_file))
    info_filename, base_filename, dir_path = get_resource_info_filename(filename, _info.output_dir)
    for f in included_files:
        file_list.append(sanitize_file_path(os.path.join(_info.root_dir, f)))
    if os.path.exists(info_filename) and os.path.getsize(info_filename) > 0:
        info_file = open(info_filename, "r")
        info = json.loads(info_file.read())
        if "cmdline" not in info or _info.cmdline_string != info["cmdline"]:
            return False
        for prev_built_with_file in info["files"]:
            sanitized_name = sanitize_file_path(prev_built_with_file["name"])
            if sanitized_name in file_list:
                if not os.path.exists(sanitized_name):
                    return False
                if prev_built_with_file["timestamp"] < os.path.getmtime(sanitized_name):
                    info_file.close()
                    print(os.path.basename(sanitized_name) + " is out of date", flush=True)
                    return False
            else:
                print(sanitized_name + " is not in list", flush=True)
                return False
        if "failures" in info.keys():
            if len(info["failures"]) > 0:
                return False
        info_file.close()
    else:
        return False
    return True


# find generic structs
def find_structs(shader_text, special_structs):
    struct_list = []
    start = 0
    while start != -1:
        op = start
        start = find_token("struct", shader_text[start:])
        if start == -1:
            break
        start = op + start
        end = shader_text.find("};", start)
        if end != -1:
            end += 2
            found_struct = shader_text[start:end]
            valid = True
            for ss in special_structs:
                if ss in found_struct:
                    valid = False
            if valid:
                struct_list.append(shader_text[start:end] + "\n")
        start = end
    return struct_list


def find_c_structs(shader_text):
    special_structs = ["vs_output", "ps_input", "ps_output"]
    return find_structs(shader_text, special_structs)


def find_struct_declarations(shader_text):
    special_structs = ["vs_input", "vs_output", "ps_input", "ps_output", "vs_instance_input"]
    return find_structs(shader_text, special_structs)


# find shader resources
def find_shader_resources(shader_text):
    start = shader_text.find("declare_texture_samplers")
    if start == -1:
        start = shader_text.find("shader_resources")
        if start == -1:
            return "\n"
    start = shader_text.find("{", start) + 1
    end = shader_text.find("};", start)
    texture_sampler_text = shader_text[start:end] + "\n"
    texture_sampler_text = texture_sampler_text.replace("\t", "")
    texture_sampler_text += "\n"
    return texture_sampler_text


# find struct in shader source
def find_struct(shader_text, decl):
    delimiters = [" ", "\n", "{"]
    start = 0
    while True:
        start = shader_text.find(decl, start)
        if start == -1:
            return ""
        for d in delimiters:
            if shader_text[start+len(decl)] == d:
                end = shader_text.find("};", start)
                end += 2
                if start != -1 and end != -1:
                    return shader_text[start:end] + "\n\n"
                else:
                    return ""
        start += len(decl)


# find cbuffers in source
def find_constant_buffers(shader_text):
    cbuffer_list = []
    start = 0
    while start != -1:
        start = shader_text.find("cbuffer", start)
        if start == -1:
            break
        end = shader_text.find("};", start)
        if end != -1:
            end += 2
            cbuffer_list.append(shader_text[start:end] + "\n")
        start = end
    return cbuffer_list


# find function source
def find_function(shader_text, decl):
    start = shader_text.find(decl)
    if start == -1:
        return ""
    body_pos = shader_text.find("{", start)
    bracket_stack = ["{"]
    text_len = len(shader_text)
    while len(bracket_stack) > 0 and body_pos < text_len:
        body_pos += 1
        character = shader_text[body_pos:body_pos+1]
        if character == "{":
            bracket_stack.insert(0, "{")
        if character == "}" and bracket_stack[0] == "{":
            bracket_stack.pop(0)
            body_pos += 1
    return shader_text[start:body_pos] + "\n\n"


# find functions in source
def find_functions(shader_text):
    deliminator_list = [";", "\n"]
    function_list = []
    start = 0
    while 1:
        start = shader_text.find("(", start)
        if start == -1:
            break
        # make sure the { opens before any other deliminator
        deliminator_pos = shader_text.find(";", start)
        body_pos = shader_text.find("{", start)
        if deliminator_pos < body_pos:
            start = deliminator_pos
            continue
        # find the function name and return type
        function_name = shader_text.rfind(" ", 0, start)
        name_str = shader_text[function_name:start]
        if name_str.find("if:") != -1:
            start = deliminator_pos
            continue
        function_return_type = 0
        for delim in deliminator_list:
            decl_start = shader_text.rfind(delim, 0, function_name)
            if decl_start != -1:
                function_return_type = decl_start
        bracket_stack = ["{"]
        text_len = len(shader_text)
        while len(bracket_stack) > 0 and body_pos < text_len:
            body_pos += 1
            character = shader_text[body_pos:body_pos+1]
            if character == "{":
                bracket_stack.insert(0, "{")
            if character == "}" and bracket_stack[0] == "{":
                bracket_stack.pop(0)
                body_pos += 1
        function_list.append(shader_text[function_return_type:body_pos] + "\n\n")
        start = body_pos
    return function_list


# find #include statements
def find_includes(file_text, root):
    global added_includes
    include_list = []
    start = 0
    while 1:
        start = file_text.find("#include", start)
        if start == -1:
            break
        start = file_text.find("\"", start) + 1
        end = file_text.find("\"", start)
        if start == -1 or end == -1:
            break
        include_name = file_text[start:end]
        include_path = os.path.join(root, include_name)
        include_path = sanitize_file_path(include_path)
        if include_path not in added_includes:
            include_list.append(include_path)
            added_includes.append(include_path)
    return include_list


# recursively search for #includes
def add_files_recursive(filename, root):
    file_path = filename
    if not os.path.exists(filename):
        file_path = os.path.join(root, filename)
    included_file = open(file_path, "r")
    shader_source = included_file.read()
    included_file.close()
    shader_source = sanitize_shader_source(shader_source)
    sub_root = os.path.dirname(file_path)
    include_list = find_includes(shader_source, sub_root)
    for include_file in reversed(include_list):
        included_source, sub_includes = add_files_recursive(include_file, sub_root)
        shader_source = included_source + "\n" + shader_source
        include_list = include_list + sub_includes
    return shader_source, include_list


# gather include files and
def create_shader_set(filename, root):
    global _info
    global added_includes
    added_includes = []
    shader_file_text, included_files = add_files_recursive(filename, root)
    shader_base_name = os.path.basename(filename)
    shader_set_dir = os.path.splitext(shader_base_name)[0]
    shader_set_build_dir = os.path.join(_info.output_dir, shader_set_dir)
    if not os.path.exists(shader_set_build_dir):
        os.makedirs(shader_set_build_dir)
    return shader_file_text, included_files


# gets constants only for this current permutation
def get_permutation_conditionals(pmfx_json, permutation):
    block = pmfx_json.copy()
    if "constants" in block:
        # find conditionals
        conditionals = []
        cblock = block["constants"]
        for key in cblock.keys():
            if key.find("permutation(") != -1:
                conditionals.append((key, cblock[key]))
        # check conditionals valid
        for c in conditionals:
            # remove conditional permutation
            del block["constants"][c[0]]
            full_condition = c[0].replace("permutation", "")
            full_condition = full_condition.replace("&&", "and")
            full_condition = full_condition.replace("||", "or")
            gv = dict()
            for v in permutation:
                gv[str(v[0])] = v[1]
            try:
                if eval(full_condition, gv):
                    block["constants"] = member_wise_merge(block["constants"], c[1])
            except NameError:
                pass
    return block


# get list of technique / permutation specific
def generate_technique_texture_variables(_tp):
    technique_textures = []
    if "texture_samplers" not in _tp.technique.keys():
        return
    textures = _tp.technique["texture_samplers"]
    for t in textures.keys():
        technique_textures.append((textures[t]["type"], t, textures[t]["unit"]))
    return technique_textures


# generate cbuffer meta data, c structs for access in code
def generate_technique_constant_buffers(pmfx_json, _tp):
    offset = 0
    constant_info = [["", 0], ["float", 1], ["float2", 2], ["float3", 3], ["float4", 4], ["float4x4", 16]]

    technique_constants = [_tp.technique]
    technique_json = _tp.technique

    # find inherited constants
    if "inherit_constants" in _tp.technique.keys():
        for inherit in _tp.technique["inherit_constants"]:
            inherit_conditionals = get_permutation_conditionals(pmfx_json[inherit], _tp.permutation)
            technique_constants.append(inherit_conditionals)

    # find all constants
    shader_constant = []
    shader_struct = []
    pmfx_constants = dict()

    for tc in technique_constants:
        if "constants" in tc.keys():
            # sort constants
            sorted_constants = []
            for const in tc["constants"]:
                for ci in constant_info:
                    if ci[0] == tc["constants"][const]["type"]:
                        cc = [const, ci[1]]
                        pos = 0
                        for sc in sorted_constants:
                            if cc[1] > sc[1]:
                                sorted_constants.insert(pos, cc)
                                break
                            pos += 1
                        if pos >= len(sorted_constants):
                            sorted_constants.append(cc)
            for const in sorted_constants:
                const_name = const[0]
                const_elems = const[1]
                pmfx_constants[const_name] = tc["constants"][const_name]
                pmfx_constants[const_name]["offset"] = offset
                pmfx_constants[const_name]["num_elements"] = const_elems
                shader_constant.append("    " + tc["constants"][const_name]["type"] + " " + "m_" + const_name + ";\n")
                shader_struct.append("    " + tc["constants"][const_name]["type"] + " " + "m_" + const_name + ";\n")
                offset += const_elems

    if offset == 0:
        return _tp.technique, "", ""

    # we must pad to 16 bytes alignment
    pre_pad_offset = offset
    diff = offset / 4
    next = math.ceil(diff)
    pad = (next - diff) * 4
    if pad != 0:
        shader_constant.append("    " + constant_info[int(pad)][0] + " " + "m_padding" + ";\n")
        shader_struct.append("    " + constant_info[int(pad)][0] + " " + "m_padding" + ";\n")

    offset += pad

    cb_str = "cbuffer material_data : register(b7)\n"
    cb_str += "{\n"
    for sc in shader_constant:
        cb_str += sc
    cb_str += "};\n"

    # append permutation string to shader c struct
    skips = [
        _info.shader_platform.upper(),
        _info.shader_sub_platform.upper()
    ]
    permutation_name = ""
    if int(_tp.id) != 0:
        for p in _tp.permutation:
            if p[0] in skips or p[0] in caps_list():
                continue
            if p[1] == 1:
                permutation_name += "_" + p[0].lower()
            if p[1] > 1:
                permutation_name += "_" + p[0].lower() + p[1]

    c_struct = "struct " + _tp.technique_name + permutation_name + "\n"
    c_struct += "{\n"
    for ss in shader_struct:
        c_struct += ss
    c_struct += "};\n\n"

    technique_json["constants"] = pmfx_constants
    technique_json["constants_used_bytes"] = int(pre_pad_offset * 4)
    technique_json["constants_size_bytes"] = int(offset * 4)
    assert int(offset * 4) % 16 == 0

    return technique_json, c_struct, cb_str


# removes un-used input structures which may be empty if they have been defined out by permutation.
def strip_empty_inputs(input, main):
    conditioned = input.replace("\n", "").replace(";", "").replace(";", "").replace("}", "").replace("{", "")
    tokens = conditioned.split(" ")
    for t in tokens:
        if t == "":
            tokens.remove(t)
    if len(tokens) == 2:
        # input is empty so remove from vs_main args
        input = ""
        name = tokens[1]
        pos = main.find(name)
        prev_delim = max(us(main[:pos].rfind(",")), us(main[:pos].rfind("(")))
        next_delim = pos + min(us(main[pos:].find(",")), us(main[pos:].find(")")))
        main = main.replace(main[prev_delim:next_delim], " ")
    return input, main


# gets system value semantics (SV_InstanceID) and stores them in a tuple, for platform specific code gen later.
def get_sv_sematics(main):
    supported_sv = ["SV_InstanceID", "SV_VertexID"]
    sig = main[main.find("(")+1:main.find(")")]
    args = sig.split(',')
    sv_semantics = []
    for sv in supported_sv:
        for arg in args:
            if arg.find(sv) != -1:
                arg_split = arg.replace(":", " ").strip().split(" ")
                var_type = arg_split[0].strip()
                var_name = arg_split[1].strip()
                sv_semantics.append((sv, var_type, var_name))
    return sv_semantics


# evaluate permutation / technique defines in if: blocks and remove unused branches
def evaluate_conditional_blocks(source, permutation):
    if not permutation:
        return source
    pos = 0
    case_accepted = False
    while True:
        else_pos = source.find("else:", pos)
        else_if_pos = source.find("else if:", pos)
        pos = source.find("if:", pos)
        else_case = False
        first_case = True

        if us(else_if_pos) < us(pos):
            pos = else_if_pos
            first_case = False

        if us(else_pos) < us(pos):
            pos = else_pos
            else_case = True
            first_case = False

        if first_case:
            case_accepted = False

        if pos == -1:
            break

        if not else_case:
            conditions_start = source.find("(", pos)
            body_start = source.find("{", conditions_start) + 1
            conditions = source[conditions_start:body_start - 1]
            conditions = conditions.replace('\n', '')
            conditions = conditions.replace("&&", " and ")
            conditions = conditions.replace("||", " or ")
            conditions = conditions.replace("!", " not ")
        else:
            body_start = source.find("{", pos) + 1
            conditions = "True"

        gv = dict()
        for v in permutation:
            gv[str(v[0])] = v[1]

        lv = dict()

        conditional_block = ""
        i = body_start
        stack_size = 1
        while True:
            if source[i] == "{":
                stack_size += 1
            if source[i] == "}":
                stack_size -= 1
            if stack_size == 0:
                break
            i += 1

        if not case_accepted:
            while True:
                try:
                    if eval(conditions, gv, lv):
                        conditional_block = source[body_start:i]
                        case_accepted = True
                        break
                    else:
                        break
                except NameError as e:
                    defname = re.search("name '([^\']*)' is not defined", str(e)).group(1)
                    lv[defname] = 0
                    conditional_block = ""
        else:
            conditional_block = ""

        source = source.replace(source[pos:i+1], conditional_block)
        pos += len(conditional_block)

    return source


# recursively generate all possible permutations from inputs
def permute(define_list, permute_list, output_permutations):
    if len(define_list) == 0:
        output_permutations.append(list(permute_list))
    else:
        d = define_list.pop()
        for s in d[1]:
            ds = (d[0], s)
            permute_list.append(ds)
            output_permutations = permute(define_list, permute_list, output_permutations)
            if len(permute_list) > 0:
                permute_list.pop()
        define_list.append(d)
    return output_permutations


# generate numerical id for permutation
def generate_permutation_id(define_list, permutation):
    pid = 0
    for p in permutation:
        for d in define_list:
            if p[0] == d[0]:
                if p[1] > 0:
                    exponent = d[2]
                    if exponent < 0:
                        continue
                    if p[1] > 1:
                        exponent = p[1]+exponent-1
                    pid += pow(2, exponent)
    return pid


# return shader version as float for consistent comparisons, version will be a string
def shader_version_float(platform, version):
    if platform == "metal":
        # metal version is already a float
        return float(version)
    elif platform == "glsl" or platform == "spirv" or platform == "gles":
        # glsl version is integer 330, 400, 450..
        return float(version)
    elif platform == "hlsl":
        # hlsl version is 3_0, 5_0
        return float(version.replace("_", "."))
    assert 0


# just list of all the caps
def caps_list():
    return [
        "PMFX_TEXTURE_CUBE_ARRAY",
        "PMFX_COMPUTE_SHADER"
    ]


# based on shader platform and version, some features may or may not be available
def defines_from_caps(define_list):
    global _info
    # platform, feature version
    lookup = {
        "metal": [
            ["PMFX_TEXTURE_CUBE_ARRAY", 0.0],
            ["PMFX_COMPUTE_SHADER", 0.0]
        ],
        "glsl": [
            ["PMFX_TEXTURE_CUBE_ARRAY", 400.0],
            ["PMFX_COMPUTE_SHADER", 450.0]
        ],
        "gles": [
            ["PMFX_TEXTURE_CUBE_ARRAY", 310.0],
            ["PMFX_COMPUTE_SHADER", 310.0]
        ],
        "spirv": [
            ["PMFX_TEXTURE_CUBE_ARRAY", 400.0],
            ["PMFX_COMPUTE_SHADER", 450.0]
        ],
        "hlsl": [
            ["PMFX_TEXTURE_CUBE_ARRAY", 4.0],
            ["PMFX_COMPUTE_SHADER", 5.0]
        ]
    }
    # check platform exists
    platform = shader_sub_platform()
    if platform not in lookup.keys():
        return []
    # add features
    version = shader_version_float(platform, _info.shader_version)
    define_list = []
    for cap in lookup[platform]:
        if version >= cap[1]:
            define_list.append((cap[0], [1], -1))
    return define_list


# generate permutation list from technique json
def generate_permutations(technique, technique_json):
    global _info
    output_permutations = []
    define_list = []
    permutation_options = dict()
    permutation_option_mask = 0
    define_string = ""

    define_list.append((_info.shader_platform.upper(), [1], -1))
    define_list.append((_info.shader_sub_platform.upper(), [1], -1))
    define_list = defines_from_caps(define_list)
    if "permutations" in technique_json:
        for p in technique_json["permutations"].keys():
            pp = technique_json["permutations"][p]
            define_list.append((p, pp[1], pp[0]))
        if "defines" in technique_json.keys():
            for d in technique_json["defines"]:
                define_list.append((d, [1], -1))
        output_permutations = permute(define_list, [], [])
        for key in technique_json["permutations"]:
            tp = technique_json["permutations"][key]
            ptype = "checkbox"
            if len(tp[1]) > 2:
                ptype = "input_int"
            permutation_options[key] = {"val": pow(2, tp[0]), "type": ptype}
            mask = pow(2, tp[0])
            permutation_option_mask += mask
            define_string += "#define " + technique.upper() + "_" + key + " " + str(mask) + "\n"
        define_string += "\n"

    # generate default permutation, inherit / get permutation constants
    tp = list(output_permutations)
    if len(tp) == 0:
        default_permute = []
        if "defines" in technique_json.keys():
            for d in technique_json["defines"]:
                default_permute.append((d, 1))
        else:
            default_permute = [("SINGLE_PERMUTATION", 1)]
        tp.append(default_permute)

    return tp, permutation_options, permutation_option_mask, define_list, define_string


# look for inherit member and inherit another pmfx technique
def inherit_technique(technique, pmfx_json):
    if "inherit" in technique.keys():
        inherit = technique["inherit"]
        if inherit in pmfx_json.keys():
            technique = member_wise_merge(technique, pmfx_json[inherit])
    return technique


# parse pmfx file to find the json block pmfx: { }
def find_pmfx_json(shader_file_text):
    pmfx_loc = shader_file_text.find("pmfx:")
    if pmfx_loc != -1:
        # pmfx json exists, return the block
        json_loc = shader_file_text.find("{", pmfx_loc)
        pmfx_end = enclose_brackets(shader_file_text[pmfx_loc:])
        pmfx_json = jsn.loads(shader_file_text[json_loc:pmfx_end + json_loc])
        return pmfx_json
    else:
        # shader can have no pmfx, provided it supplies vs_main and ps_main
        if find_function(shader_file_text, "vs_main") and find_function(shader_file_text, "ps_main"):
            pmfx_json = dict()
            pmfx_json["default"] = {"vs": "vs_main", "ps": "ps_main"}
            return pmfx_json
    return None


# find only used shader resources
def find_used_resources(shader_source, resource_decl):
    if not resource_decl:
        return
    # find resource uses
    uses = ["sample_texture", "read_texture", "write_texture", "sample_depth"]
    resource_uses = []
    pos = 0
    while True:
        sampler, tok = cgu.find_first(shader_source, uses, pos)
        if sampler == sys.maxsize:
            break;
        start = shader_source.find("(", sampler)
        end = shader_source.find(";", sampler)
        if us(sampler) < us(start) < us(end):
            args = shader_source[start+1:end-1].split(",")
            if len(args) > 0:
                name = args[0].strip(" ")
                if name not in resource_uses:
                    resource_uses.append(name)
        pos = end
    used_resource_decl = ""
    resource_list = resource_decl.split(";")
    for resource in resource_list:
        start = resource.find("(") + 1
        end = resource.find(")") - 1
        args = resource[start:end].split(",")
        name_positions = [0, 2]  # 0 = single sample texture, 2 = msaa texture
        # texture or msaa texture sampled with sample_texture...
        for p in name_positions:
            if len(args) > p:
                name = args[p].strip(" ")
                if name in resource_uses:
                    used_resource_decl = used_resource_decl.strip(" ")
                    used_resource_decl += resource + ";\n"
        # structured buffer with [] operator access
        if resource_decl.find("structured_buffer") != -1:
            if len(args) > 1:
                name = args[1].strip(" ")
                if shader_source.find(name + "[") != -1:
                    used_resource_decl = used_resource_decl.strip(" ")
                    used_resource_decl += resource + ";\n"
    return used_resource_decl


# find only used cbuffers
def find_used_cbuffers(shader_source, cbuffers):
    # turn source to tokens
    non_tokens = ["(", ")", "{", "}", ".", ",", "+", "-", "=", "*", "/", "&", "|", "~", "\n", "<", ">", "[", "]", ";"]
    token_source = shader_source
    for nt in non_tokens:
        token_source = token_source.replace(nt, " ")
    token_list = token_source.split(" ")
    used_cbuffers = []
    for cbuf in cbuffers:
        member_list = parse_and_split_block(cbuf)
        for i in range(1, len(member_list), 2):
            member = member_list[i].strip()
            array = member.find("[")
            if array != -1:
                if array == 0:
                    i += 1
                    continue
                else:
                    member = member[:array]
            if member in token_list:
                used_cbuffers.append(cbuf)
                break
    return used_cbuffers


# find only used functions from a given entry point
def find_used_functions(entry_func, function_list):
    used_functions = [entry_func]
    added_function_names = []
    ordered_function_list = [entry_func]
    for used_func in used_functions:
        for func in function_list:
            if func == used_func:
                continue
            name = func.split(" ")[1]
            end = name.find("(")
            name = name[0:end]
            if used_func.find(name + "(") != -1:
                if name in added_function_names:
                    continue
                used_functions.append(func)
                added_function_names.append(name)
    for func in function_list:
        name = func.split(" ")[1]
        end = name.find("(")
        name = name[0:end]
        if name in added_function_names:
            ordered_function_list.append(func)
    ordered_function_list.remove(entry_func)
    used_function_source = ""
    for used_func in ordered_function_list:
        used_function_source += used_func + "\n\n"
    return used_function_source


# generate a vs, ps or cs from _tp (technique permutation data)
def generate_single_shader(main_func, _tp):
    _si = SingleShaderInfo()
    _si.main_func_name = main_func

    # find main func
    main = ""
    for func in _tp.functions:
        pos = func.find(main_func)
        if pos != -1:
            if func[pos+len(main_func)] == "(" and func[pos-1] == " ":
                main = func

    if main == "":
        print("error: could not find main function " + main_func, flush=True)
        return None

    # find used functions,
    _si.functions_source = find_used_functions(main, _tp.functions)

    # find inputs / outputs
    _si.instance_input_struct_name = None
    _si.output_struct_name = main[0:main.find(" ")].strip()
    input_signature = main[main.find("(")+1:main.find(")")].split(" ")
    for i in range(0, len(input_signature)):
        input_signature[i] = input_signature[i].replace(",", "")
        if input_signature[i] == "_input" or input_signature[i] == "input":
            _si.input_struct_name = input_signature[i-1]
        elif input_signature[i] == "_instance_input" or input_signature[i] == "instance_input":
            _si.instance_input_struct_name = input_signature[i-1]

    # find source decl for inputs / outputs
    if _si.instance_input_struct_name:
        _si.instance_input_decl = find_struct(_tp.source, "struct " + _si.instance_input_struct_name)
    _si.input_decl = find_struct(_tp.source, "struct " + _si.input_struct_name)
    _si.output_decl = find_struct(_tp.source, "struct " + _si.output_struct_name)

    # remove empty inputs which have no members due to permutation conditionals
    _si.input_decl, main = strip_empty_inputs(_si.input_decl, main)

    # get sv sematics to insert gl / metal specific equivalent
    _si.sv_semantics = get_sv_sematics(main)

    # condition main function with stripped inputs
    if _si.instance_input_struct_name:
        _si.instance_input_decl, main = strip_empty_inputs(_si.instance_input_decl, main)
        if _si.instance_input_decl == "":
            _si.instance_input_struct_name = None
    _si.main_func_source = main

    # find only used textures by this shader
    full_source = _si.functions_source + main
    _si.resource_decl = find_used_resources(full_source, _tp.resource_decl)
    _si.cbuffers = find_used_cbuffers(full_source, _tp.cbuffers)
    _si.threads = _tp.threads

    return _si


# format source with indents
def format_source(source, indent_size):
    formatted = ""
    lines = source.split("\n")
    indent = 0
    indents = ["{"]
    unindnets = ["}"]
    for line in lines:
        cur_indent = indent
        line = line.strip(" ")
        if len(line) < 1:
            continue
        if line[0] in indents:
            indent += 1
        elif line[0] in unindnets:
            indent -= 1
            cur_indent = indent
        for i in range(0, cur_indent*indent_size):
            formatted += " "
        formatted += line
        formatted += "\n"
    return formatted


# hashes a shader to find identical shaders which have different permutation options
def shader_hash(_shader):
    hash_source = ""
    hash_source += _shader.input_decl
    hash_source += _shader.instance_input_decl
    hash_source += _shader.output_decl
    hash_source += _shader.resource_decl
    hash_source += _shader.functions_source
    hash_source += _shader.main_func_source
    for cb in _shader.cbuffers:
        hash_source += cb
    return hashlib.md5(hash_source.encode('utf-8')).hexdigest()


# hlsl source.. pssl is similar
def _hlsl_source(_info, pmfx_name, _tp, _shader):
    shader_source = _info.macros_source
    shader_source += _tp.struct_decls
    for cb in _shader.cbuffers:
        shader_source += cb
    shader_source += _shader.input_decl
    shader_source += _shader.instance_input_decl
    shader_source += _shader.output_decl
    shader_source += _shader.resource_decl
    shader_source += _shader.functions_source
    if _shader.shader_type == "cs":
        shader_source += "[numthreads("
        for i in range(0, 3):
            shader_source += str(_tp.threads[i])
            if i < 2:
                shader_source += ", "
        shader_source += ")]"
    shader_source += _shader.main_func_source
    shader_source = format_source(shader_source, 4)
    return shader_source


# compile pssl
def compile_pssl(_info, pmfx_name, _tp, _shader):
    orbis_sdk = os.getenv("SCE_ORBIS_SDK_DIR")
    if not orbis_sdk:
        print("error: you must have orbis sdk installed, "
              "'SCE_ORBIS_SDK_DIR' environment variable is set and is added to your PATH.", flush=True)
        sys.exit(1)

    shader_source = _hlsl_source(_info, pmfx_name, _tp, _shader)

    # apply syntax changes
    token_swaps = {
        "cbuffer": "ConstantBuffer",
        "SV_POSITION": "S_POSITION",
        "SV_POSITION0": "S_POSITION",
        "SV_Target": "S_TARGET_OUTPUT",
        "SV_Target0": "S_TARGET_OUTPUT0",
        "SV_Target1": "S_TARGET_OUTPUT1",
        "SV_Target2": "S_TARGET_OUTPUT2",
        "SV_Target3": "S_TARGET_OUTPUT3",
        "SV_Target4": "S_TARGET_OUTPUT4",
        "SV_Target5": "S_TARGET_OUTPUT5",
        "SV_Target6": "S_TARGET_OUTPUT6",
        "SV_Target7": "S_TARGET_OUTPUT7",
        "SV_Depth": "S_DEPTH_OUTPUT",
        "SV_InstanceID": "S_INSTANCE_ID",
        "SV_VertexID": "S_VERTEX_ID"
    }

    for token in token_swaps:
        shader_source = replace_token(token, token_swaps[token], shader_source)

    extension = {
        "vs": ".vs",
        "ps": ".ps",
        "cs": ".cs"
    }

    profile = {
        "vs": "sce_vs_vs_orbis",
        "ps": "sce_ps_orbis",
        "cs": "sce_cs_orbis"
    }

    temp_path = os.path.join(_info.temp_dir, pmfx_name)
    output_path = os.path.join(_info.output_dir, pmfx_name)
    os.makedirs(temp_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    temp_file_and_path = sanitize_file_path(os.path.join(temp_path, _tp.name + extension[_shader.shader_type]))
    output_file_and_path = os.path.join(output_path, _tp.name + extension[_shader.shader_type] + "c")

    temp_shader_source = open(temp_file_and_path, "w")
    temp_shader_source.write(shader_source)
    temp_shader_source.close()

    cmdline = "orbis-wave-psslc" + " -profile " + profile[_shader.shader_type] + \
              " -entry " + _shader.main_func_name + " " + temp_file_and_path + " -o " + output_file_and_path

    error_code, error_list, output_list = call_wait_subprocess(cmdline)

    if error_code != 0:
        _tp.error_code = error_code
        _tp.error_list = error_list
        _tp.output_list = output_list


# compile hlsl shader model 4
def compile_hlsl(_info, pmfx_name, _tp, _shader):
    shader_source = _hlsl_source(_info, pmfx_name, _tp, _shader)
    exe = os.path.join(_info.tools_dir, "bin", "fxc", "fxc")

    # default sm 4
    if _tp.shader_version == "0":
        _tp.shader_version = "4_0"

    sm = str(_tp.shader_version)

    shader_model = {
        "vs": "vs_" + sm,
        "ps": "ps_" + sm,
        "cs": "cs_" + sm
    }

    extension = {
        "vs": ".vs",
        "ps": ".ps",
        "cs": ".cs"
    }

    temp_path = os.path.join(_info.temp_dir, pmfx_name)
    output_path = os.path.join(_info.output_dir, pmfx_name)
    os.makedirs(temp_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    temp_file_and_path = os.path.join(temp_path, _tp.name + extension[_shader.shader_type])
    output_file_and_path = os.path.join(output_path, _tp.name + extension[_shader.shader_type] + "c")

    temp_shader_source = open(temp_file_and_path, "w")
    temp_shader_source.write(shader_source)
    temp_shader_source.close()

    cmdline = exe + " "
    cmdline += "/T " + shader_model[_shader.shader_type] + " "
    cmdline += "/E " + _shader.main_func_name + " "
    if _info.debug:
        cmdline += "/Fc /Od /Zi" + " "
    cmdline += "/Fo " + output_file_and_path + " " + temp_file_and_path + " "

    error_code, error_list, output_list = call_wait_subprocess(cmdline)

    if error_code != 0:
        _tp.error_code = error_code
        _tp.error_list = error_list
        _tp.output_list = output_list


# parse shader inputs annd output source into a list of elements and semantics
def parse_io_struct(source):
    if len(source) == 0:
        return [], []
    io_source = source
    start = io_source.find("{")
    end = io_source.find("}")
    elements = []
    semantics = []
    prev_input = start+1
    next_input = 0
    while next_input < end:
        next_input = io_source.find(";", prev_input)
        if next_input > 0:
            next_semantic = io_source.find(":", prev_input)
            elements.append(io_source[prev_input:next_semantic].strip())
            semantics.append(io_source[next_semantic+1:next_input].strip())
            prev_input = next_input + 1
        else:
            break
    # the last input will always be "};" pop it out
    elements.pop(len(elements)-1)
    semantics.pop(len(semantics)-1)
    return elements, semantics


# generate a global struct to access input structures in a hlsl like manner
def generate_global_io_struct(io_elements, decl):
    # global input struct for hlsl compatibility to access like input.value
    struct_source = decl
    struct_source += "\n{\n"
    for element in io_elements:
        struct_source += element + ";\n"
    struct_source += "};\n"
    struct_source += "\n"
    return struct_source


# assign vs or ps inputs to the global struct
def generate_input_assignment(io_elements, decl, local_var, suffix):
    assign_source = "//assign " + decl + " struct from glsl inputs\n"
    assign_source += decl + " " + local_var + ";\n"
    for element in io_elements:
        if element.split()[1] == "position" and "vs_output" in decl:
            continue
        var_name = element.split()[1]
        assign_source += local_var + "." + var_name + " = " + var_name + suffix + ";\n"
    return assign_source


# assign vs or ps outputs from the global struct to the output locations
def generate_output_assignment(_info, io_elements, local_var, suffix, gles2=False):
    assign_source = "\n//assign glsl global outputs from structs\n"
    for element in io_elements:
        var_name = element.split()[1]
        if var_name == "position":
            assign_source += "gl_Position = " + local_var + "." + var_name + ";\n"
            if _info.v_flip:
                assign_source += "gl_Position.y *= v_flip;\n"
            if _info.shader_sub_platform == "spirv":
                assign_source += "gl_Position.y *= -1.0;\n"
        else:
            if gles2:
                if suffix == "_ps_output":
                    assign_source += "gl_FragColor" + " = " + local_var + "." + var_name + ";\n"
                    continue
            assign_source += var_name + suffix + " = " + local_var + "." + var_name + ";\n"
    return assign_source


# generates a texture declaration from a texture list
def generate_texture_decl(texture_list):
    if not texture_list:
        return ""
    texture_decl = ""
    for alias in texture_list:
        decl = str(alias[0]) + "( " + str(alias[1]) + ", " + str(alias[2]) + " );\n"
        texture_decl += decl
    return texture_decl


# insert glsl location if we need it
def insert_layout_location(loc):
    if _info.shader_sub_platform == "spirv":
       return "layout(location = " + str(loc) + ") "
    return ""


# gets structured buffers from resource decls (type, name, binding)
def get_structured_buffers(shader):
    res = shader.resource_decl.split(";")
    sb = []
    for r in res:
        r = r.strip()
        if len(r) == 0:
            continue
        if r.find("structured_buffer") != -1:
            decl = r[r.find("("):].split(",")
            args = []
            for d in decl:
                args.append(d.strip().strip("(").strip(")").strip())
            sb.append(args)
    return sb


# extracts the texture types into dictionary from resource decl to replace sample calls
def texture_types_from_resource_decl(resource_decl):
    tex_dict = dict()
    resource_list = resource_decl.split(";")
    for resource in resource_list:
        start = resource.find("(") + 1
        end = resource.find(")") - 1
        args = resource[start:end].split(",")
        name_positions = [0, 2]  # 0 = single sample texture, 2 = msaa texture
        # texture or msaa texture sampled with sample_texture...
        name = ""
        for p in name_positions:
            if len(args) > p:
                name = args[p].strip(" ")
        tex_type = resource[:start-1]
        if len(name) > 0:
            tex_dict[name] = tex_type.strip()
    return tex_dict


# locates pmfx sample_texture calls and replaces with non-polymorphic function calls
def replace_texture_samples(shader, texture_types_dict):
    sampler_tokens = ["sample_texture", "sample_texture_level", "sample_texture_grad", "sample_texture_array"]
    pos = 0
    while True:
        sample, tok = cgu.find_first_token(shader, sampler_tokens, pos)
        if sample == sys.maxsize:
            break
        name_start = sample + shader[sample:].find("(") + 1
        name_end = name_start+ shader[name_start:].find(",")
        name_str = shader[name_start:name_end].strip()
        if name_str in texture_types_dict:
            tex_type = texture_types_dict[name_str]
            tex_type = tex_type.replace("texture_", "")
            tex_type = tex_type.replace("_array", "")
            insert = shader[:sample+len(tok)] + "_" + tex_type
            insert += shader[sample+len(tok):]
            shader = insert
        end = shader[sample:].find(")")
        pos = sample+end+1
    return shader


# generates gles 2 compatible uniforms packed into glUniformMatrixfv
def generate_uniform_pack(cbuffer_name, cbuffer_body):
    v4_type = {
        "float4": 1,
        "float4x4": 4
    }
    output = dict()
    cbuffer_body = cbuffer_body.strip("{")
    cbuffer_body = cbuffer_body.strip("};").strip()
    cbuffer_name = cbuffer_name.strip()
    members = cbuffer_body.split(";")
    v4_counter = 0
    member_pairs = []
    for member in members:
        member = member.strip()
        if len(member) <= 0:
            continue
        pair = member.split(" ")
        type = pair[0]
        name = pair[1]
        member_pairs.append((type, name))
        if type not in v4_type.keys():
            print("cannot pack type into float4 array: " + type)
            exit(1)
        v4_counter += v4_type[type]
    output["decl"] = "uniform float4 " + cbuffer_name + "[" + str(v4_counter) + "];\n"
    v4_pos = 0
    assign = ""
    for member in member_pairs:
        if member[0] == "float4x4":
            assign += (member[0] + " " + member[1] + ";\n")
            assign += (member[1] + "[0] = " + cbuffer_name + "[" + str(v4_pos) + "];\n")
            assign += (member[1] + "[1] = " + cbuffer_name + "[" + str(v4_pos+1) + "];\n")
            assign += (member[1] + "[2] = " + cbuffer_name + "[" + str(v4_pos+2) + "];\n")
            assign += (member[1] + "[3] = " + cbuffer_name + "[" + str(v4_pos+3) + "];\n")
        else:
            assign += (member[0] + " " + member[1] + " = " + cbuffer_name + "[" + str(v4_pos) + "];\n")
        v4_pos += v4_type[member[0]]
    output["assign"] = assign
    return output


# unpacks a uniform pack into variables of the correct type, this is relying on the optimiser to rip out the reduant assigns
def insert_uniform_unpack_assignment(functions_source, uniform_pack):
    pos = 0
    inserted_source = ""
    while True:
        bp = functions_source[pos:].find("{")
        if bp == -1:
            break
        bp = pos + bp
        ep = enclose_brackets(functions_source[bp:])
        if ep == -1:
            break
        ep = bp + ep
        pos = ep + 1
        inserted_source += functions_source[:bp+1]
        inserted_source += "\n" + uniform_pack["assign"] + "\n"
        inserted_source += functions_source[bp+1:ep]
    return inserted_source

# replace token pasting, since gles does not support it by default
def replace_token_pasting(shader):
    tokens = ["structured_buffer", "structured_buffer_rw"]
    pos = 0
    
    new_shader = ""
    decls = shader.split(";")
    for decl in decls:
        if decl.strip() == "":
            continue
        contains_token = False
        for token in tokens:
            if token in decl:
                contains_token = True
        if not contains_token:
            new_shader += decl.strip() + ";\n"
            continue
        decl_start = decl.find("(") + 1
        decl_end = decl.find(")")
        decl_str = decl[decl_start:decl_end].strip()
        decl_params = decl_str.split(",")
        new_decl_str = decl_str + ", " + decl_params[1].strip() + "_buffer"
        new_decl_str = decl.replace(decl_str, new_decl_str).strip()
        
        # replace atomic uint with uint while we're at it
        new_decl_str = new_decl_str.replace("atomic_uint", "uint")
        new_shader += new_decl_str + ";\n"
        
    return new_shader
    
# compile glsl
def compile_glsl(_info, pmfx_name, _tp, _shader):
    # parse inputs and outputs into semantics
    inputs, input_semantics = parse_io_struct(_shader.input_decl)
    outputs, output_semantics = parse_io_struct(_shader.output_decl)
    instance_inputs, instance_input_semantics = parse_io_struct(_shader.instance_input_decl)

    # default 330
    if _tp.shader_version == "0":
        _tp.shader_version = "330"

    # some capabilities
    # binding points for samples and uniform buffers are only supported 420 onwards..
    binding_points = int(_tp.shader_version) >= 420
    texture_cube_array = int(_tp.shader_version) >= 400
    texture_arrays = True
    attribute_stage_in = False
    varying_in = False
    gl_frag_color = False
    explicit_texture_sampling = False
    use_uniform_pack = False
    uniform_pack = None
    if _info.shader_sub_platform == "gles":
        if shader_version_float("gles", _tp.shader_version) <= 200: 
            attribute_stage_in = True
            varying_in = True
            gl_frag_color = True
            explicit_texture_sampling = True
            use_uniform_pack = True
            uniform_pack = dict()
            uniform_pack["decl"] = ""
            uniform_pack["assign"] = ""

    # uniform buffers
    uniform_buffers = ""
    for cbuf in _shader.cbuffers:
        name_start = cbuf.find(" ")
        name_end = cbuf.find(":")
        if name_end == -1:
            continue
        if binding_points:
            reg_start = cbuf.find("register(") + len("register(")
            reg_end = reg_start + cbuf[reg_start:].find(")")
            reg = cbuf[reg_start:reg_end]
            reg = reg.replace("b", " ")
            uniform_buf = "layout (binding=" + reg + ",std140) uniform"
        else:
            uniform_buf = "layout (std140) uniform"
        body_start = cbuf.find("{")
        body_end = cbuf.find("};") + 2
        cbuffer_body = cbuf[body_start:body_end]
        cbuffer_name = cbuf[name_start:name_end]
        if not use_uniform_pack:
            uniform_buf += cbuf[name_start:name_end]
            uniform_buf += "\n"
            uniform_buf += cbuf[body_start:body_end] + "\n"
            uniform_buffers += uniform_buf + "\n"
        else:
            uniform_pack_cbuf = generate_uniform_pack(cbuffer_name, cbuffer_body)
            uniform_pack["decl"] += uniform_pack_cbuf["decl"]
            uniform_pack["assign"] += uniform_pack_cbuf["assign"]
            uniform_buffers += uniform_pack_cbuf["decl"]

    # header and macros
    shader_source = ""
    if _info.shader_sub_platform == "gles":
        if shader_version_float("gles", _tp.shader_version) >= 300: 
            shader_source += "#version " + _tp.shader_version + " es\n"
            # extensions
            for ext in _info.extensions:
                shader_source += "#extension " + ext + " : require\n"
                shader_source += "#define PMFX_" + ext + " 1\n"
            shader_source += "#define GLES3\n"
        else:
            shader_source += "#define GLES2\n"
        shader_source += "#define GLSL\n"
        shader_source += "#define GLES\n"
        if texture_arrays:
            shader_source += "#define PMFX_TEXTURE_ARRAYS\n"
        if shader_version_float("gles", _tp.shader_version) >= 320:
            binding_points = True
        if binding_points:
            shader_source += "#define PMFX_BINDING_POINTS\n"
        if shader_version_float("gles", _tp.shader_version) >= 320:
            shader_source += "#define PMFX_GLES_COMPUTE\n"
    else:
        shader_source += "#version " + _tp.shader_version + " core\n"
        shader_source += "#define GLSL\n"
        if binding_points:
            shader_source += "#define PMFX_BINDING_POINTS\n"
        if texture_cube_array:
            shader_source += "#define PMFX_TEXTURE_CUBE_ARRAY\n"
        if texture_arrays:
            shader_source += "#define PMFX_TEXTURE_ARRAYS\n"


    # texture offset is to avoid collisions on descriptor set slots in vulkan
    if _info.shader_sub_platform == "spirv":
        shader_source += "#define PMFX_TEXTURE_OFFSET " + str(_info.texture_offset) + "\n"
    else:
        shader_source += "#define PMFX_TEXTURE_OFFSET 0\n"
    shader_source += "//" + pmfx_name + " " + _tp.name + " " + _shader.shader_type + " " + str(_tp.id) + "\n"
    shader_source += _info.macros_source

    # input structs
    skip_0 = _info.shader_sub_platform == "spirv"
    index_counter = 0
    for input in inputs:
        if _shader.shader_type == "vs":
            if attribute_stage_in:
                shader_source += "attribute " + input + "_vs_input;\n"
            else:
                shader_source += "layout(location = " + str(index_counter) + ") in " + input + "_vs_input;\n"
        elif _shader.shader_type == "ps":
            if index_counter != 0 or not skip_0:
                if varying_in:
                    shader_source += "varying " + input + "_vs_output;\n"
                else:
                    shader_source += insert_layout_location(index_counter)
                    shader_source += "in " + input + "_vs_output;\n"
        index_counter += 1
    for instance_input in instance_inputs:
        shader_source += insert_layout_location(index_counter)
        shader_source += "layout(location = " + str(index_counter) + ") in " + instance_input + "_instance_input;\n"
        index_counter += 1

    # outputs structs
    index_counter = 0
    if _shader.shader_type == "vs":
        for output in outputs:
            if output.split()[1] != "position":
                if varying_in:
                    shader_source += "varying " + output + "_" + _shader.shader_type + "_output;\n"
                else:
                    shader_source += insert_layout_location(index_counter)
                    shader_source += "out " + output + "_" + _shader.shader_type + "_output;\n"
            index_counter += 1
    elif _shader.shader_type == "ps":
        for p in range(0, len(outputs)):
            if "SV_Depth" in output_semantics[p]:
                continue
            else:
                if not gl_frag_color:
                    output_index = output_semantics[p].replace("SV_Target", "")
                    if output_index != "":
                        shader_source += "layout(location = " + output_index + ") "
                    else:
                        shader_source += insert_layout_location(0)
                    shader_source += "out " + outputs[p] + "_ps_output;\n"

    if _info.v_flip:
        shader_source += "uniform float v_flip;\n"
        
    # global structs for access to inputs or outputs from any function in vs or ps
    if _shader.shader_type != "cs":
        shader_source += generate_global_io_struct(inputs, "struct " + _shader.input_struct_name)
        if _shader.instance_input_struct_name:
            if len(instance_inputs) > 0:
                shader_source += generate_global_io_struct(instance_inputs, "struct " + _shader.instance_input_struct_name)
        if len(outputs) > 0:
            shader_source += generate_global_io_struct(outputs, "struct " + _shader.output_struct_name)

    # convert sample_texture to sample_texture_2d etc
    if explicit_texture_sampling:
        texture_types = texture_types_from_resource_decl(_shader.resource_decl)
        _shader.functions_source = replace_texture_samples(_shader.functions_source, texture_types)
        _shader.main_func_source = replace_texture_samples(_shader.main_func_source, texture_types)
        if uniform_pack:
            _shader.functions_source = insert_uniform_unpack_assignment(_shader.functions_source, uniform_pack)
            _shader.main_func_source = insert_uniform_unpack_assignment(_shader.main_func_source, uniform_pack)

    resource_decl = _shader.resource_decl
    if _info.shader_sub_platform == "gles":
        resource_decl = replace_token_pasting(resource_decl)
    
    shader_source += _tp.struct_decls
    shader_source += uniform_buffers
    shader_source += resource_decl
    shader_source += _shader.functions_source

    glsl_main = _shader.main_func_source
    skip_function_start = glsl_main.find("{") + 1
    skip_function_end = glsl_main.rfind("return")
    glsl_main = glsl_main[skip_function_start:skip_function_end].strip()

    input_name = {
        "vs": "_vs_input",
        "ps": "_vs_output",
        "cs": "_cs_input"
    }

    output_name = {
        "vs": "_vs_output",
        "ps": "_ps_output",
        "cs": "_cs_output"
    }

    if _shader.shader_type == "cs":
        shader_source += "layout("
        shader_source += "local_size_x = " + str(_tp.threads[0]) + ", "
        shader_source += "local_size_y = " + str(_tp.threads[1]) + ", "
        shader_source += "local_size_z = " + str(_tp.threads[2])
        shader_source += ") in;\n"
        shader_source += "void main()\n{\n"
        shader_source += "ivec3 gid = ivec3(gl_GlobalInvocationID);\n"
        shader_source += glsl_main
        shader_source += "\n}\n"
    else:
        # vs and ps need to assign in / out attributes to structs
        pre_assign = generate_input_assignment(inputs, _shader.input_struct_name, "_input", input_name[_shader.shader_type])
        if _shader.instance_input_struct_name:
            if len(instance_inputs) > 0:
                pre_assign += generate_input_assignment(instance_inputs,
                                                        _shader.instance_input_struct_name, "instance_input", "_instance_input")

        post_assign = generate_output_assignment(_info, outputs, "_output", output_name[_shader.shader_type], gl_frag_color)

        shader_source += "void main()\n{\n"
        shader_source += "\n" + pre_assign + "\n"
        shader_source += glsl_main
        shader_source += "\n" + post_assign + "\n"
        shader_source += "}\n"

    # condition source
    shader_source = replace_io_tokens(shader_source)
    shader_source = format_source(shader_source, 4)

    # replace sv_semantic tokens
    for sv in _shader.sv_semantics:
        if sv[0] == "SV_InstanceID":
            shader_source = replace_token(sv[2], "gl_InstanceID", shader_source)
        elif sv[0] == "SV_VertexID":
            shader_source = replace_token(sv[2], "gl_VertexID", shader_source)

    extension = {
        "vs": ".vsc",
        "ps": ".psc",
        "cs": ".csc"
    }

    temp_extension = {
        "vs": ".vert",
        "ps": ".frag",
        "cs": ".comp"
    }

    temp_path = os.path.join(_info.temp_dir, pmfx_name)
    output_path = os.path.join(_info.output_dir, pmfx_name)
    os.makedirs(temp_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    temp_file_and_path = os.path.join(temp_path, _tp.name + temp_extension[_shader.shader_type])

    temp_shader_source = open(temp_file_and_path, "w")
    temp_shader_source.write(shader_source)
    temp_shader_source.close()

    output_path = os.path.join(_info.output_dir, pmfx_name)
    os.makedirs(output_path, exist_ok=True)

    output_file_and_path = os.path.join(output_path, _tp.name + extension[_shader.shader_type])

    if _info.shader_sub_platform == "nvn":
        nvn_sdk = os.getenv("NINTENDO_SDK_ROOT")
        if not nvn_sdk:
            print("error: you must have nintendo switch sdk installed, "
                "'NINTENDO_SDK_ROOT' environment variable is set and is added to your PATH.", flush=True)
            sys.exit(1)

        exe = os.path.normpath(_info.nvn_exe)

        nvn_type = {
            "vs": "-stage vertex",
            "ps": "-stage fragment",
            "cs": "-stage compute"
        }
        cmd = "-input " + sanitize_file_path(temp_file_and_path) + " "
        cmd += nvn_type[_shader.shader_type] + " " + sanitize_file_path(temp_file_and_path) + " "
        cmd += "-output " + sanitize_file_path(output_file_and_path) + " "

        error_code, error_list, output_list = call_wait_subprocess(exe + " " + cmd)
        _tp.error_code = error_code
        _tp.error_list = error_list
        _tp.output_list = output_list

    else:
        exe = os.path.join(_info.tools_dir, "bin", "glsl", get_platform_name(), "validator" + get_platform_exe())

        if _info.shader_sub_platform == "spirv":
            exe += " -V "
            exe += " -o " + output_file_and_path

        error_code, error_list, output_list = call_wait_subprocess(exe + " " + temp_file_and_path)
        _tp.error_code = error_code
        _tp.error_list = error_list
        _tp.output_list = output_list

        if _info.shader_sub_platform != "spirv":
            # copy glsl shader to data
            shader_file = open(output_file_and_path, "w")
            shader_file.write(shader_source)
            shader_file.close()

    return error_code


# we need to convert ubytes 255 to float 1.0
def convert_ubyte_to_float(semantic):
    if semantic.find("COLOR"):
        return False
    return True


# gets metal packed types from hlsl semantic, all types are float except COLOR: uchar, BLENDINDICES uchar
def get_metal_packed_decl(stage_in, input, semantic):
    vector_sizes = ["2", "3", "4"]
    packed_decl = ""
    if not stage_in:
        packed_decl = "packed_"
    split = input.split(" ")
    type = split[0]
    if semantic.find("COLOR") != -1 or semantic.find("BLENDINDICES") != -1:
        packed_decl += "uchar"
        count = type[len(type)-1]
        if count in vector_sizes:
            packed_decl += count
    else:
        packed_decl += type
    for i in range(1, len(split)):
        packed_decl += " " + split[i]
    return packed_decl


# finds token in source code
def find_token(token, string):
    delimiters = [
        "(", ")", "{", "}", ".", ",", "+", "-", "=", "*", "/",
        "&", "|", "~", "\n", "\t", "<", ">", "[", "]", ";", " "
    ]
    fp = string.find(token)
    if fp != -1:
        left = False
        right = False
        # check left
        if fp > 0:
            for d in delimiters:
                if string[fp-1] == d:
                    left = True
                    break
        else:
            left = True
        # check right
        ep = fp + len(token)
        if fp < ep-1:
            for d in delimiters:
                if string[ep] == d:
                    right = True
                    break
        else:
            right = True
        if left and right:
            return fp
        # try again
        tt = find_token(token, string[fp+len(token):])
        if tt == -1:
            return -1
        return fp+len(token) + tt
    return -1


# replace all occurences of token in source code
def replace_token(token, replace, string):
    iter = 0
    while True:
        pos = find_token(token, string)
        if pos == -1:
            break
        else:
            string = string[:pos] + replace + string[pos+len(token):]
            pass
    return string


# metal main functions require textures and buffers to be passed in as args, and do not support global decls
def metal_functions(functions, cbuffers, textures):
    cbuf_members_list = []
    for c in cbuffers:
        cbuf_members = parse_and_split_block(c)
        cbuf_members_list.append(cbuf_members)
    texture_list = textures.split(";")
    texture_args = []
    for t in texture_list:
        cpos = t.find(",")
        if cpos == -1:
            continue
        spos = t.find("(")
        macro_args = t[spos + 1:].split(",")
        tex_type = t[:spos] + "_arg"
        name_pos = 0
        if t.find("texture_2dms") != -1:
            name_pos = 2
        name = macro_args[name_pos].strip()
        texture_args.append((name, tex_type + "(" + name + ")"))
    fl = find_functions(functions)
    final_funcs = ""
    func_sig_additions = dict()
    for f in fl:
        bp = f.find("(")
        ep = f.find(")")
        fb = f[ep:]
        fn = f.find(" ")
        fn = f[fn+1:bp]
        sig = f[:bp+1]
        count = 0
        # insert cbuf members
        for c in cbuf_members_list:
            for i in range(0, len(c), 2):
                ap = c[i+1].find("[")
                member = c[i+1]
                if ap != -1:
                    member = member[:ap]
                if find_token(member, fb) != -1:
                    if count > 0:
                        sig += ",\n"
                    if fn in func_sig_additions.keys():
                        func_sig_additions[fn].append(member)
                    else:
                        func_sig_additions[fn] = [member]
                    ref_type = "& "
                    if ap != -1:
                        ref_type = "* "
                    sig += "constant " + c[i] + ref_type + member
                    count += 1
        # insert texture members
        for t in texture_args:
            if find_token(t[0], fb) != -1:
                if count > 0:
                    sig += ",\n"
                sig += t[1]
                count += 1
                if fn in func_sig_additions.keys():
                    func_sig_additions[fn].append(t[0])
                    func_sig_additions[fn].append("sampler_" + t[0])
                else:
                    func_sig_additions[fn] = [t[0]]
                    func_sig_additions[fn].append("sampler_" + t[0])
        if bp != -1 and ep != -1:
            args = f[bp+1:ep]
            arg_list = args.split(",")
            for arg in arg_list:
                if count > 0:
                    sig += ",\n"
                count += 1
                address_space = "thread"
                toks = arg.split(" ")
                if '' in toks:
                    toks.remove('')
                if '\n' in toks:
                    toks.remove('\n')
                ref = False
                for t in toks:
                    if t == "out" or t == "inout":
                        ref = True
                    if t == "in":
                        address_space = "constant"
                        ref = True
                if not ref:
                    sig += arg
                else:
                    array = toks[2].find("[")
                    if array == -1:
                        sig += address_space + " " + toks[1] + "& " + toks[2]
                    else:
                        sig += address_space + " " + toks[1] + "* " + toks[2][:array]
        # find used cbuf memb
        func = sig + fb
        final_funcs += func
    return final_funcs, func_sig_additions


# cascade through and pass textures and buffers to function calls in metal source code
def insert_function_sig_additions(function_body, function_sig_additions):
    for k in function_sig_additions.keys():
        op = 0
        fp = 0
        while fp != -1:
            fp = find_token(k, function_body[op:])
            if fp != -1:
                fp = op + fp
                fp += len(k)
                insert_string = function_body[:fp+1]
                for a in function_sig_additions[k]:
                    insert_string += a + ", "
                insert_string += function_body[fp+1:]
                function_body = insert_string
                op = fp
    return function_body


# compile shader for apple metal
def compile_metal(_info, pmfx_name, _tp, _shader):
    # parse inputs and outputs into semantics
    inputs, input_semantics = parse_io_struct(_shader.input_decl)
    outputs, output_semantics = parse_io_struct(_shader.output_decl)
    instance_inputs, instance_input_semantics = parse_io_struct(_shader.instance_input_decl)

    shader_source = "#include <metal_stdlib>\n"
    shader_source += "using namespace metal;\n"
    shader_source += "#define BUF_OFFSET " + str(_info.cbuffer_offset) + "\n"
    shader_source += _info.macros_source

    # struct decls
    shader_source += _tp.struct_decls

    stream_out = False
    if "stream_out" in _tp.technique.keys():
        if _tp.technique["stream_out"]:
            stream_out = True

    # cbuffer decls
    metal_cbuffers = []
    for cbuf in _shader.cbuffers:
        name_start = cbuf.find(" ")
        name_end = cbuf.find(":")
        body_start = cbuf.find("{")
        body_end = cbuf.find("};") + 2
        register_start = cbuf.find("(") + 1
        register_end = cbuf.find(")")
        name = cbuf[name_start:name_end].strip()
        reg = cbuf[register_start:register_end]
        reg = reg.replace('b', '')
        metal_cbuffers.append((name, reg))
        shader_source += "struct c_" + name + "\n"
        shader_source += cbuf[body_start:body_end]
        shader_source += "\n"

    # packed inputs
    vs_stage_in = _info.stage_in
    attrib_index = 0
    if _shader.shader_type == "vs":
        if vs_stage_in:
            if len(inputs) > 0:
                shader_source += "struct packed_" + _shader.input_struct_name + "\n{\n"
                for i in range(0, len(inputs)):
                    shader_source += get_metal_packed_decl(vs_stage_in, inputs[i], input_semantics[i])
                    shader_source += " [[attribute(" + str(attrib_index) + ")]]"
                    shader_source += ";\n"
                    attrib_index += 1
            if _shader.instance_input_struct_name:
                for i in range(0, len(instance_inputs)):
                    shader_source += get_metal_packed_decl(vs_stage_in, instance_inputs[i], instance_input_semantics[i])
                    shader_source += " [[attribute(" + str(attrib_index) + ")]]"
                    shader_source += ";\n"
                    attrib_index += 1
            shader_source += "};\n"
        else:
            if len(inputs) > 0:
                shader_source += "struct packed_" + _shader.input_struct_name + "\n{\n"
                for i in range(0, len(inputs)):
                    shader_source += get_metal_packed_decl(vs_stage_in, inputs[i], input_semantics[i])
                    shader_source += ";\n"
                    attrib_index += 1
                shader_source += "};\n"
            if _shader.instance_input_struct_name:
                if len(instance_inputs) > 0:
                    shader_source += "struct packed_" + _shader.instance_input_struct_name + "\n{\n"
                    for i in range(0, len(instance_inputs)):
                        shader_source += get_metal_packed_decl(vs_stage_in, instance_inputs[i], instance_input_semantics[i])
                        shader_source += ";\n"
                        attrib_index += 1
                    shader_source += "};\n"

    # inputs
    if len(inputs) > 0:
        shader_source += "struct " + _shader.input_struct_name + "\n{\n"
        for i in range(0, len(inputs)):
            shader_source += inputs[i] + ";\n"
        shader_source += "};\n"

    if _shader.instance_input_struct_name:
        if len(instance_inputs) > 0:
            shader_source += "struct " + _shader.instance_input_struct_name + "\n{\n"
            for i in range(0, len(instance_inputs)):
                shader_source += instance_inputs[i] + ";\n"
            shader_source += "};\n"

    # outputs
    if len(outputs) > 0:
        shader_source += "struct " + _shader.output_struct_name + "\n{\n"
        for i in range(0, len(outputs)):
            shader_source += outputs[i]
            if output_semantics[i].find("SV_POSITION") != -1:
                shader_source += " [[position]]"
            # mrt
            sv_pos = output_semantics[i].find("SV_Target")
            if sv_pos != -1:
                channel_pos = sv_pos + len("SV_Target")
                if channel_pos < len(output_semantics[i]):
                    shader_source += " [[color(" + output_semantics[i][channel_pos] + ")]]"
                else:
                    shader_source += " [[color(0)]]"
            sv_pos = output_semantics[i].find("SV_Depth")
            if sv_pos != -1:
                shader_source += " [[depth(any)]]"
            shader_source += ";\n"
        shader_source += "};\n"

    main_type = {
        "vs": "vertex",
        "ps": "fragment",
        "cs": "kernel"
    }

    # functions
    function_source, function_sig_additions = metal_functions(_shader.functions_source,
                                                              _shader.cbuffers, _shader.resource_decl)
    shader_source += function_source

    # main decl
    stream_out_name = _shader.output_struct_name
    if stream_out:
        _shader.output_struct_name = "void"

    # sv sematics
    vertex_id_var = "vid"
    instance_id_var = "iid"

    for sv in _shader.sv_semantics:
        if sv[0] == "SV_InstanceID":
            instance_id_var = sv[2]
        elif sv[0] == "SV_VertexID":
            vertex_id_var = sv[2]

    shader_source += main_type[_shader.shader_type] + " "
    shader_source += _shader.output_struct_name + " " + _shader.shader_type + "_main" + "("

    if _shader.shader_type == "vs":
        shader_source += "\n  uint " + vertex_id_var + " [[vertex_id]]"
        shader_source += "\n, uint " + instance_id_var + " [[instance_id]]"

    if _shader.shader_type == "vs" and not vs_stage_in:
        shader_source += "\n, const device packed_" + _shader.input_struct_name + "* vertices" + "[[buffer(0)]]"
        if _shader.instance_input_struct_name:
            if len(instance_inputs) > 0:
                shader_source += "\n, const device packed_" + _shader.instance_input_struct_name + "* instances" + "[[buffer(1)]]"
    elif _shader.shader_type == "vs":
        shader_source += "\n, packed_" + _shader.input_struct_name + " in_vertex [[stage_in]]"
    elif _shader.shader_type == "ps":
        shader_source += _shader.input_struct_name + " input [[stage_in]]"
    elif _shader.shader_type == "cs":
        shader_source += "uint3 gid[[thread_position_in_grid]]"

    # vertex stream out
    if stream_out:
        shader_source += "\n,  device " + stream_out_name + "* stream_out_vertices" + "[[buffer(7)]]"

    # pass in textures and buffers
    invalid = ["", "\n"]
    texture_list = _shader.resource_decl.split(";")
    for texture in texture_list:
        if texture not in invalid:
            shader_source += "\n, " + texture.strip("\n")

    cbuffer_offset = _info.cbuffer_offset
    # pass in cbuffers.. cbuffers start at cbuffer_offset reserving space for (cbuffer_offset-1) vertex buffers..
    for cbuf in metal_cbuffers:
        regi = int(cbuf[1]) + cbuffer_offset
        shader_source += "\n, " + "constant " "c_" + cbuf[0] + " &" + cbuf[0] + " [[buffer(" + str(regi) + ")]]"

    shader_source += ")\n{\n"

    vertex_array_index = "(vertices[" + vertex_id_var + "]."
    instance_array_index = "(instances[" + instance_id_var + "]."
    if vs_stage_in:
        vertex_array_index = "(in_vertex."
        instance_array_index = "(in_vertex."

    # create function prologue for main and insert assignment to unpack vertex
    from_ubyte = "0.00392156862"
    if _shader.shader_type == "vs":
        shader_source += _shader.input_struct_name + " input;\n"
        v_inputs = [(inputs, input_semantics, "input.", vertex_array_index)]
        if _shader.instance_input_struct_name:
            if len(instance_inputs) > 0:
                shader_source += _shader.instance_input_struct_name + " instance_input;\n"
                v_inputs.append((instance_inputs, instance_input_semantics, "instance_input.", instance_array_index))
        for vi in v_inputs:
            for i in range(0, len(vi[0])):
                split_input = vi[0][i].split(" ")
                input_name = split_input[1]
                input_unpack_type = split_input[0]
                shader_source += vi[2] + input_name + " = "
                shader_source += input_unpack_type
                shader_source += vi[3] + input_name
                # convert ubyte to float
                if convert_ubyte_to_float(vi[1][i]):
                    shader_source += ") * " + from_ubyte + ";"
                else:
                    shader_source += ");\n"

    used_code = function_source + " " + _shader.main_func_source

    # create a function prologue for cbuffer assignment
    for c in range(0, len(_shader.cbuffers)):
        cbuf_members = parse_and_split_block(_shader.cbuffers[c])
        for i in range(0, len(cbuf_members), 2):
            ref_type = "& "
            point = ""
            decl = cbuf_members[i + 1]
            assign = decl
            array_pos = cbuf_members[i + 1].find("[")
            if array_pos != -1:
                decl = decl[:array_pos]
                ref_type = "* "
                assign = decl + "[0]"
                point = "&"
            # check for use
            if find_token(decl, used_code) == -1:
                continue
            shader_source += "constant " + cbuf_members[i] + ref_type + decl
            shader_source += " = " + point + metal_cbuffers[c][0] + "." + assign
            shader_source += ";\n"

    main_func_body = _shader.main_func_source.find("{") + 1
    main_body_source = _shader.main_func_source[main_func_body:]
    main_body_source = insert_function_sig_additions(main_body_source, function_sig_additions)

    shader_source += main_body_source
    shader_source = format_source(shader_source, 4)

    if stream_out:
        shader_source = shader_source.replace("return output;", "stream_out_vertices[vid] = output;")

    temp_path = os.path.join(_info.temp_dir, pmfx_name)
    output_path = os.path.join(_info.output_dir, pmfx_name)
    os.makedirs(temp_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    extension = {
        "vs": "_vs.metal",
        "ps": "_ps.metal",
        "cs": "_cs.metal"
    }

    output_extension = {
        "vs": ".vsc",
        "ps": ".psc",
        "cs": ".csc"
    }

    temp_file_and_path = os.path.join(temp_path, _tp.name + extension[_shader.shader_type])
    output_file_and_path = os.path.join(output_path, _tp.name + output_extension[_shader.shader_type])

    compiled = _info.compiled
    if not compiled:
        temp_shader_source = open(output_file_and_path, "w")
        temp_shader_source.write(shader_source)
        temp_shader_source.close()
        return 0
    else:
        # default to metal 2.0, but allow cmdline override
        metal_version = "2.0"
        if _tp.shader_version != "0":
            metal_version = _tp.shader_version

        # selection of sdk, macos, ios, tvos
        metal_sdk = "macosx"
        if _info.metal_sdk != "":
            metal_sdk = _info.metal_sdk

        # insert some defaults fo version min based on os
        metal_min_os = ""
        if metal_sdk == "macosx":
            metal_min_os = "10.11"
            if _info.metal_min_os != "":
                metal_min_os = _info.metal_min_os
            metal_min_os = "-mmacosx-version-min=" + metal_min_os
        elif metal_sdk == "iphoneos":
            metal_min_os = "9.0"
            if _info.metal_min_os != "":
                metal_min_os = _info.metal_min_os
            metal_min_os = "-mios-version-min=" + metal_min_os
        elif metal_sdk == "appletvos":
            metal_min_os = "13.0"
            if _info.metal_min_os != "":
                metal_min_os = _info.metal_min_os
            metal_min_os = "-mtvos-version-min=" + metal_min_os

        # finally set metal -std.
        if metal_sdk == "iphoneos" or metal_sdk == "appletvos":
            metal_version = "-std=ios-metal" + metal_version
        else:
            metal_version = "-std=macos-metal" + metal_version

        temp_shader_source = open(temp_file_and_path, "w")
        temp_shader_source.write(shader_source)
        temp_shader_source.close()

        intermediate_file_and_path = temp_file_and_path.replace(".frag", "_frag.air")
        intermediate_file_and_path = intermediate_file_and_path.replace(".vert", "_vert.air")

        # compile .air
        cmdline = "xcrun -sdk " + metal_sdk + " metal " + metal_min_os + " " + metal_version + " -c "
        cmdline += temp_file_and_path + " "
        cmdline += "-o " + intermediate_file_and_path

        error_code, error_list, output_list = call_wait_subprocess(cmdline)

        if error_code == 0:
            cmdline = "xcrun -sdk " + metal_sdk + " metallib "
            cmdline += intermediate_file_and_path + " "
            cmdline += "-o " + output_file_and_path

            error_code, error_list_2, output_list_2 = call_wait_subprocess(cmdline)
            error_list.extend(error_list_2)
            output_list.extend(output_list_2)

        if error_code != 0:
            _tp.error_code = error_code
            _tp.error_list = error_list
            _tp.output_list = output_list


# generate a shader info file with an array of technique permutation descriptions and dependency timestamps
def generate_shader_info(filename, included_files, techniques):
    global _info
    info_filename, base_filename, dir_path = get_resource_info_filename(filename, _info.output_dir)

    shader_info = dict()
    shader_info["cmdline"] = _info.cmdline_string
    shader_info["files"] = []
    shader_info["techniques"] = techniques["techniques"]
    shader_info["failures"] = techniques["failures"]

    # special files which affect the validity of compiled shaders
    shader_info["files"].append(create_dependency(_info.this_file))
    shader_info["files"].append(create_dependency(_info.macros_file))
    shader_info["files"].append(create_dependency(_info.platform_macros_file))

    included_files.insert(0, os.path.join(dir_path, base_filename))
    for ifile in included_files:
        full_name = os.path.join(_info.root_dir, ifile)
        shader_info["files"].append(create_dependency(full_name))

    output_info = open(info_filename, 'wb+')
    output_info.write(bytes(json.dumps(shader_info, indent=4), 'UTF-8'))
    output_info.close()
    return shader_info


# generate json description of vs inputs and outputs
def generate_input_info(inputs):
    semantic_info = [
        ["SV_POSITION", "4"],
        ["POSITION", "4"],
        ["TEXCOORD", "4"],
        ["NORMAL", "4"],
        ["TANGENT", "4"],
        ["BITANGENT", "4"],
        ["BLENDWEIGHTS", "4"],
        ["COLOR", "1"],
        ["BLENDINDICES", "1"]
    ]
    type_info = ["int", "uint", "float", "double"]
    input_desc = []
    inputs_split = parse_and_split_block(inputs)
    offset = int(0)
    for i in range(0, len(inputs_split), 3):
        num_elements = 1
        element_size = 1
        for type in type_info:
            if inputs_split[i].find(type) != -1:
                str_num = inputs_split[i].replace(type, "")
                if str_num != "":
                    num_elements = int(str_num)
        for sem in semantic_info:
            if inputs_split[i+2].find(sem[0]) != -1:
                semantic_id = semantic_info.index(sem)
                semantic_name = sem[0]
                semantic_index = inputs_split[i+2].replace(semantic_name, "")
                if semantic_index == "":
                    semantic_index = "0"
                element_size = sem[1]
                break
        size = int(element_size) * int(num_elements)
        input_attribute = {
            "name": inputs_split[i+1],
            "semantic_index": int(semantic_index),
            "semantic_id": int(semantic_id),
            "size": int(size),
            "element_size": int(element_size),
            "num_elements": int(num_elements),
            "offset": int(offset),
        }
        input_desc.append(input_attribute)
        offset += size
    return input_desc


# generate metadata for the technique with info about textures, cbuffers, inputs, outputs, binding points and more
def generate_technique_permutation_info(_tp):
    _tp.technique["name"] = _tp.technique_name
    # textures
    shader_resources_split = parse_and_split_block(_tp.resource_decl)
    i = 0
    _tp.technique["texture_sampler_bindings"] = []
    _tp.technique["structured_buffers"] = []
    while i < len(shader_resources_split):
        offset = i
        tex_type = shader_resources_split[i+0]
        # structured buffers
        if tex_type.find("structured_buffer") != -1:
            offset = i+1
            buffer_desc = {
                "type": shader_resources_split[i+1],
                "name": shader_resources_split[i+2],
                "location": shader_resources_split[i+3]
            }
            _tp.technique["structured_buffers"].append(buffer_desc)
        else:
            # textures
            if tex_type == "texture_2dms":
                data_type = shader_resources_split[i+1]
                fragments = shader_resources_split[i+2]
                offset = i+2
            else:
                data_type = "float4"
                fragments = 1
            sampler_desc = {
                "name": shader_resources_split[offset+1],
                "data_type": data_type,
                "fragments": fragments,
                "type": tex_type,
                "unit": int(shader_resources_split[offset+2])
            }
            _tp.technique["texture_sampler_bindings"].append(sampler_desc)
        i = offset+3
    # cbuffers
    _tp.technique["cbuffers"] = []
    for buffer in _tp.cbuffers:
        pos = buffer.find("{")
        if pos == -1:
            continue
        buffer_decl = buffer[0:pos-1]
        buffer_decl_split = buffer_decl.split(":")
        buffer_name = buffer_decl_split[0].split()[1]
        buffer_loc_start = buffer_decl_split[1].find("(") + 1
        buffer_loc_end = buffer_decl_split[1].find(")", buffer_loc_start)
        buffer_reg = buffer_decl_split[1][buffer_loc_start:buffer_loc_end]
        buffer_reg = buffer_reg.strip('b')
        buffer_desc = {"name": buffer_name, "location": int(buffer_reg)}
        _tp.technique["cbuffers"].append(buffer_desc)
    # io structs from vs.. vs input, instance input, vs output (ps input)
    _tp.technique["vs_inputs"] = generate_input_info(_tp.shader[0].input_decl)
    _tp.technique["instance_inputs"] = generate_input_info(_tp.shader[0].instance_input_decl)
    _tp.technique["vs_outputs"] = generate_input_info(_tp.shader[0].output_decl)
    # vs and ps files
    if "vs" in _tp.filenames.keys():
        _tp.technique["vs_file"] = _tp.filenames["vs"] + ".vsc"
    if "ps" in _tp.filenames.keys():
        _tp.technique["ps_file"] = _tp.filenames["ps"] + ".psc"
    if "cs" in _tp.filenames.keys():
        _tp.technique["cs_file"] = _tp.filenames["cs"] + ".csc"
    # permutation
    _tp.technique["permutations"] = _tp.permutation_options
    _tp.technique["permutation_id"] = _tp.id
    _tp.technique["permutation_option_mask"] = _tp.mask
    return _tp.technique


# compiles single shader using platform specific compiler or validator, _tp is technique / permutation info
def compile_single_shader(_tp):
    for s in _tp.shader:
        if s.duplicate:
            continue
        if _info.shader_platform == "hlsl":
            compile_hlsl(_info, _tp.pmfx_name, _tp, s)
        elif _info.shader_platform == "pssl":
            compile_pssl(_info, _tp.pmfx_name, _tp, s)
        elif _info.shader_platform == "glsl":
            compile_glsl(_info, _tp.pmfx_name, _tp, s)
        elif _info.shader_platform == "metal":
            compile_metal(_info, _tp.pmfx_name, _tp, s)
        else:
            print("error: invalid shader platform " + _info.shader_platform, flush=True)


# parse a pmfx file which is a collection of techniques and permutations, made up of vs, ps, cs combinations
def parse_pmfx(file, root):
    global _info

    # new pmfx info
    _pmfx = PmfxInfo()

    file_and_path = os.path.join(root, file)
    shader_file_text, included_files = create_shader_set(file_and_path, root)

    _pmfx.json = find_pmfx_json(shader_file_text)
    _pmfx.source = shader_file_text
    _pmfx.json_text = json.dumps(_pmfx.json)

    # pmfx file may be an include or library module containing only functions
    if not _pmfx.json:
        return

    # check dependencies
    force = False
    up_to_date = check_dependencies(file_and_path, included_files)
    if up_to_date and not force:
        print(file + " file up to date", flush=True)
        return

    print(file, flush=True)
    c_code = ""

    pmfx_name = os.path.basename(file).replace(".pmfx", "")

    pmfx_output_info = dict()
    pmfx_output_info["techniques"] = []

    # add cbuffers and structs as c structs
    c_code += "namespace " + pmfx_name + "\n{\n"
    global_cbuffers = find_constant_buffers(_pmfx.source)
    # structs
    global_structs = find_struct_declarations(_pmfx.source)
    for s in global_structs:
        c_code += s
    # cbuffers
    for buf in global_cbuffers:
        decl = buf[:buf.find(":")].split(" ")
        c_code += "\nstruct " + decl[1] + "\n"
        body = buf.find("{")
        c_code += buf[body:]

    # for techniques in pmfx
    success = True
    compile_jobs = []
    for technique in _pmfx.json:
        pmfx_json = json.loads(_pmfx.json_text)
        technique_json = pmfx_json[technique].copy()
        technique_json = inherit_technique(technique_json, pmfx_json)
        technique_permutations, permutation_options, mask, define_list, c_defines = generate_permutations(technique, technique_json)
        c_code += c_defines

        # for permutations in technique
        for permutation in technique_permutations:
            pmfx_json = json.loads(_pmfx.json_text)
            _tp = TechniquePermutationInfo()
            _tp.pmfx_name = pmfx_name
            _tp.shader = []
            _tp.cbuffers = []

            # gather technique permutation info
            _tp.id = generate_permutation_id(define_list, permutation)
            _tp.permutation = permutation
            _tp.technique_name = technique
            _tp.technique = inherit_technique(pmfx_json[technique], pmfx_json)
            _tp.mask = mask
            _tp.permutation_options = permutation_options

            valid = True
            _tp.shader_version = _info.shader_version
            if "supported_platforms" in _tp.technique:
                p = shader_sub_platform()
                sp = _tp.technique["supported_platforms"]
                if p not in sp:
                    print(_tp.technique_name + " not supported on " + p, flush=True)
                    valid = False
                else:
                    sv = sp[p]
                    if "all" in sv:
                        pass
                    elif _tp.shader_version not in sv:
                        valid = False
                        print(_tp.technique_name + " not supported on " +
                              p + " " + _info.shader_version +
                              ", forcing to version " + sv[0], flush=True)
                        # force shader version to specified
                        _tp.shader_version = sv[0]

            if not valid:
                continue

            if _tp.id != 0:
                _tp.name = _tp.technique_name + "__" + str(_tp.id) + "__"
            else:
                _tp.name = _tp.technique_name

            # strip condition permutations from source
            permutation.append((_info.shader_platform.upper(), 1))
            permutation.append((shader_sub_platform().upper(), 1))
            _tp.source = evaluate_conditional_blocks(_pmfx.source, permutation)

            # get permutation constants..
            _tp.technique = get_permutation_conditionals(_tp.technique, _tp.permutation)

            # global cbuffers
            _tp.cbuffers = find_constant_buffers(_pmfx.source)

            # technique, permutation specific constants
            _tp.technique, c_struct, tp_cbuffer = generate_technique_constant_buffers(pmfx_json, _tp)
            c_code += c_struct

            # add technique / permutation specific cbuffer to the list
            _tp.cbuffers.append(tp_cbuffer)

            # technique, permutation specific textures..
            _tp.textures = generate_technique_texture_variables(_tp)
            _tp.resource_decl = find_shader_resources(_tp.source)

            # add technique textures
            if _tp.textures:
                _tp.resource_decl += generate_texture_decl(_tp.textures)

            # find functions
            _tp.functions = find_functions(_tp.source)

            # find structs
            struct_list = find_struct_declarations(_tp.source)
            _tp.struct_decls = ""
            for struct in struct_list:
                _tp.struct_decls += struct + "\n"

            # number of threads for cs
            if "threads" in pmfx_json[technique]:
                threads = pmfx_json[technique]["threads"]
                _tp.threads = [1, 1, 1]
                for i in range(0, len(threads)):
                    _tp.threads[i] = threads[i]

            # generate single shader data
            shader_types = ["vs", "ps", "cs"]
            for s in shader_types:
                if s in _tp.technique.keys():
                    single_shader = generate_single_shader(_tp.technique[s], _tp)
                    single_shader.shader_type = s
                    if single_shader:
                        _tp.shader.append(single_shader)
            compile_jobs.append(copy.copy(_tp))

    # find duplicated / redundant permutation combinations
    unique = dict()
    for j in compile_jobs:
        j.filenames = dict()
        for s in j.shader:
            hash = shader_hash(s)
            if hash not in unique:
                s.duplicate = False
                unique[str(hash)] = j.name
                j.filenames[s.shader_type] = j.name
            else:
                s.duplicate = True
                j.filenames[s.shader_type] = unique[str(hash)]

    threads = []
    for j in compile_jobs:
        x = threading.Thread(target=compile_single_shader, args=(j,))
        threads.append(x)
        x.start()

    # wait for threads
    for t in threads:
        t.join()

    pmfx_output_info["failures"] = dict()
    for i in range(0, len(compile_jobs)):
        c = compile_jobs[i]
        str_id = ""
        if c.id != 0:
            str_id = "__" + str(c.id) + "__"
        output_name = c.pmfx_name + "::" + c.technique_name + str_id
        if c.error_code == 0:
            print(output_name, flush=True)
        else:
            print(output_name + " failed to compile", flush=True)
            pmfx_output_info["failures"][c.pmfx_name] = True
        for out in c.output_list:
            print(out, flush=True)
        for err in c.error_list:
            print(err, flush=True)
        pmfx_output_info["techniques"].append(generate_technique_permutation_info(compile_jobs[i]))

    # write a shader info file with timestamp for dependencies
    generate_shader_info(file_and_path, included_files, pmfx_output_info)

    # write out a c header for accessing materials in code
    if c_code != "":
        c_code += "}\n"
        fmt = ""
        lines = c_code.split("\n")
        if len(lines) > 3:
            indents = 0
            for l in lines:
                if l == "":
                    continue
                if l.find("}") != -1:
                    indents -= 1
                for i in range(0, indents):
                    fmt += "    "
                fmt += l.strip() + "\n"
                if l.find("{") != -1:
                    indents += 1
            h_filename = file.replace(".pmfx", ".h")
            h_filename = os.path.basename(h_filename)
            if not os.path.exists(_info.struct_dir):
                os.mkdir(_info.struct_dir)
            h_filename = os.path.join(_info.struct_dir, h_filename)
            h_file = open(h_filename, "w+")
            h_file.write(fmt)
            h_file.close()


# handles some hardcoded cases of platform varitions
def configure_sub_platforms():
    global _info
    if _info.shader_platform == "spirv":
        _info.shader_platform = "glsl"
        _info.shader_version = "450"
        _info.shader_sub_platform = "spirv"
    elif _info.shader_platform == "gles":
        _info.shader_platform = "glsl"
        _info.shader_sub_platform = "gles"
    elif _info.shader_platform == "nvn":
        _info.shader_platform = "glsl"
        _info.shader_sub_platform = "nvn"


# main function to avoid shadowing
def main():
    print("--------------------------------------------------------------------------------", flush=True)
    print("pmfx shader (v3) ---------------------------------------------------------------", flush=True)
    print("--------------------------------------------------------------------------------", flush=True)

    global _info
    _info = BuildInfo()
    _info.error_code = 0

    parse_args()
    configure_sub_platforms()

    # get dirs for build output
    _info.root_dir = os.getcwd()
    _info.this_file = os.path.realpath(__file__)
    _info.pmfx_dir = os.path.dirname(_info.this_file)
    _info.macros_file = os.path.join(_info.pmfx_dir, "platform", "pmfx.h")
    _info.platform_macros_file = os.path.join(_info.pmfx_dir, "platform", _info.shader_platform + ".h")
    _info.tools_dir = _info.pmfx_dir

    # global shader macros for glsl, hlsl and metal portability
    mf = open(_info.platform_macros_file)
    _info.macros_source = mf.read()
    mf.close()
    mf = open(_info.macros_file)
    _info.macros_source += mf.read()
    mf.close()

    source_list = _info.inputs
    for source in source_list:
        if os.path.isdir(source):
            for root, dirs, files in os.walk(source):
                for file in files:
                    if file.endswith(".pmfx"):
                        try:
                            parse_pmfx(file, root)
                        except Exception as e:
                            print("ERROR: while processing", os.path.join(root, file), flush=True)
                            raise e
        else:
            parse_pmfx(source, "")

    # error code for ci
    sys.exit(_info.error_code)


# entry
if __name__ == "__main__":
    main()
