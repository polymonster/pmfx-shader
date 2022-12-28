# cgu - code gen utilities for parsing c-like languages for use in code generation tools
# copyright Alex Dixon 2020: https://github.com/polymonster/cgu/blob/master/license
import re
import json
import sys


# make code gen more readable and less fiddly
def in_quotes(string):
    return "\"" + string + "\""


# append to string with newline print() style
def src_line(line):
    line += "\n"
    return line


# like c style unsigned wraparound
def us(val):
    if val < 0:
        val = sys.maxsize + val
    return val


# remove all single and multi line comments
def remove_comments(source):
    lines = source.split("\n")
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


# generates a nice UI friendly name from, snake_case, camelCase or SCAREY_CASE and strip known prefixes
def display_name(token, title):
    prefix = ["m_", "s_", "k_", "g_"]
    for p in prefix:
        if token.startswith(p):
            token = token[len(p):]
            break
    spaced = ""
    for i in range(len(token)):
        if i > 0:
            if token[i-1].islower() and token[i].isupper():
                spaced += " "
        spaced += token[i]
    spaced = spaced.replace("_", " ")
    if title:
        spaced = spaced.title()
    else:
        spaced = spaced.capitalize()
    return spaced.strip()


# separate alpha and numeric characters ie. TEXCOORD0 becomes (TEXCOORD, 0)
def separate_alpha_numeric(src):
    name = re.sub(r'[0-9]', '', src)
    index = re.sub(r'[^0-9]','', src)
    if len(index) == 0:
        index = 0
    index = int(index)
    return (name, index)


# finds the end of a body of text enclosed between 2 symbols ie. [], {}, <>
def enclose(open_symbol, close_symbol, source, pos):
    pos = source.find(open_symbol, pos)
    stack = [open_symbol]
    pos += 1
    while len(stack) > 0 and pos < len(source):
        if source[pos] == open_symbol:
            stack.append(open_symbol)
        if source[pos] == close_symbol:
            stack.pop()
        pos += 1
    return pos


# returns the interior range of the string contained within open_symbol and close_symbol
def enclose_start_end(open_symbol, close_symbol, source, pos):
    pos = source.find(open_symbol, pos)
    if pos == -1:
        return (-1, -1)
    start_pos = pos+1
    stack = [open_symbol]
    pos += 1
    while len(stack) > 0 and pos < len(source):
        if source[pos] == open_symbol:
            stack.append(open_symbol)
        if source[pos] == close_symbol:
            stack.pop()
        pos += 1
    return (start_pos, pos-1)


# parse a string and return the end position in source, taking into account escaped \" quotes
def enclose_string(start, source):
    pos = start+1
    while True:
        pos = source.find("\"", pos)
        prev = pos - 1
        if prev > 0:
            if source[prev] == "\\":
                pos = pos + 1
                continue
            return pos+1
    # un-terminated string
    print("ERROR: unterminated string")
    assert 0


# format source with indents
def format_source(source, indent_size):
    formatted = ""
    lines = source.splitlines()
    indent = 0
    indents = ["{"]
    unindnets = ["}"]
    newline = False
    for line in lines:
        if newline and len(line) > 0 and line[0] != "}":
            formatted += "\n"
        newline = False
        cur_indent = indent
        line = line.strip()
        attr = line.find("[[")
        if len(line) < 1 or attr != -1:
            continue
        for c in line:
            if c in indents:
                indent += 1
            elif c in unindnets:
                indent -= 1
                cur_indent = indent
                newline = True
        formatted += " " * cur_indent * indent_size
        formatted += line
        formatted += "\n"
    return formatted


# returns the name of a type.. ie struct <name>, enum <name>
def type_name(type_declaration):
    pos = type_declaration.find("{")
    name = type_declaration[:pos].strip().split()[1]
    return name


# get the typename stripping resource_array[21] arrays and return (resource_array, 21)
def type_name_array(type_declaration):
    pos = type_declaration.find("[")
    if pos != -1:
        start, end = enclose_start_end("[", "]", type_declaration, 0)
        array_size_str = type_declaration[start:end]
        if len(array_size_str) == 0:
            array_size = -1
        else:
            array_size = int(array_size_str)
        return (type_declaration[:pos], array_size)
    return (type_declaration, None)


# tidy source with consistent spaces, remove tabs and comments to make subsequent operations easier
def sanitize_source(source):
    # replace tabs with spaces
    source = source.replace("\t", " ")
    # replace all spaces with single space
    source = re.sub(' +', ' ', source)
    # remove comments
    source = remove_comments(source)
    # remove empty lines and strip whitespace
    sanitized = ""
    for line in source.splitlines():
        line = line.strip()
        if len(line) > 0:
            sanitized += src_line(line)
    return sanitized


# finds token in source code
def find_token(token, source):
    delimiters = [
        "(", ")", "{", "}", ".", ",", "+", "-", "=", "*", "/",
        "&", "|", "~", "\n", "\t", "<", ">", "[", "]", ";", " "
    ]
    fp = source.find(token)
    if fp != -1:
        left = False
        right = False
        # check left
        if fp > 0:
            for d in delimiters:
                if source[fp - 1] == d:
                    left = True
                    break
        else:
            left = True
        # check right
        ep = fp + len(token)
        if fp < ep-1:
            for d in delimiters:
                if source[ep] == d:
                    right = True
                    break
        else:
            right = True
        if left and right:
            return fp
        # try again
        tt = find_token(token, source[fp + len(token):])
        if tt == -1:
            return -1
        return fp+len(token) + tt
    return -1


# replace all occurrences of token in source code
def replace_token(token, replace, source):
    while True:
        pos = find_token(token, source)
        if pos == -1:
            break
        else:
            source = source[:pos] + replace + source[pos + len(token):]
            pass
    return source


# find all occurences of token, with their location within source
def find_all_tokens(token, source):
    pos = 0
    locations = []
    while True:
        token_pos = find_token(token, source[pos:])
        if token_pos != -1:
            token_pos += pos
            locations.append(token_pos)
            pos = token_pos + len(token)
        else:
            break
    return locations


# find all string literals in source
def find_string_literals(source):
    pos = 0
    strings = []
    while True:
        pos = source.find("\"", pos)
        if pos == -1:
            break
        end = enclose_string(pos, source)
        string = source[pos:end]
        strings.append(string)
        pos = end+1
    return strings


# removes string literals and inserts a place holder, returning the ist of string literals and the conditioned source
def placeholder_string_literals(source):
    strings = find_string_literals(source)
    index = 0
    for s in strings:
        source = source.replace(s, '"<placeholder_string_literal_{}>"'.format(index))
        index += 1
    return strings, source


# replace placeholder literals with the strings
def replace_placeholder_string_literals(strings, source):
    index = 0
    for s in strings:
        source = source.replace('"<placeholder_string_literal_{}>"'.format(index), s)
        index += 1
    return source


# get all enum member names and values
def get_enum_members(declaration):
    start = declaration.find("{")+1
    end = enclose("{", "}", declaration, 0)-1
    body = declaration[start:end]
    members = body.split(",")
    conditioned = []
    for member in members:
        if len(member.strip()) > 0:
            conditioned.append(member.strip())
    enum_value = 0
    enum_members = []
    for member in conditioned:
        if member.find("=") != -1:
            name_value = member.split("=")
            enum_members.append({
                "name": name_value[0],
                "value": name_value[1]
            })
        else:
            enum_members.append({
                "name": member,
                "value": enum_value
            })
            enum_value += 1
    return enum_members


# get all struct member names, types, defaults and other metadata
def get_struct_members(declaration):
    members = []
    pos = declaration.find("{")+1
    while pos != -1:
        semantic_pos = declaration.find(":", pos)
        end_pos = declaration.find(";", pos)
        if end_pos == -1:
            break
        skip = end_pos
        if us(semantic_pos) < end_pos:
            end_pos = semantic_pos
        bracket_pos = declaration.find("{", pos)
        start_pos = pos
        if us(bracket_pos) < end_pos:
            end_pos = enclose("{", "}", declaration, start_pos)
        statement = declaration[start_pos:end_pos]
        member_type = "variable"
        if statement.find("(") != -1 and statement.find("=") == -1:
            member_type = "function"
        attrubutes = None
        attr_start = statement.find("[[")
        if attr_start != -1:
            attr_end = statement.find("]]")
            attrubutes = statement[attr_start+2:attr_end]
        decl = statement.strip()
        type_decl = breakdown_type_decl(decl)
        semantic = None
        if semantic_pos != -1:
            semantic = declaration[semantic_pos:skip].strip().strip(";").strip(":").strip()
        members.append({
            "member_type": member_type,
            "data_type": type_decl["type"].strip(),
            "name": type_decl["name"].strip(),
            "default": type_decl["default"],
            "declaration": decl,
            "attributes": attrubutes,
            "semantic": semantic
        })
        pos = skip + 1
    return members


def get_members(type_specifier, declaration):
    lookup = {
        "enum": get_enum_members,
        "struct": get_struct_members,
        "cbuffer": get_struct_members
    }
    if type_specifier in lookup:
        return lookup[type_specifier](declaration)
    return []


# finds the fully qualified scope for a type declaration
def get_type_declaration_scope(source, type_pos):
    scope_identifier = [
        "namespace",
        "struct",
        "class"
    ]
    pos = 0
    scopes = []
    while True:
        scope_start, i = find_first(source, scope_identifier, pos)
        if scope_start != -1:
            scp = source.find(";", scope_start)
            pp = source.find("{", scope_start)
            if us(pp) > scp:
                if scp == -1:
                    return scopes
                pos = scp
                continue
            scope_end = enclose("{", "}", source, scope_start)
            if scope_end > type_pos > scope_start:
                scope_name = type_name(source[scope_start:scope_end])
                scopes.append({
                    "type": i,
                    "name": scope_name
                })
                pos = source.find("{", scope_start) + 1
            else:
                pos = scope_end
        else:
            return scopes
        if pos > type_pos:
            return scopes
    return []


# return list of any typedefs for a particular type
def find_typedefs(fully_qualified_name, source):
    pos = 0
    typedefs = []
    typedef_names = []
    while True:
        start_pos = find_token("typedef", source[pos:])
        if start_pos != -1:
            start_pos += pos
            end_pos = start_pos + source[start_pos:].find(";")
            typedef = source[start_pos:end_pos]
            q = find_token(fully_qualified_name, typedef)
            if q != -1:
                typedefs.append(source[start_pos:end_pos])
                name = typedef[q+len(fully_qualified_name):end_pos].strip()
                typedef_names.append(name)
            pos = end_pos
        else:
            break
    return typedefs, typedef_names


def find_type_attributes(source, type_pos):
    delimiters = [";", "}"]
    attr = source[:type_pos].rfind("[[")
    first_d = us(-1)
    for d in delimiters:
        first_d = min(us(source[:type_pos].rfind(d)), first_d)
    if first_d == us(-1):
        first_d = -1
    if attr > first_d:
        attr_end = source[attr:].find("]]")
        return source[attr+2:attr+attr_end]
    return None


# extract the type of the template parameter ie. StructuredBuffer<float4> (template_type = float4)
def get_template_type(type_decl):
    start, end = enclose_start_end("<", ">", type_decl, 0)
    if start != -1 and end != -1:
        return type_decl[start:end]
    return None
    

# finds all type declarations.. ie struct, enum. returning them in dict with name, and code
def find_type_declarations(type_specifier, source):
    results = []
    names = []
    pos = 0
    while True:
        start_pos = find_token(type_specifier, source[pos:])
        if start_pos != -1:
            start_pos += pos
            # handle forward decl
            fp, tok = find_first(source, ["{", ";"], start_pos)
            forward = False
            members = []
            if tok == ";":
                declaration = source[start_pos:fp]
                forward = True
                end_pos = fp
            else:
                end_pos = enclose("{", "}", source, start_pos)
                declaration = source[start_pos:end_pos]
                members = get_members(type_specifier, declaration)
            scope = get_type_declaration_scope(source, start_pos)
            name = type_name(declaration)
            name, array_size = type_name_array(name)
            qualified_name = ""
            for s in scope:
                if s["type"] == "namespace":
                    qualified_name += s["name"] + "::"
            qualified_name += name
            typedefs, typedef_names = find_typedefs(qualified_name, source)
            attributes = find_type_attributes(source, start_pos)
            decl = declaration.strip()
            results.append({
                "type": type_specifier,
                "name": name,
                "qualified_name": qualified_name,
                "declaration": declaration.strip(),
                "position": pos,
                "members": members,
                "scope": scope,
                "typedefs": typedefs,
                "typedef_names": typedef_names,
                "attributes": attributes,
                "forward_declaration": forward,
                "template_type": get_template_type(decl),
                "array_size": array_size
            })
            pos = end_pos+1
        else:
            break
    for r in results:
        names.append(r["name"])
        names.append(r["qualified_name"])
        for name in r["typedef_names"]:
            names.append(name)
    return results, names


# find include statements
def find_include_statements(source):
    includes = []
    for line in source.splitlines():
        if line.strip().startswith("#include"):
            includes.append(line)
    return includes


# finds the next token ignoring white space
def next_token(source, start):
    white_space = [" ", "\n"]
    pos = start+1
    while True:
        if source[pos] in white_space:
            pos += 1
        else:
            return source[pos]
        if pos >= len(source):
            break
    return None


# find first string
def find_first(source, tokens, start):
    first = sys.maxsize
    first_token = ""
    for t in tokens:
        i = source.find(t, start)
        if first > i > -1:
            first = i
            first_token = t
    return first, first_token


# find first token
def find_first_token(source, tokens, start):
    first = sys.maxsize
    first_token = ""
    for t in tokens:
        i = find_token(t, source)
        if first > i > -1:
            first = i
            first_token = t
    return first, first_token


def arg_list(args):
    args = args.replace("\n", " ")
    args = re.sub(' +', ' ', args)
    pos = 0
    a = []
    while True:
        # find comma separators
        cp = args.find(",", pos)
        ep = args.find("=", pos)
        if cp == -1:
            # add final arg
            aa = args[pos:].strip()
            if len(aa) > 0:
                a.append(args[pos:].strip())
            break
        if -1 < ep < cp:
            # handle function with default
            end, tok = find_first(args, [",", "{", "("], ep)
            if tok == ",":
                a.append(args[pos:end].strip())
                pos = end+1
                continue
            elif tok == "(":
                end = enclose(tok, ")", args, end)
            else:
                end = enclose(tok, "}", args, end)
            a.append(args[pos:end])
            end = args.find(",", end)
            if end == -1:
                end = len(args)
            pos = end+1
        else:
            # plain arg
            a.append(args[pos:cp].strip())
            pos = cp+1
    return a


# breakdown a type decl ie. StructName variable_name = {}, into data_type: StructName, name: variable_name, default = {}
def breakdown_type_decl(a):
    decl = a
    # extract default
    dp = a.find("=")
    default = None
    if dp != -1:
        default = a[dp + 1:].strip()
        decl = a[:dp - 1]
        sp = decl.find(":")
    # extract semantic
    sp = decl.find(":")
    semantic = None
    if sp != -1:
        semantic = decl[sp + 1:].strip()
        decl = decl[:sp - 1]
    name_pos = decl.rfind(" ")
    name_pos = decl.rfind(" ")
    return {
        "type": decl[:name_pos].strip(),
        "name": decl[name_pos:].strip(),
        "default": default,
        "semantic": semantic
    }


# break down arg decl (int a, int b, int c = 0) into contextual info
def breakdown_function_args(args):
    args = arg_list(args)
    args_context = []
    for a in args:
        a = a.strip()
        if len(a) == "":
            continue
        if a == "...":
            # va list
            args_context.append({
                "type": "...",
                "name": "va_list",
                "default": None,
                "semantic": None
            })
        else:
            args_context.append(breakdown_type_decl(a))
    return args_context


# parse return type of function and split out any template or inline
def parse_return_type(statement):
    template = None
    inline = None
    rt = statement.strip("}")
    rt = rt.strip()
    tp = rt.find("template")
    if tp != -1:
        etp = enclose("<", ">", rt, tp)
        template = rt[tp:etp]
        rt = rt[etp:]
    ip = rt.find("inline")
    if ip != -1:
        ipe = ip+len("inline")
        inline = rt[:ipe]
        rt = rt[ipe:]
    return rt, template, inline


# prepends attributes to soiurce coe of functions, structs etc
def combine_src_attributes(attribs, src):
    if len(attribs) == 0:
        return src
    out = "["
    for attrib in attribs:
        out += attrib
    out += "]\n" + src
    return out


# find functions
def find_functions(source):
    # look for parenthesis to identiy function decls
    functions = []
    function_names = []
    pos = 0
    attributes = []
    while True:
        statement_end, statement_token = find_first(source, [";", "{"], pos)
        if statement_end == -1:
            break
        statement = source[pos:statement_end].strip()
        src_start = pos
        pp = statement.find("(")
        ep = statement.find("=")
        skip = statement_end+1
        
        # check for attributes
        if len(statement) > 0:
            if statement[0] == "[":
                start, end = enclose_start_end("[", "]", statement, 0)
                attributes.append(statement[start:end])
                pos += start + end + 1
                continue

        if pp != -1:
            next = next_token(statement, pp)
            if (ep == -1 or pp < ep) and next != "*":
                # this a function decl, so break down into context
                body = ""
                body_end = skip
                if statement_token == "{":
                    body_end = enclose("{", "}", source, statement_end-1)
                    body = source[statement_end-1:body_end]
                    statement_end = body_end+1
                    skip = body_end
                args_end = enclose("(", ")", statement, pp)-1
                name_pos = statement[:pp].rfind(" ")
                name = statement[name_pos+1:pp].strip()
                name_unscoped = name.rfind(":")
                qualifier = ""
                if name_unscoped != -1:
                    qualifier = name[:name_unscoped-1]
                    name = name[name_unscoped+1:]
                return_type, template, inline = parse_return_type(statement[:name_pos])
                args = breakdown_function_args(statement[pp+1:args_end])
                scope = get_type_declaration_scope(source, pos)
                functions.append({
                    "name": name,
                    "qualifier": qualifier,
                    "return_type": return_type,
                    "args": args,
                    "scope": scope,
                    "body": body,
                    "template": template,
                    "inline": inline,
                    "source": combine_src_attributes(attributes, source[src_start:body_end].strip()).strip(),
                    "attributes": list(attributes)
                })
                attributes.clear()
                function_names.append(name)
        pos = skip
        if pos > len(source):
            break
    return functions, function_names


# returns prototype with no names, ie (int, int, float), from a function context
def get_funtion_prototype(func):
    args = ""
    num_args = len(func["args"])
    for a in range(0, num_args):
        args += func["args"][a]["type"]
        if a < num_args - 1:
            args += ", "
    if num_args == 0:
        args = "void"
    return "(" + args + ")"


# main function for scope
def test():
    # read source from file
    source = open("test.h", "r").read()

    # sanitize source to make further ops simpler
    source = sanitize_source(source)
    print("--------------------------------------------------------------------------------")
    print("sanitize source ----------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")
    print(source)

    # find all include statements, fromsanitized source to ignore commented out ones
    includes = find_include_statements(source)
    print("--------------------------------------------------------------------------------")
    print("find includes ------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")
    print(includes)

    # find string literals within source
    print("--------------------------------------------------------------------------------")
    print("find strings literals ----------------------------------------------------------")
    print("--------------------------------------------------------------------------------")
    strings = find_string_literals(source)
    print(strings)

    # remove string literals to avoid conflicts when parsing
    print("--------------------------------------------------------------------------------")
    print("placeholder literals -----------------------------------------------------------")
    print("--------------------------------------------------------------------------------")
    strings, source = placeholder_string_literals(source)
    print(format_source(source, 4))

    # find single token
    print("--------------------------------------------------------------------------------")
    print("find token ---------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")
    token = "SOME_TOKEN"
    token_pos = find_token(token, source)
    print("token pos: {}".format(token_pos))
    print("token:" + source[token_pos:token_pos+len(token)])

    print("--------------------------------------------------------------------------------")
    print("find all tokens ----------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")
    token = "int"
    token_locations = find_all_tokens(token, source)
    for loc in token_locations:
        print("{}: ".format(loc) + source[loc:loc+10] + "...")

    # find structs
    print("--------------------------------------------------------------------------------")
    print("find structs -------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")
    structs, struct_names = find_type_declarations("struct", source)
    print(struct_names)
    print(json.dumps(structs, indent=4))

    # find enums
    print("--------------------------------------------------------------------------------")
    print("find enums ---------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")
    enums, enum_names = find_type_declarations("enum", source)
    print(enum_names)
    print(json.dumps(enums, indent=4))

    # find free functions
    print("--------------------------------------------------------------------------------")
    print("find functions -----------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")
    functions, function_names = find_functions(source)
    print(function_names)
    print(json.dumps(functions, indent=4))

    # replace placeholder literals
    print("--------------------------------------------------------------------------------")
    print("replace placeholder literals ---------------------------------------------------")
    print("--------------------------------------------------------------------------------")
    source = replace_placeholder_string_literals(strings, source)
    print(format_source(source, 4))

    # display names
    print("--------------------------------------------------------------------------------")
    print(" display name ------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")
    print(display_name("m_snake_case_variable", False))
    print(display_name("m_camelCaseVariable", False))
    print(display_name("SCAREY_CASE_DEFINE", True))


# entry
if __name__ == "__main__":
    test()
