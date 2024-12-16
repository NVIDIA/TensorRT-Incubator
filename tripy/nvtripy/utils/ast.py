#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import ast
from typing import List, Optional, Tuple, Set

from nvtripy.utils.result import Result
from nvtripy.utils.stack_info import SourceInfo


def get_parsed_ast(code: str) -> Result[Tuple[str, int]]:
    # Returns the parsed AST and additional indentation that needs to be accounted for
    # when determining column offsets.
    raw_code = code or ""
    code = raw_code.lstrip()
    indentation = len(raw_code) - len(code)

    # In some cases, we may not be able to parse the line of code in isolation.
    # e.g. something like this:
    #   "X": (tp.ones((1, 2, 4, 4)) * 4.0)
    # is invalid on its own, but syntactically correct as an element of a dictionary.
    try:
        parsed_ast = ast.parse(code)
    except Exception as err:
        return Result.err([str(err)])
    return Result.ok((parsed_ast, indentation))


def get_callee_func_name_candidates(callee: SourceInfo) -> Set[str]:
    # Some functions (e.g. tensor methods) are routed through a function registry.
    # We don't actually care about the dispatch function, so we look at the `key`
    # to determine which underlying method we're actually calling.
    if callee._dispatch_target:
        dispatch_target = callee._dispatch_target
        # The function registry may have prepended a class name. If so, strip it out.
        if "." in dispatch_target:
            dispatch_target = dispatch_target.split(".")[-1]
        candidates = {dispatch_target}
    else:
        candidates = {callee.function}

    # Some methods are called by other builtins:
    SPECIAL_METHODS = {
        "__repr__": {"repr", "print"},
        "__str__": {"str"},
        "__int__": {"int"},
        "__bool__": {"bool"},
    }
    candidates.update(SPECIAL_METHODS.get(callee.function, set()))

    return candidates


def get_ast_node_func_name(node) -> Optional[str]:
    # Returns the function name for the given AST node, or None
    # if the node is not a function call or the name could not be determined.
    if isinstance(node, ast.BinOp):
        MAPPING = {
            ast.Add: "__add__",
            ast.Sub: "__sub__",
            ast.Mult: "__mul__",
            ast.Div: "__truediv__",
            ast.FloorDiv: "__floordiv__",
            ast.Mod: "__mod__",
            ast.Pow: "__pow__",
            ast.BitAnd: "__and__",
            ast.BitOr: "__or__",
            ast.BitXor: "__xor__",
            ast.LShift: "__lshift__",
            ast.RShift: "__rshift__",
        }
        return MAPPING.get(type(node.op))

    if isinstance(node, ast.Subscript):
        return "__getitem__"

    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Call):
            return get_ast_node_func_name(func)

        if isinstance(func, ast.Attribute):
            return func.attr

        return func.id
    return None


# Gets the column offset of the argument at `index` to function called `func_name` in the provided `code` snippet.
def get_arg_candidate_column_offsets(
    code: str, index: int, num_positional: int, func_name: str, is_kwarg: bool, arg_names: List[str]
) -> Tuple[int, int]:

    candidates = []

    result = get_parsed_ast(code)
    if not result:
        return candidates
    parsed_ast, indentation = result.value
    for node in ast.walk(parsed_ast):
        if get_ast_node_func_name(node) != func_name:
            continue

        arg_node = None
        if isinstance(node, ast.BinOp):
            assert index < 2
            arg_node = node.left if index == 0 else node.right
        elif isinstance(node, ast.Call):
            if is_kwarg:
                arg_node = node.keywords[index - num_positional]
            else:
                # For methods, the `self` argument is omitted from ast.Call.args
                if "self" in arg_names:
                    index -= 1
                # If the final argument is a starred object, then we treat any args
                # past the end as pointing to the starred object (this would be a variadic call,
                # and the starred object would be a catchall)
                if index >= len(node.args) and isinstance(node.args[-1], ast.Starred):
                    arg_node = node.args[-1]
                else:
                    arg_node = node.args[index]

        elif isinstance(node, ast.Subscript):
            # For slices, index into the fields if they are specified.
            # If it's not a slice or it's absent, it's best to indicate the whole expr
            def index_into_expr(node: ast.expr, index: int) -> ast.expr:
                if isinstance(node, ast.Slice):
                    if index == 0 and node.lower is not None:
                        return node.lower
                    elif index == 1 and node.upper is not None:
                        return node.upper
                    elif index == 2 and node.step is not None:
                        return node.step
                return node

            # If we have multiple dimensions specified, then we have a tuple of slices.
            # NOTE: We subtract num_positional from the index because the slice arguments would
            # be passed as *variadic arguments* to slice_helper and so would come after the positional argument
            if isinstance(node.slice, ast.Tuple):
                element = node.slice.elts[(index - num_positional) // 3]
                arg_node = index_into_expr(element, (index - num_positional) % 3)
            else:
                arg_node = index_into_expr(node.slice, (index - num_positional))

        if arg_node is not None:
            candidates.append((indentation + arg_node.col_offset, indentation + arg_node.end_col_offset))

    return candidates


# Grab column offsets for a given frame based on information from its callee.
# This method is not perfect and is not required for Python 3.11+, where frames include column offsets.
def get_candidate_column_offsets(cur_frame: SourceInfo, callee: SourceInfo) -> List[Tuple[int, int]]:
    candidate_callee_names = get_callee_func_name_candidates(callee)

    candidate_column_offsets = []

    result = get_parsed_ast(cur_frame.code)
    if not result:
        return candidate_column_offsets
    parsed_ast, indentation = result.value

    for node in ast.walk(parsed_ast):

        try:
            ast_node_name = get_ast_node_func_name(node)
        except:
            continue

        if ast_node_name is None:
            continue

        def check_name_matches():
            # We need special checking for __init__ methods since the AST node will just be the class name, e.g. `Tensor`.
            if "__init__" not in candidate_callee_names:
                return ast_node_name in candidate_callee_names

            # We hardcode names of some common classes here to avoid creating an import dependency:
            if ast_node_name in {"Tensor"}:
                return True
            return False

        # Since there could be multiple different function calls on the same line, we use the callee name
        # to determine which one(s) to look at.
        if check_name_matches():
            candidate_column_offsets.append((indentation + node.col_offset, indentation + node.end_col_offset))

    return candidate_column_offsets
