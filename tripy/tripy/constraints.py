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

import functools
from collections import namedtuple
from typing import Any, Dict, List, Tuple, Sequence

from tripy import config, utils
from tripy.common.exception import raise_error

TYPE_VERIFICATION = {}
RETURN_VALUE = "__RETURN_VALUE"


def get_arg_dtype(arg, func_name, arg_name) -> utils.Result["tripy.dtype"]:
    from tripy.common.datatype import dtype
    from tripy.frontend.tensor import Tensor

    if isinstance(arg, Sequence):
        arg_dtypes = []
        for elem in arg:
            dtype_result = get_arg_dtype(elem, func_name, arg_name)
            if not dtype_result:
                return utils.Result.err(
                    [f"Could not determine data type of elements in sequence: {arg_name}"] + dtype_result.error_details
                )
            arg_dtypes.append(dtype_result.value)

        if len(set(arg_dtypes)) != 1:
            return utils.Result.err(
                [
                    f"Mismatched data types in sequence argument for '{func_name}'.\n",
                    f"For parameter: '{arg_name}', all arguments must have the same data type, but got: "
                    f"{arg_dtypes}",
                ],
            )
        arg_dtype = arg_dtypes[0]
    elif isinstance(arg, Tensor):
        arg_dtype = arg.dtype
    elif isinstance(arg, dtype):
        arg_dtype = arg
    else:
        return utils.Result.err([f"Expected a tensor or data type argument for {arg_name}, but got: {arg}"])
    return utils.Result.ok(arg_dtype)


def dtypes(
    constraints: Dict[str, str] = {},
    variables: Dict[str, List[str]] = {},
    exceptions: List[Dict[str, str]] = [],
    aliases: List[str] = [],
):
    """
    Specifies data type constraints for the decorated function.

    **IMPORTANT: This should be applied before the `convert_to_tensors` (i.e. must follow it in the code) if the signature contains `TensorLike`s.**
        **to make type checking work reliably. This ensures that the inputs coming in to the wrapped functions are Tensors.**
        **To avoid pitfalls, you can make this the innermost decorator.**

    NOTE: When annotating a new API, you should also update `tests/constraints/object_builders.py`.

    Args:
        constraints: Maps parameters and return values to data type constraint variables.
            Use the special value `constraints.RETURN_VALUE` to denote return values.
            For example:
                {"input": "T1", "other": T2, constraints.RETURN_VALUE: "T1"}
        variables: Maps data type constraints variables to their supported data types.
            For example:
                {"T1": ["float32", "float16"], "T2": ["int32", "int64"]}
        exceptions: Indicates specific combinations of data types that are not supported by the API.
            For example:
                [
                    {"T1": "float16", "T2": "int32"},
                ]
        aliases: A list of function name aliases. For methods that are exposed as multiple APIs
            (e.g. __add__ and __radd__), this will enable type information to be added to the
            documentation for the aliases as well.
    """

    def decorator(func):
        return_dtype = constraints.get(RETURN_VALUE, None)
        VerifInfo = namedtuple("VerifInfo", ["obj", "inputs", "exceptions", "return_dtype", "dtypes", "constraints"])
        verif_info = VerifInfo(func, {}, exceptions, return_dtype, variables, constraints)

        for key in [func.__qualname__] + aliases:
            TYPE_VERIFICATION[key] = verif_info

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if config.enable_dtype_checking:
                from tripy.common.datatype import dtype
                from tripy.frontend.tensor import Tensor

                merged_args = utils.merge_function_arguments(func, *args, **kwargs)

                # The first arguments seen for each type variable. Other arguments with the same variable
                # must use the same data types.
                type_var_first_args: Dict[str, Tuple[str, dtype, Any]] = {}

                for name, arg in merged_args:
                    if name not in constraints:
                        continue

                    if arg is None:
                        # This is only possible for ommitted optional arguments. Otherwise, None will
                        # be disallowed by the function registry's type checking.
                        continue

                    type_var = constraints[name]

                    arg_dtype = get_arg_dtype(arg, func.__qualname__, name)
                    if not arg_dtype:
                        raise_error(f"Could not determine datatype of {name}.", arg_dtype.error_details)
                    arg_dtype = arg_dtype.value

                    # Check if the type is supported at all
                    supported_dtypes = variables[type_var]
                    if arg_dtype.name not in supported_dtypes:
                        raise_error(
                            f"Unsupported data type for '{func.__qualname__}'.",
                            [
                                f"For parameter: '{name}', got unsupported data type: '{arg_dtype}'.\n"
                                f"Supported data types are: {supported_dtypes}.\n"
                            ]
                            + (
                                [
                                    f"Note: '{name}' was: ",
                                    arg,
                                ]
                                if isinstance(arg, Tensor)
                                else []
                            ),
                        )

                    # Check if the type matches that of other inputs with the same type_var.
                    if type_var in type_var_first_args:
                        other_name, other_arg_dtype, other_arg = type_var_first_args[type_var]
                        if other_arg_dtype != arg_dtype:
                            raise_error(
                                f"Mismatched data types for '{func.__qualname__}'.",
                                [
                                    f"Parameters: '{other_name}' and '{name}' must have matching data types, but got: "
                                    f"'{other_arg_dtype.name}' and '{arg_dtype.name}' respectively.\n"
                                ]
                                + (
                                    [
                                        f"Note: '{other_name}' was: ",
                                        other_arg,
                                        f"While '{name}' was: ",
                                        arg,
                                    ]
                                    if isinstance(arg, Tensor)
                                    else []
                                ),
                            )

                    type_var_first_args[type_var] = (name, arg_dtype, arg)

            return func(*args, **kwargs)

        return wrapper

    return decorator
