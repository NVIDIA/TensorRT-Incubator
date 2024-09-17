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

from typing import List, Dict, Tuple, Any
from collections import namedtuple
import functools
from tripy import utils
from tripy.common.exception import raise_error
from tripy import config

TYPE_VERIFICATION = {}
RETURN_VALUE = "RETURN_VALUE"


def dtype_info(
    dtype_variables: dict = {},
    dtype_constraints: dict = {},
    dtype_exceptions: dict = [],
    aliases: List[str] = [],
):
    """
    This function is a decorator that populates TYPE_VERIFICATION global dictionary which will be used by
    test_dtype_constraints.py to verify the dtypes of all operations.

    **IMPORTANT: This should be applied before the `convert_inputs_to_tensors` decorator (i.e. must follow it in the code)**
        **to make type checking work reliably. This ensures that the inputs coming in to the wrapped functions are Tensors**

    Args:
        dtype_variables: This input must be a dictionary with the names of groups of variables as the keys and lists of datatypes as the values.
            Example: dtype_variables={"T": ["float32", "float16", "int8", "int32", "int64", "bool"], "T1": ["int32"]}.
            Any datatype not included will be tested to ensure it fails the test cases.
        dtype_constraints: This input assigns inputs and return parameters to variable groups.
            It must be a dictionary with parameter names as keys and variable group names as values.
            For assigning the return value, the key must be constraints.RETURN_VALUE.
            Example: dtype_constraints={"input": "T", "index": "T1", constraints.RETURN_VALUE: "T"}.
        aliases: A list of function name aliases. For methods that are exposed as multiple APIs (e.g. __add__ and __radd__), this
            will enable type information to be added to the documentation for the aliases as well.
    """

    def decorator(func):
        return_dtype = dtype_constraints.get(RETURN_VALUE, None)
        VerifInfo = namedtuple(
            "VerifInfo", ["obj", "inputs", "dtype_exceptions", "return_dtype", "dtypes", "dtype_constraints"]
        )
        verif_info = VerifInfo(func, {}, dtype_exceptions, return_dtype, dtype_variables, dtype_constraints)

        for key in [func.__qualname__] + aliases:
            TYPE_VERIFICATION[key] = verif_info

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if config.enable_dtype_checking:
                from tripy.frontend.tensor import Tensor
                from tripy.common.datatype import dtype

                merged_args = utils.merge_function_arguments(func, *args, **kwargs)

                # The first arguments seen for each type variable. Other arguments with the same variable
                # must use the same data types.
                type_var_first_args: Dict[str, Tuple[str, dtype, Any]] = {}

                for name, arg in merged_args:
                    if name in dtype_constraints:
                        type_var = dtype_constraints[name]

                        if isinstance(arg, Tensor):
                            arg_dtype = arg.dtype
                        elif isinstance(arg, dtype):
                            arg_dtype = arg
                        else:
                            continue

                        # Check if the type is supported at all
                        supported_dtypes = dtype_variables[type_var]
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
