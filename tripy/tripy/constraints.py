#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import inspect
from typing import Union, Optional, get_origin, get_args, ForwardRef
from collections import namedtuple

TYPE_VERIFICATION = {}
FUNC_W_DOC_VERIF = []
RETURN_VALUE = "RETURN_VALUE"

class InputValues:
    dtype: str = None
    init = None


def dtype_info(
    dtype_variables: dict = {},
    dtype_constraints: dict = {},
    function_name : Optional[str] = "",
):
    """
    This function is a decorator that populates TYPE_VERIFICATION global dictionary which will be used by
    test_dtype_constraints.py to verify the dtypes of all operations.

    Args:
        dtype_variables: This input must be a dictionary with the names of groups of variables as the keys and lists of datatypes as the values.
            Example: dtype_variables={"T": ["float32", "float16", "int8", "int32", "int64", "bool"], "T1": ["int32"]}. 
            Any datatype not included will be tested to ensure it fails the test cases.
        dtype_constraints: This input assigns inputs and return parameters to variable groups. 
            It must be a dictionary with parameter names as keys and variable group names as values. 
            For assigning the return value, the key must be constraints.RETURN_VALUE. 
            Example: dtype_constraints={"input": "T", "index": "T1", constraints.RETURN_VALUE: "T"}.
        function_name: This parameter is only needed if a function is being mapped to multiple APIs. Takes a string with the function name as input.
    """
    def decorator(func_obj):
        # Get names and type hints for each param.
        func_sig = inspect.signature(func_obj)
        param_dict = func_sig.parameters
        # Construct inputs_dict.
        inputs_dict = {}
        for param_name, param_type in param_dict.items():
            input_values = InputValues()
            # If parameter had a default then use it otherwise skip.
            if param_type.default is not param_type.empty:
                # Checking if not equal to None since we want to enter if default is 0 or similar.
                if param_type.default != None:
                    input_values.init = param_type.default
            param_type = param_type.annotation
            # If type is an optional or union get the first type.
            while get_origin(param_type) in [Union, Optional]:
                param_type = get_args(param_type)[0]
                # ForwardRef refers to any case where type hint is a string.
                if isinstance(param_type, ForwardRef):
                    param_type = param_type.__forward_arg__
            # Check if there are any provided constraints and add them to the constraint dict.
            input_values.dtype = dtype_constraints.get(param_name, None)
            inputs_dict[param_name] = {param_type: input_values}
        return_dtype = dtype_constraints.get(RETURN_VALUE, -1)
        func_name = func_obj.__qualname__ if not function_name else function_name
        VerifInfo = namedtuple('VerifInfo', ['obj', 'inputs', 'return_dtype', 'dtypes', 'dtype_constraints'])
        TYPE_VERIFICATION[func_name] = VerifInfo(func_obj, inputs_dict, return_dtype, dtype_variables, dtype_constraints)
        FUNC_W_DOC_VERIF.append(func_obj.__qualname__)
        return func_obj

    return decorator