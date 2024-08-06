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

TYPE_VERIFICATION = {}
RETURN_VALUE = "RETURN_VALUE"

class InputValues:
    dtype: str = None
    init: Union[tuple,list] = None
    shape: tuple = None
    target: str = None
    count: int = None

def dtype_info(
    dtype_variables: dict = {},
    dtype_constraints: dict = {},
    default_constraints: dict = {},
    param_type_specification: dict = {},
    function_name : Optional[str] = "",
):
    """
    This function is a decorator that populates TYPE_VERIFICATION global dictionary which will be used by
    test_dtype_constraints.py to verify the dtypes of all operations.
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
                if not param_type.default == None:
                    input_values.init = param_type.default
            param_type = param_type.annotation
            # Check if there is a specific type that should be used.
            if param_type_specification.get(param_name, None):
                param_type = param_type_specification[param_name]
            # If type is an optional or union get the first type.
            while get_origin(param_type) in [Union, Optional]:
                param_type = get_args(param_type)[0]
                # ForwardRef refers to any case where type hint is a string.
                if isinstance(param_type, ForwardRef):
                    param_type = param_type.__forward_arg__
            # Check if there are any provided constraints and add them to the constraint dict.
            
            input_values.dtype = dtype_constraints.get(param_name, None)
            other_constraint = default_constraints.get(param_name, None)
            if other_constraint:
                for key, val in other_constraint.items():
                    if key == "init":
                        input_values.init = val
                    elif key == "shape":
                        input_values.shape = val
                    elif key == "target":
                        input_values.target = val
                    elif key == "count":
                        input_values.count = val
                    else:
                        raise RuntimeError(f"Could not match key for default_constraints. Key was {key}, value was {val}")
            inputs_dict[param_name] = {param_type: input_values}
        return_dtype = dtype_constraints.get(RETURN_VALUE, -1)
        parsed_dict = {"inputs": inputs_dict, "return_dtype": return_dtype, "types": dtype_variables}
        func_name = func_obj.__qualname__ if not function_name else function_name
        TYPE_VERIFICATION[func_name] = (func_obj, parsed_dict, dtype_constraints)
        return func_obj

    return decorator
