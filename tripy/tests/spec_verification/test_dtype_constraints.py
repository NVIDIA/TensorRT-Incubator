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

from typing import List
from tripy.common.datatype import DATA_TYPES
import itertools
import pytest
from tests.spec_verification.object_builders import create_obj
from tripy.tripy.constraints import TYPE_VERIFICATION, RETURN_VALUE
import tripy as tp


def _method_handler(func_name, kwargs, func_obj, api_call_locals):
    _METHOD_OPS = {
        "__add__": (lambda self, other: self + other),
        "__sub__": (lambda self, other: self - other),
        "__rsub__": (lambda self, other: self - other),
        "__pow__": (lambda self, other: self**other),
        "__rpow__": (lambda self, other: self**other),
        "__mul__": (lambda self, other: self * other),
        "__rmul__": (lambda self, other: self * other),
        "__truediv__": (lambda self, other: self / other),
        "__rtruediv__": (lambda self, other: self / other),
        "__lt__": (lambda self, other: self < other),
        "__le__": (lambda self, other: self <= other),
        "__eq__": (lambda self, other: self == other),
        "__ne__": (lambda self, other: self != other),
        "__ge__": (lambda self, other: self >= other),
        "__gt__": (lambda self, other: self > other),
        "__matmul__": (lambda self, other: self @ other),
        "shape": (lambda self: self.shape),
        "__getitem__": (lambda self, index: self[index]),
    }
    if func_obj.__name__ in _METHOD_OPS.keys():
        # Function needs to be executed in a specific way
        rtn_builder = _METHOD_OPS.get(func_obj.__name__, None)
        api_call_locals[RETURN_VALUE] = rtn_builder(**kwargs)
    else:
        # Execute API call normally.
        exec(f"{RETURN_VALUE} = " + f"tp.{func_obj.__name__}(**kwargs)", globals(), api_call_locals)
    # Print out debugging info.
    print("API call: ", func_name, ", with parameters: ", kwargs)



'''
default_constraints: This dictionary helps set specific constraints and values for parameters. These constraints correspond to the type hint of each parameter. 
Each type has different constraints that can be set, and some have default values, so you might not need to pass other_constraints for every operation. 
If there is no default, you must specify an initialization value, or the testcase may fail. 
The dictionary's keys must be the name of the function that they are constraining and the value must be a dictionary with the constraints. Here is the list of possible parameter types and constraints:
    - **tensor** - constraints: init and shape default: tp.ones(shape=(3,2)). If init is passed then value must be in the form of a list. Example: "scale": {"init": [1, 1, 1]} or "scale": {"shape": (3,3)}
    - **int** - constraints: init default: **no default**. Example: "dim": {"init": 0}.
    - **dtype** - constraints: **no constraints** default: **no default**. Dtype parameters will be set using dtype_constraints input.
    - **tuple** - constraints: init default: **no default**. Example: "dims": {"init": (3,3)}. 
    - **list/sequence of tensors** - constraints: count, init, and shape default: count=2, shape=(3,2). Example: "dim": {"count": 3}. No default means that you must specify an initialization value or an error will be thrown. 
        This will create a list/sequence of tensors of size count and each tensor will follow the init and shape value similar to tensor parameters.
    - **device** - constraints: target default: target="gpu". Example: {"device": {"target": "cpu"}}.
    - **int list** - constraints: init default: **no default**. Example: "dim": {"init": [1, 2, 3]}.
    - **bool** - constraints: init default: **no default**. Example: "dim": {"init": True}. 
    - **float** - constraints: init default: **no default**. Example: "dim": {"init": 1.23}
'''
default_constraints_all = {"__rtruediv__": {"self": {"init": 1}},
                           "__rsub__": {"self": {"init": 1}},
                           "__radd__": {"self": {"init": 1}},
                           "__rpow__": {"self": {"init": 1}},
                           "__rmul__": {"self": {"init": 1}},
                           "softmax": {"dim": {"init": 1}},
                           "concatenate": {"dim": {"init": 0}},
                           "expand": {"sizes": {"init": (3, 4)}, "input": {"shape": (3, 1)}},
                           "full": {"shape": {"shape": (3)}, "value": {"init": 1}},
                           "full_like": {"value": {"init": 1}},
                           "flip": {"dim": {"init": 1}},
                           "gather": {"dim": {"init": 0}, "index": {"shape": (1)}},
                           "iota": {"shape": {"shape": (3)}},
                           "__matmul__": {"self": {"shape": (2, 3)}},
                           "transpose": {"dim0": {"init": 0}, "dim1": {"init": 1}},
                           "permute": {"perm": {"init": (1, 0)}},
                           "quantize": {"scale": {"init": [1, 1, 1]}, "dim": {"init": 0}},
                           "sum": {"dim": {"init": 0}},
                           "all":{"dim": {"init": 0}},
                           "any": {"dim": {"init": 0}},
                           "max": {"dim": {"init": 0}},
                           "prod": {"dim": {"init": 0}},
                           "mean": {"dim": {"init": 0}},
                           "var": {"dim": {"init": 0}},
                           "argmax": {"dim": {"init": 0}},
                           "argmin": {"dim": {"init": 0}},
                           "reshape": {"shape": {"init": (6,)}},
                           "squeeze": {"input": {"shape": (3, 1)}, "dims": {"init": (1)}},
                           "__getitem__": {"index": {"init": 2}},
                           "split": {"indices_or_sections": {"init": 2}},
                           "unsqueeze": {"dim": {"init": 1}},
                           "masked_fill": {"value": {"init": 1}},
                           "ones": {"shape": {"init": (3,2)}},
                           "zeros": {"shape": {"init": (3,2)}},
                           "arange": {"start": {"init": 0}, "stop": {"init": 5}},
                          }

# Add default_constraints to input_values within TYPE_VERIFICATION
for func_name, (func_obj, input_dict, _, _, types_assignments) in TYPE_VERIFICATION.items():
    default_constraints = default_constraints_all.get(func_name, None)
    if default_constraints != None:
        for param_name, input_info in input_dict.items():
            input_values = list(input_dict[param_name].values())[0]
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

func_list = []
for func_name, (func_obj, inputs, return_dtype, types, types_assignments) in TYPE_VERIFICATION.items():
    # Issue #268 exclude float8 until casting to float8 gets fixed.
    # Issue #268 exclude int4 until int4 is representable.
    types_to_exclude = ["int4", "float8"]
    positive_test_dtypes = {}
    negative_test_dtypes = {}
    for name, dt in types.items():
        # Get all of the dtypes for positive test case by excluding types_to_exclude.
        positive_test_dtypes[name] = list(filter(lambda item: item not in types_to_exclude, set(dt)))
        # Get all dtypes for negative test case.
        total_dtypes = set(filter(lambda item: item not in types_to_exclude, map(str, DATA_TYPES.values())))
        pos_dtypes = set(dt)
        negative_test_dtypes[name] = list(total_dtypes - pos_dtypes)
    for positive_case in [True, False]:
        if positive_case:
            dtype_lists_list = [positive_test_dtypes]
            case_name = "valid: "
        else:
            dtype_lists_list = []
            all_dtypes = set(
                            filter(
                                lambda item: item not in types_to_exclude,
                                map(str, DATA_TYPES.values()),
                            )
                        )
            # Create a list of dictionary lists and then go over each dictionary.
            for name_temp, dt in negative_test_dtypes.items():
                temp_dict = {name_temp: dt}
                # Iterate through and leave one dtype set to negative and the rest is all dtypes.
                for name_not_equal in negative_test_dtypes.keys():
                    if name_temp != name_not_equal:
                        temp_dict[name_not_equal] = all_dtypes
                dtype_lists_list.append(temp_dict)
            case_name = "invalid: "
        for dtype_lists in dtype_lists_list:
            for combination in itertools.product(*(dtype_lists.values())):
                # Create a tuple with keys and corresponding elements.
                namespace = dict(zip(dtype_lists.keys(), map(lambda val: getattr(tp, val), combination)))
                ids = [f"{dtype_name}={dtype}" for dtype_name, dtype in namespace.items()]
                func_list.append(
                    (
                        func_name,
                        func_obj,
                        inputs,
                        return_dtype,
                        namespace,
                        positive_case,
                        func_name + "_" + case_name + ", ".join(ids),
                    )
                )


def _run_dtype_constraints_subtest(test_data):
    func_name, func_obj, inputs, _, namespace, _, _ = test_data
    kwargs = {}
    # Create all input objects using object_builders.create_obj.
    for param_name, input_desc in inputs.items():
        kwargs[param_name] = create_obj(param_name, input_desc, namespace)
    # Run api call through _method_handler and setup namespace (api_call_locals).
    api_call_locals = {"kwargs": kwargs}
    _method_handler(func_name, kwargs, func_obj, api_call_locals)
    # If output does not have dtype skip .eval().
    if isinstance(api_call_locals[RETURN_VALUE], int): 
        return api_call_locals, namespace
    # If output is a list then checking the return the first element in the list. (Assumes list of Tensors)
    if isinstance(api_call_locals[RETURN_VALUE], List): 
        api_call_locals[RETURN_VALUE] = api_call_locals[RETURN_VALUE][0]
    # Run eval to check for any backend errors.
    api_call_locals[RETURN_VALUE].eval()
    return api_call_locals, namespace


@pytest.mark.parametrize("test_data", func_list, ids=lambda val: val[6])
def test_dtype_constraints(test_data):
    _, _, _, return_dtype, _, positive_case, _ = test_data
    if positive_case:
        api_call_locals, namespace = _run_dtype_constraints_subtest(test_data)
        if isinstance(api_call_locals[RETURN_VALUE], int): 
                return
        else:
            assert api_call_locals[RETURN_VALUE].dtype == namespace[return_dtype]
    else:
        with pytest.raises(Exception):
            api_call_locals, namespace = _run_dtype_constraints_subtest(test_data)
            if isinstance(api_call_locals[RETURN_VALUE], int): 
                return
            else:
                assert api_call_locals[RETURN_VALUE].dtype == namespace[return_dtype]
