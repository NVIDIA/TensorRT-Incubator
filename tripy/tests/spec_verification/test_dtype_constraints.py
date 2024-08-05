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
from colored import fg, attr
import itertools
import pytest
from tests.spec_verification.object_builders import create_obj
from tripy.dtype_info import TYPE_VERIFICATION, RETURN_VALUE
import tripy as tp


def _method_handler(kwargs, func_obj, api_call_locals):
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
        "__getitem__": (lambda self, index: self.__getitem__(index)),
    }
    if func_obj.__name__ in _METHOD_OPS.keys():
        # Function needs to be executed in a specific way
        rtn_builder = _METHOD_OPS.get(func_obj.__name__, None)
        api_call_locals[RETURN_VALUE] = rtn_builder(**kwargs)
    else:
        # Execute API call normally.
        exec(f"{RETURN_VALUE} = " + f"tp.{func_obj.__name__}(**kwargs)", globals(), api_call_locals)
    # Print out debugging info.
    print("API call: ", func_obj.__name__, ", with parameters: ", kwargs)


func_list = []
for func_obj, parsed_dict, types_assignments in TYPE_VERIFICATION.values():
    inputs = parsed_dict["inputs"]
    return_dtype = parsed_dict["return_dtype"]
    types = parsed_dict["types"]
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
                        func_obj,
                        inputs,
                        return_dtype,
                        namespace,
                        positive_case,
                        func_obj.__name__ + "_" + case_name + ", ".join(ids),
                    )
                )


def _run_dtype_constraints_subtest(test_data):
    func_obj, inputs, _, namespace, _, _ = test_data
    kwargs = {}
    # Create all input objects using object_builders.create_obj.
    for param_name, input_desc in inputs.items():
        kwargs[param_name] = create_obj(param_name, input_desc, namespace)
    # Run api call through _method_handler and setup namespace (api_call_locals).
    api_call_locals = {"kwargs": kwargs}
    _method_handler(kwargs, func_obj, api_call_locals)
    # If output is a list then checking the return the first element in the list. (Assumes list of Tensors)
    if isinstance(api_call_locals[RETURN_VALUE], List): 
        api_call_locals[RETURN_VALUE] = api_call_locals[RETURN_VALUE][0]
    # If output is a boolean skip .eval().
    if isinstance(api_call_locals[RETURN_VALUE], bool): 
        return api_call_locals, namespace
    # Run eval to check for any backend errors.
    api_call_locals[RETURN_VALUE].eval()
    return api_call_locals, namespace


@pytest.mark.parametrize("test_data", func_list, ids=lambda val: val[5])
def test_dtype_constraints(test_data):
    _, _, return_dtype, _, positive_case, _ = test_data
    if positive_case:
        api_call_locals, namespace = _run_dtype_constraints_subtest(test_data)
        if isinstance(api_call_locals[RETURN_VALUE],bool):
                assert namespace[return_dtype]==tp.bool
        else:
            assert api_call_locals[RETURN_VALUE].dtype == namespace[return_dtype]
    else:
        with pytest.raises(Exception):
            api_call_locals, namespace = _run_dtype_constraints_subtest(test_data)
            assert api_call_locals[RETURN_VALUE].dtype == namespace[return_dtype]
