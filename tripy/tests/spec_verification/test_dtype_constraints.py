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
from typing import List, Union, Optional, get_origin, get_args, ForwardRef, get_type_hints
from tripy.common.datatype import DATA_TYPES
import itertools
import pytest
from tests.spec_verification.object_builders import create_obj
from tripy.constraints import TYPE_VERIFICATION, RETURN_VALUE, FUNC_W_DOC_VERIF
import tripy as tp
import tests.helper


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


# Create list of all test that will be run.
pos_func_list = []
neg_func_list = []
for func_name, (
    func_obj,
    inputs,
    dtype_exceptions,
    return_dtype,
    dtype_variables,
    dtype_constraints,
) in TYPE_VERIFICATION.items():
    # Complete the rest of the processing for TEST_VERIFICATION:
    # Get names and type hints for each param.
    func_sig = inspect.signature(func_obj)
    param_dict = func_sig.parameters
    # Construct inputs.
    for param_name in param_dict.keys():
        # Check if there are any provided constraints and add them to the constraint dict.
        inputs[param_name] = dtype_constraints.get(param_name, None)
    # Update dtype_exceptions from string to tripy dtype.
    for i, dtype_exception in enumerate(dtype_exceptions):
        for key, val in dtype_exception.items():
            dtype_exceptions[i][key] = getattr(tp, val)

    # Create test case list.
    positive_test_dtypes = {}
    negative_test_dtypes = {}
    for name, dt in dtype_variables.items():
        # Get all of the dtypes for positive test case by excluding types_to_exclude.
        pos_dtypes = set(dt)
        positive_test_dtypes[name] = list(pos_dtypes)
        # Get all dtypes for negative test case.
        total_dtypes = set(map(str, DATA_TYPES.values()))
        negative_test_dtypes[name] = list(total_dtypes - pos_dtypes)
    for positive_case in [True, False]:
        if positive_case:
            dtype_lists_list = [positive_test_dtypes]
        else:
            dtype_lists_list = []
            # Create a list of dictionary lists and then go over each dictionary.
            for name_temp, dt in negative_test_dtypes.items():
                temp_dict = {name_temp: dt}
                # Iterate through and leave one dtype set to negative and the rest is all dtypes.
                for name_not_equal in negative_test_dtypes.keys():
                    if name_temp != name_not_equal:
                        temp_dict[name_not_equal] = total_dtypes
                dtype_lists_list.append(temp_dict)
        for dtype_lists in dtype_lists_list:
            for combination in itertools.product(*(dtype_lists.values())):
                # Create a tuple with keys and corresponding elements.
                namespace = dict(zip(dtype_lists.keys(), map(lambda val: getattr(tp, val), combination)))
                exception = False
                for dtype_exception in dtype_exceptions:
                    if positive_case and namespace == dtype_exception:
                        exception = True
                        positive_case = False
                ids = [f"{dtype_name}={dtype}" for dtype_name, dtype in namespace.items()]
                if positive_case:
                    pos_func_list.append(
                        (
                            func_name,
                            func_obj,
                            inputs,
                            return_dtype,
                            namespace,
                            func_name + "_valid: " + ", ".join(ids),
                        )
                    )
                else:
                    neg_func_list.append(
                        (
                            func_name,
                            func_obj,
                            inputs,
                            return_dtype,
                            namespace,
                            func_name + "_invalid: " + ", ".join(ids),
                        )
                    )
                if exception:
                    positive_case = True


def _run_dtype_constraints_subtest(test_data):
    func_name, func_obj, inputs, _, namespace, _ = test_data
    kwargs = {}
    # Create all input objects using object_builders.create_obj.
    for param_name, param_type in inputs.items():
        kwargs[param_name] = create_obj(func_obj, func_name, param_name, param_type, namespace)
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


# Positive dtype testing is run during L1 testing.
@pytest.mark.l1
@pytest.mark.parametrize("test_data", pos_func_list, ids=lambda val: val[5])
def test_pos_dtype_constraints(test_data):
    _, _, _, return_dtype, _, _ = test_data
    api_call_locals, namespace = _run_dtype_constraints_subtest(test_data)
    if isinstance(api_call_locals[RETURN_VALUE], tp.Tensor):
        assert api_call_locals[RETURN_VALUE].dtype == namespace[return_dtype]


# Run xfail test cases only during L1 testing.
@pytest.mark.l1
@pytest.mark.parametrize("test_data", neg_func_list, ids=lambda val: val[5])
def test_neg_dtype_constraints(test_data):
    _, _, _, return_dtype, _, _ = test_data
    with pytest.raises(Exception):
        api_call_locals, namespace = _run_dtype_constraints_subtest(test_data)
        if isinstance(api_call_locals[RETURN_VALUE], tp.Tensor):
            assert api_call_locals[RETURN_VALUE].dtype == namespace[return_dtype]


def get_all_possible_verif_ops():
    qualnames = set()
    tripy_interfaces = tests.helper.get_all_tripy_interfaces()
    for obj in tripy_interfaces:
        if not obj.__doc__:
            continue

        blocks = [
            (block.code())
            for block in tests.helper.consolidate_code_blocks(obj.__doc__)
            if isinstance(block, tests.helper.DocstringCodeBlock)
        ]
        if blocks is None:
            continue

        if isinstance(obj, property):
            continue

        func_sig = inspect.signature(func_obj)
        param_dict = func_sig.parameters
        contains_tensor_input = False
        for type_hint in param_dict.values():
            type_hint = type_hint.annotation
            while get_origin(type_hint) in [Union, Optional, list] and not contains_tensor_input:
                type_hint = get_args(type_hint)[0]
                # ForwardRef refers to any case where type hint is a string.
                if isinstance(type_hint, ForwardRef):
                    type_hint = type_hint.__forward_arg__
                    if type_hint == "tripy.Tensor":
                        print(type_hint)
                        contains_tensor_input = True

        if not contains_tensor_input:
            continue

        qualnames.add(obj.__qualname__)

    return qualnames


print(get_all_possible_verif_ops())
print(len(get_all_possible_verif_ops()))

operations = get_all_possible_verif_ops()
# add any function that you do not want to be verified:
func_exceptions = [
    "plugin",
    "dequantize",
    "default",
    "dtype",
    "function",
    "type",
    "tolist",
    "md5",
    "integer",
    "volume",
    "save",
    "floating",
    "load",
    "device",
]


# Check if there are any operations that are not included (Currently does not test any __<op>__ functions)
@pytest.mark.parametrize("func_qualname", operations, ids=lambda val: f"is_{val}_verified")
def test_all_ops_verified(func_qualname):
    if func_qualname not in func_exceptions:
        assert (
            func_qualname in FUNC_W_DOC_VERIF
        ), f"function {func_qualname}'s data types have not been verified. Please add data type verification by following the guide within tripy/tests/spec_verification or exclude it from this test."
    else:
        pytest.skip("Data type constraints are not required for this API")
