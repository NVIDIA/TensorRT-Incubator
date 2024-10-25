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


import inspect
import itertools
from textwrap import dedent
from typing import List

import pytest
from tests import helper
from tests.conftest import skip_if_older_than_sm89
from tests.constraints.object_builders import create_obj

import tripy as tp
from tripy import constraints
from tripy.common.datatype import DATA_TYPES
from tripy.constraints import TYPE_VERIFICATION
from tripy.export import PUBLIC_APIS

# Get all functions/methods which have tensors in the type signature
PUBLIC_API_TENSOR_FUNCTIONS = []
PUBLIC_API_TENSOR_FUNCTION_NAMES = []
for api in PUBLIC_APIS:
    is_module = False
    if inspect.isfunction(api.obj):
        funcs = [api.obj]
    elif inspect.isclass(api.obj):
        if issubclass(api.obj, tp.Module):
            # Skip over modules since the dtype constraint decorator doesn't work for them yet.
            continue
        funcs = [val for _, val in inspect.getmembers(api.obj, predicate=inspect.isfunction)]

    for func in funcs:
        if "Tensor" in str(inspect.signature(func)):
            PUBLIC_API_TENSOR_FUNCTIONS.append(func)
            name = api.qualname
            if func.__name__ not in name:
                name += f".{func.__name__}"
            PUBLIC_API_TENSOR_FUNCTION_NAMES.append(name)


@pytest.mark.parametrize("api", PUBLIC_API_TENSOR_FUNCTIONS, ids=PUBLIC_API_TENSOR_FUNCTION_NAMES)
def test_all_public_apis_verified(api):
    NON_VERIFIABLE_APIS = {"plugin", "Executable.__call__"}
    if api.__qualname__ in NON_VERIFIABLE_APIS:
        pytest.skip(f"Cannot do data type verification for: {NON_VERIFIABLE_APIS}")

    assert api.__qualname__ in TYPE_VERIFICATION, f"Missing datatype constraints for: {api.__qualname__}"


DTYPE_CONSTRAINT_CASES = []

for func_name, (
    func_obj,
    inputs,
    dtype_exceptions,
    return_dtype,
    dtype_variables,
    dtype_constraints,
) in sorted(TYPE_VERIFICATION.items()):
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
    for name, dt in sorted(dtype_variables.items()):
        # Get all of the dtypes for positive test case by excluding types_to_exclude.
        pos_dtypes = set(dt)
        positive_test_dtypes[name] = list(sorted(pos_dtypes))
        # Get all dtypes for negative test case.
        total_dtypes = set(map(str, DATA_TYPES.values()))
        negative_test_dtypes[name] = list(sorted(total_dtypes - pos_dtypes))

    for positive_case in [True, False]:
        if positive_case:
            dtype_lists_list = [positive_test_dtypes]
        else:
            dtype_lists_list = []
            # Create a list of dictionary lists and then go over each dictionary.
            for name_temp, dt in sorted(negative_test_dtypes.items()):
                temp_dict = {name_temp: dt}
                # Iterate through and leave one dtype set to negative and the rest is all dtypes.
                for name_not_equal in negative_test_dtypes.keys():
                    if name_temp != name_not_equal:
                        temp_dict[name_not_equal] = total_dtypes
                dtype_lists_list.append(temp_dict)

        for dtype_lists in dtype_lists_list:
            for combination in sorted(itertools.product(*(dtype_lists.values()))):
                # Create a tuple with keys and corresponding elements.
                namespace = dict(zip(dtype_lists.keys(), map(lambda val: getattr(tp, val), combination)))
                exception = False
                for dtype_exception in dtype_exceptions:
                    if positive_case and namespace == dtype_exception:
                        exception = True
                        positive_case = False

                ids = [f"{dtype_name}={dtype}" for dtype_name, dtype in namespace.items()]

                DTYPE_CONSTRAINT_CASES.append(
                    pytest.param(
                        (
                            func_name,
                            func_obj,
                            inputs,
                            return_dtype,
                            namespace,
                            positive_case,
                            func_name + ("-valid" if positive_case else "-invalid") + ":" + ",".join(ids),
                        ),
                        # float8 is not supported before SM 89
                        marks=(
                            skip_if_older_than_sm89 if any(dtype is tp.float8 for dtype in namespace.values()) else []
                        ),
                    )
                )

                if exception:
                    positive_case = True


def _run_dtype_constraints_subtest(test_data):
    func_name, func_obj, inputs, _, namespace, _, _ = test_data
    kwargs = {}

    # Create all input objects using object_builders.create_obj.
    for param_name, param_type in inputs.items():
        kwargs[param_name] = create_obj(func_obj, func_name, param_name, param_type, namespace)

    args = []

    # Passing "self" as a keyword argument does not work with wrappers like `FuncOverload`.
    if "self" in kwargs:
        args = [kwargs["self"]]
        del kwargs["self"]

    SPECIAL_FUNCS = {
        "__radd__": (lambda self, other: self + other),
        "__rsub__": (lambda self, other: self - other),
        "__rpow__": (lambda self, other: self**other),
        "__rmul__": (lambda self, other: self * other),
        "__rtruediv__": (lambda self, other: self / other),
        "shape": (lambda self: self.shape),
    }

    if func_name in SPECIAL_FUNCS:
        ret_val = SPECIAL_FUNCS[func_name](*args, **kwargs)
    else:
        # We can't call `func_obj` directly because there may be other decorators
        # applied after the dtype constraints one. By importing it like this, we
        # get the final version of the function/class.
        #
        # NOTE: inspect.ismethod() will not work, possibly because of our decorators.
        if "." in func_obj.__qualname__:
            cls, method = func_obj.__qualname__.split(".")

            # For methods, the first argument will be the instance
            obj = args.pop(0)

            code = f"""
            from {func_obj.__module__} import {cls}

            ret_val = obj.{method}(*args, **kwargs)
            """
        else:
            code = f"""
            from {func_obj.__module__} import {func_obj.__qualname__}

            if {func_name} == "shape":
                ret_val = args[0].shape
            else:
                ret_val = {func_obj.__qualname__}(*args, **kwargs)
            """

        all_locals = locals()
        exec(dedent(code), globals(), all_locals)
        ret_val = all_locals["ret_val"]

    # If output does not have dtype skip .eval().
    if isinstance(ret_val, int):
        return ret_val, namespace

    # If output is a list then checking the return the first element in the list. (Assumes list of Tensors)
    if isinstance(ret_val, List):
        ret_val = ret_val[0]

    # Run eval to check for any backend errors.
    ret_val.eval()
    return ret_val, namespace


@pytest.mark.parametrize("test_data", DTYPE_CONSTRAINT_CASES, ids=lambda val: val[-1])
def test_dtype_constraints(test_data):
    # If data type checking is enabled, negative tests will trivially pass (we will throw an
    # error before even trying to call the function).
    with helper.config("enable_dtype_checking", False):
        _, _, _, return_dtype, _, positive_case, _ = test_data
        if positive_case:
            ret_val, namespace = _run_dtype_constraints_subtest(test_data)
            if isinstance(ret_val, tp.Tensor):
                assert ret_val.dtype == namespace[return_dtype]
        else:
            with helper.raises(Exception):
                ret_val, namespace = _run_dtype_constraints_subtest(test_data)
                if isinstance(ret_val, tp.Tensor):
                    assert ret_val.dtype == namespace[return_dtype]


@constraints.dtypes(constraints={"tensors": "T1"}, variables={"T1": ["float32"]})
def sequence_func(tensors: List[tp.Tensor]):
    return


class TestDtypes:
    def test_works_with_sequences(self):
        sequence_func([tp.ones((2, 2), dtype=tp.float32), tp.ones((2, 2), dtype=tp.float32)])

    def test_raises_on_mismatched_sequence_dtypes(self):
        with helper.raises(tp.TripyException, match="Mismatched data types in sequence argument for 'sequence_func'."):
            sequence_func([tp.ones((2, 2), dtype=tp.float32), tp.ones((2, 2), dtype=tp.int32)])
