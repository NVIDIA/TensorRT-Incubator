#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import List

import nvtripy as tp
import pytest
from nvtripy.export import PUBLIC_APIS
from nvtripy.frontend import wrappers
from nvtripy.frontend.wrappers import DATA_TYPE_CONSTRAINTS
from tests import helper

# Get all functions/methods which have tensors in the type signature
PUBLIC_API_TENSOR_FUNCTIONS = []
PUBLIC_API_TENSOR_FUNCTION_NAMES = []
NON_VERIFIABLE_APIS = {"plugin", "Executable.__call__", "Tensor.eval", "DimensionSize.eval"}
for api in PUBLIC_APIS:
    if inspect.isfunction(api.obj):
        funcs = [api.obj]
    elif inspect.isclass(api.obj):
        if issubclass(api.obj, tp.Module):
            # Skip over modules since the dtype constraint decorator doesn't work for them yet.
            continue
        funcs = [val for _, val in inspect.getmembers(api.obj, predicate=inspect.isfunction)]

    for func in funcs:
        if "Tensor" in str(inspect.signature(func)) and func.__qualname__ not in NON_VERIFIABLE_APIS:
            PUBLIC_API_TENSOR_FUNCTIONS.append(func)
            PUBLIC_API_TENSOR_FUNCTION_NAMES.append(
                api.qualname + f".{func.__name__}" if func.__name__ not in api.qualname else ""
            )

DATA_TYPE_CONSTRAINTS_FUNC_NAMES = {dtc.func.__qualname__ for dtc in DATA_TYPE_CONSTRAINTS}


@pytest.mark.parametrize("api", PUBLIC_API_TENSOR_FUNCTIONS, ids=PUBLIC_API_TENSOR_FUNCTION_NAMES)
def test_all_public_apis_verified(api):
    assert api.__qualname__ in DATA_TYPE_CONSTRAINTS_FUNC_NAMES, f"Missing datatype constraints for: {api.__qualname__}"


@wrappers.interface(dtype_constraints={"tensors": "T1"}, dtype_variables={"T1": ["float32"]})
def sequence_func(tensors: List[tp.Tensor]):
    return


class TestDtypes:
    def test_works_with_sequences(self):
        sequence_func([tp.ones((2, 2), dtype=tp.float32), tp.ones((2, 2), dtype=tp.float32)])

    def test_raises_on_mismatched_sequence_dtypes(self):
        with helper.raises(tp.TripyException, match="Mismatched data types in sequence argument for 'sequence_func'."):
            sequence_func([tp.ones((2, 2), dtype=tp.float32), tp.ones((2, 2), dtype=tp.int32)])


STACK_DEPTH_OF_CALLER = 3


class TestTensorConversion:
    def test_no_effect_on_non_tensor_likes(self):
        @wrappers.interface(convert_to_tensors=True)
        def func(a: tp.Tensor, b: int):
            return a, b

        original_a = tp.Tensor([1, 2])
        a, b = func(original_a, 4)

        assert a is original_a
        assert b is 4

    def test_tensor_likes(self):
        @wrappers.interface(convert_to_tensors=True)
        def func(a: tp.types.TensorLike):
            return a

        a = func(1.0)

        assert isinstance(a, tp.Tensor)
        assert a.stack_info[STACK_DEPTH_OF_CALLER].column_range == (17, 20)

    def test_converts_to_dimension_size(self):
        # The decorator should convert to DimensionSizes when possible.
        @wrappers.interface(convert_to_tensors=True)
        def func(a: tp.types.TensorLike):
            return a

        a = func(1)
        assert type(a) is tp.DimensionSize

        # floats cannot be DimensionSizes
        a = func(1.0)
        assert type(a) is tp.Tensor

    def test_shape_likes(self):
        @wrappers.interface(convert_to_tensors=True)
        def func(a: tp.types.ShapeLike):
            return a

        a = func([1, 2, 3])

        assert isinstance(a, tp.Tensor)
        assert a.shape == (3,)
        assert bool(tp.all(a == tp.Tensor([1, 2, 3])))

        # Should also work from shapes of tensors
        inp = tp.Tensor([[1, 2], [2, 3]])
        a = inp.shape + (3, 5)  # Should yield: [2, 2, 3, 5]

        a = func(a)

        assert isinstance(a, tp.Tensor)
        assert a.shape == (4,)
        assert bool(tp.all(a == tp.Tensor([2, 2, 3, 5])))

    def test_keyword_args(self):
        @wrappers.interface(convert_to_tensors=True)
        def func(a: tp.types.TensorLike):
            return a

        a = func(a=1.0)

        assert isinstance(a, tp.Tensor)
        assert a.stack_info[STACK_DEPTH_OF_CALLER].column_range == (17, 22)

    def test_multiple_args(self):
        @wrappers.interface(convert_to_tensors=True)
        def func(a: tp.types.TensorLike, b: tp.types.TensorLike):
            return a, b

        a, b = func(1.0, 2.0)

        assert isinstance(a, tp.Tensor)
        assert a.stack_info[STACK_DEPTH_OF_CALLER].column_range == (20, 23)

        assert isinstance(b, tp.Tensor)
        assert b.stack_info[STACK_DEPTH_OF_CALLER].column_range == (25, 28)

    def test_args_out_of_order(self):
        @wrappers.interface(convert_to_tensors=True)
        def func(a: tp.types.TensorLike, b: tp.types.TensorLike):
            return a, b

        a, b = func(b=1.0, a=2.0)

        assert isinstance(a, tp.Tensor)
        assert a.stack_info[STACK_DEPTH_OF_CALLER].column_range == (27, 32)
        assert a.tolist() == 2.0

        assert isinstance(b, tp.Tensor)
        assert b.stack_info[STACK_DEPTH_OF_CALLER].column_range == (20, 25)
        assert b.tolist() == 1.0

    def test_cast_dtype(self):
        # When type constraints are included, the decorator should automatically cast when possible.
        @wrappers.interface(
            dtype_constraints={"a": "T1", "b": "T1", wrappers.RETURN_VALUE: "T1"},
            dtype_variables={"T1": ["float16"]},
            convert_to_tensors=True,
        )
        def func(a: tp.Tensor, b: tp.types.TensorLike):
            return a, b

        a, b = func(tp.ones([1], dtype=tp.float16), 4.0)

        assert isinstance(b, tp.Tensor)
        assert b.dtype == tp.float16

        a, b = func(tp.ones([1], dtype=tp.float16), 4)

        assert isinstance(b, tp.Tensor)
        assert b.dtype == tp.float16

    @pytest.mark.parametrize("arg, dtype", [(1.0, tp.int32), (1.0, tp.int64), (2, tp.bool)])
    def test_refuse_unsafe_cast(self, arg, dtype):
        @wrappers.interface(
            dtype_constraints={"a": "T1", "b": "T1", wrappers.RETURN_VALUE: "T1"},
            dtype_variables={"T1": ["int32", "int64"]},
            convert_to_tensors=True,
        )
        def func(a: tp.Tensor, b: tp.types.TensorLike):
            return a, b

        with helper.raises(tp.TripyException, "Refusing to automatically cast"):
            func(tp.ones([2], dtype=dtype), arg)

    def test_preprocess_func(self):

        def add_a_to_b(a, b):
            return {"b": a + b}

        @wrappers.interface(convert_to_tensors=True, conversion_preprocess_func=add_a_to_b)
        def func(a: tp.types.TensorLike, b: tp.types.TensorLike):
            return a, b

        a, b = func(1, 2)

        assert b.tolist() == 3

    def test_variadic_args(self):

        def increment(a, *args):
            return {"a": a + 1, "args": list(map(lambda arg: arg + 1, args))}

        @wrappers.interface(convert_to_tensors=True, conversion_preprocess_func=increment)
        def func(a: tp.Tensor, *args):
            return [a] + list(args)

        a, b, c = func(tp.Tensor(1), tp.Tensor(2), tp.Tensor(3))
        assert a.tolist() == 2
        assert b.tolist() == 3
        assert c.tolist() == 4
