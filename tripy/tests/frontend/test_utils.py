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

import pytest
from tests import helper

import tripy as tp
from tripy import constraints
from tripy.frontend.utils import convert_to_tensors, tensor_from_shape_like


@pytest.mark.parametrize(
    "shape, expected",
    [
        ([1, 2, 3], [1, 2, 3]),
        ([tp.DimensionSize(1), tp.DimensionSize(2)], [1, 2]),
        ([], []),
        ([1, tp.DimensionSize(2), 3], [1, 2, 3]),
        ([1, tp.DimensionSize(2), 3, 4], [1, 2, 3, 4]),
    ],
)
def test_tensor_from_shape_like(shape, expected):
    tensor = tensor_from_shape_like(shape)

    assert tensor.tolist() == expected


class TestConvertToTensors:
    def test_no_effect_on_non_tensor_likes(self):
        @convert_to_tensors()
        def func(a: tp.Tensor, b: int):
            return a, b

        original_a = tp.Tensor([1, 2])
        a, b = func(original_a, 4)

        assert a is original_a
        assert b is 4

    def test_tensor_likes(self):
        @convert_to_tensors()
        def func(a: tp.types.TensorLike):
            return a

        a = func(1.0)

        assert isinstance(a, tp.Tensor)
        assert a.stack_info[2].column_range == (17, 20)

    def test_converts_to_dimension_size(self):
        # The decorator should convert to DimensionSizes when possible.
        @convert_to_tensors()
        def func(a: tp.types.TensorLike):
            return a

        a = func(1)
        assert type(a) is tp.DimensionSize

        # floats cannot be DimensionSizes
        a = func(1.0)
        assert type(a) is tp.Tensor

    def test_shape_likes(self):
        @convert_to_tensors()
        def func(a: tp.types.ShapeLike):
            return a

        a = func([1, 2, 3])

        assert isinstance(a, tp.Tensor)
        assert a.shape == [3]
        assert bool(tp.all(a == tp.Tensor([1, 2, 3])))

        # Should also work from shapes of tensors
        inp = tp.Tensor([[1, 2], [2, 3]])
        a = inp.shape + [3, 5]  # Should yield: [2, 2, 3, 5]

        a = func(a)

        assert isinstance(a, tp.Tensor)
        assert a.shape == [4]
        assert bool(tp.all(a == tp.Tensor([2, 2, 3, 5])))

    def test_keyword_args(self):
        @convert_to_tensors()
        def func(a: tp.types.TensorLike):
            return a

        a = func(a=1.0)

        assert isinstance(a, tp.Tensor)
        assert a.stack_info[2].column_range == (17, 22)

    def test_multiple_args(self):
        @convert_to_tensors()
        def func(a: tp.types.TensorLike, b: tp.types.TensorLike):
            return a, b

        a, b = func(1.0, 2.0)

        assert isinstance(a, tp.Tensor)
        assert a.stack_info[2].column_range == (20, 23)

        assert isinstance(b, tp.Tensor)
        assert b.stack_info[2].column_range == (25, 28)

    def test_args_out_of_order(self):
        @convert_to_tensors()
        def func(a: tp.types.TensorLike, b: tp.types.TensorLike):
            return a, b

        a, b = func(b=1.0, a=2.0)

        assert isinstance(a, tp.Tensor)
        assert a.stack_info[2].column_range == (27, 32)
        assert a.tolist() == 2.0

        assert isinstance(b, tp.Tensor)
        assert b.stack_info[2].column_range == (20, 25)
        assert b.tolist() == 1.0

    def test_cast_dtype(self):
        # When type constraints are included, the decorator should automatically cast when possible.
        @convert_to_tensors()
        @constraints.dtypes(
            constraints={"a": "T1", "b": "T1", constraints.RETURN_VALUE: "T1"},
            variables={"T1": ["float16"]},
        )
        def func(a: tp.Tensor, b: tp.types.TensorLike):
            return a, b

        a, b = func(tp.Tensor([1.0], dtype=tp.float16), 4.0)

        assert isinstance(b, tp.Tensor)
        assert b.dtype == tp.float16

        a, b = func(tp.Tensor([1.0], dtype=tp.float16), 4)

        assert isinstance(b, tp.Tensor)
        assert b.dtype == tp.float16

    @pytest.mark.parametrize("arg, dtype", [(1.0, tp.int32), (1.0, tp.int64), (2, tp.bool)])
    def test_refuse_unsafe_cast(self, arg, dtype):
        @convert_to_tensors()
        @constraints.dtypes(
            constraints={"a": "T1", "b": "T1", constraints.RETURN_VALUE: "T1"},
            variables={"T1": ["int32", "int64"]},
        )
        def func(a: tp.Tensor, b: tp.types.TensorLike):
            return a, b

        with helper.raises(tp.TripyException, "Refusing to automatically cast"):
            func(tp.Tensor([1, 2], dtype=dtype), arg)

    def test_preprocess_args(self):

        def add_a_to_b(a, b):
            return {"b": a + b}

        @convert_to_tensors(preprocess_args=add_a_to_b)
        def func(a: tp.types.TensorLike, b: tp.types.TensorLike):
            return a, b

        a, b = func(1, 2)

        assert b.tolist() == 3

    def test_variadic_args(self):

        def increment(a, *args):
            return {"a": a + 1, "args": list(map(lambda arg: arg + 1, args))}

        @convert_to_tensors(preprocess_args=increment)
        def func(a: tp.Tensor, *args):
            return [a] + list(args)

        a, b, c = func(tp.Tensor(1), tp.Tensor(2), tp.Tensor(3))
        assert a.tolist() == 2
        assert b.tolist() == 3
        assert c.tolist() == 4
