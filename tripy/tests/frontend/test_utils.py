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

import cupy as cp
import numpy as np

import tripy as tp
from tripy.frontend.shape import ShapeScalar
from tripy.frontend.utils import convert_shape_inputs, convert_to_tensors
from tests import helper
from tripy import constraints
import pytest


@convert_shape_inputs(["s"])
def convert_shape(s):
    return s


@convert_shape_inputs(["s"])
def ignore_not_named(s, t):
    return (s, t)


class TestConvertShapeInputs:
    def test_convert_shape_basic(self):
        s = convert_shape([1, 2, 3])
        assert isinstance(s, tp.Shape)
        assert s.shape == [3]

    def test_convert_shape_already_shape(self):
        s1 = tp.Shape([1, 2, 3])
        s2 = convert_shape(s1)
        assert s1.trace_tensor == s2.trace_tensor

    def test_convert_shape_tensor(self):
        t = tp.Tensor([1, 2, 3], dtype=tp.int32)
        s2 = convert_shape(t)
        assert isinstance(s2, tp.Shape)
        assert t.trace_tensor == s2.trace_tensor

    def test_convert_mixed_type_list_to_shape(self):
        from tripy.frontend.trace.ops.concatenate import Concatenate

        t = tp.Tensor([3], dtype=tp.int32)
        s1 = tp.Shape([1, 2])
        s2 = convert_shape([1, 2, 3, s1[0], 4, 5, t[0], s1[1], 6, 7])
        assert isinstance(s2, tp.Shape)
        assert isinstance(s2.trace_tensor.producer, Concatenate)
        # ensure the concatenation is done correctly
        assert cp.from_dlpack(s2).get().tolist() == [1, 2, 3, 1, 4, 5, 3, 2, 6, 7]

    def test_negative_non_scalar_in_shape(self):
        s1 = tp.Shape([1, 2])
        with helper.raises(tp.TripyException, match="Tensor in a shape argument must be a scalar."):
            s2 = convert_shape([1, s1])

    def test_convert_empty_shape(self):
        s = convert_shape([])
        assert isinstance(s, tp.Shape)
        assert cp.from_dlpack(s).get().tolist() == []

    def test_convert_shape_unsqueeze_tensors(self):
        t1 = tp.Tensor(1, dtype=tp.int32)
        t2 = tp.Tensor(2, dtype=tp.int32)
        s = convert_shape([t1, t2])
        assert isinstance(s, tp.Shape)
        assert cp.from_dlpack(s).get().tolist() == [1, 2]

    def test_convert_only_specified_argument_to_shape(self):
        t1 = tp.Tensor([1, 2, 3], dtype=tp.int32)
        s, t2 = ignore_not_named(t1, [4, 5, 6])
        assert isinstance(s, tp.Shape)
        assert t2 == [4, 5, 6]

    def test_includes_column_range_for_non_tensors_for_magic_methods(self):
        c = tp.ones((2, 3)) + 3

        stack_info = c.trace_tensor.producer.inputs[1].stack_info

        # Column offset of the `3` above.
        assert stack_info[stack_info.get_first_user_frame_index()].column_range == (30, 31)

    def test_includes_column_range_for_non_tensors_for_magic_methods_with_kwargs(self):
        c = tp.ones((2, 3)).__add__(other=3)

        stack_info = c.trace_tensor.producer.inputs[1].stack_info

        # Column offset of the `3` above.
        assert stack_info[stack_info.get_first_user_frame_index()].column_range == (36, 43)


class TestConvertToTensors:
    def test_no_effect_on_non_tensor_likes(self):
        @convert_to_tensors()
        def func(a: int, b: int):
            return a, b

        a, b = func(3, 4)

        assert a is 3
        assert b is 4

    def test_tensor_likes(self):
        @convert_to_tensors()
        def func(a: tp.types.TensorLike):
            return a

        a = func(1.0)

        assert isinstance(a, tp.Tensor)
        assert a.stack_info[3].column_range == (17, 20)

    def test_tensor_not_modified(self):
        @convert_to_tensors()
        def func(a: tp.types.TensorLike):
            return a

        a = tp.Tensor([1.0])
        b = func(a)

        assert b is a

    def test_keyword_args(self):
        @convert_to_tensors()
        def func(a: tp.types.TensorLike):
            return a

        a = func(a=1.0)

        assert isinstance(a, tp.Tensor)
        assert a.stack_info[3].column_range == (17, 22)

    def test_multiple_args(self):
        @convert_to_tensors()
        def func(a: tp.types.TensorLike, b: tp.types.TensorLike):
            return a, b

        a, b = func(1.0, 2.0)

        assert isinstance(a, tp.Tensor)
        assert a.stack_info[3].column_range == (20, 23)

        assert isinstance(b, tp.Tensor)
        assert b.stack_info[3].column_range == (25, 28)

    def test_args_out_of_order(self):
        @convert_to_tensors()
        def func(a: tp.types.TensorLike, b: tp.types.TensorLike):
            return a, b

        a, b = func(b=1.0, a=2.0)

        assert isinstance(a, tp.Tensor)
        assert a.stack_info[3].column_range == (27, 32)
        assert a.tolist() == 2.0

        assert isinstance(b, tp.Tensor)
        assert b.stack_info[3].column_range == (20, 25)
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
            func(tp.ones((2, 2), dtype=dtype), arg)
