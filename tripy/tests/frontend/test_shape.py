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

import numpy as np
import cupy as cp
import pytest

import tripy as tp
from tests.helper import raises


@pytest.fixture(params=[[1, 2, 3]], ids=["simple_shape"])
def values(request):
    return request.param


@pytest.fixture(
    params=[[4, 5], tp.Tensor([4, 5], dtype=tp.int32), np.array([4, 5], dtype=np.int32)],
    ids=[
        "python_list",
        "tripy_tensor",
        "numpy_array",
    ],
)
def other_values(request):
    return request.param


class TestShapeScalar:
    @pytest.mark.parametrize("value", [1, tp.Tensor(1), np.array(2)])
    def test_scalar_shape(self, value):
        s = tp.ShapeScalar(values)

        assert isinstance(s, tp.ShapeScalar)
        assert s.trace_tensor.producer.inputs == []

    def test_scalar_slice(self):
        a = tp.iota((3, 3))
        assert isinstance(a.shape[0], tp.ShapeScalar)

        s = a.shape[0] * a.shape[1]
        b = tp.reshape(a, tp.reshape(s, (1,)))
        assert tp.allclose(tp.flatten(a), b)

    def test_scalar_scalar_op(self):
        a = tp.iota((3, 4))
        s1 = a.shape[0]
        s2 = a.shape[1]
        s = s1 + s2
        assert isinstance(s, tp.ShapeScalar)


class TestShape:
    def test_shape(self, values):
        s = tp.Shape(values)

        assert isinstance(s, tp.Shape)
        assert len(s) == len(values)
        assert s.trace_tensor.producer.inputs == []
        assert cp.from_dlpack(s).get().tolist() == values

    def test_empty_shape(self):
        s = tp.Shape([])

        assert isinstance(s, tp.Shape)
        assert len(s) == 0
        assert s.trace_tensor.producer.inputs == []
        assert cp.from_dlpack(s).get().tolist() == []

    def test_constructor_from_tensor(self, values):
        t = tp.Tensor(values)
        s = tp.Shape(t)

        assert isinstance(s, tp.Shape)
        assert len(s) == len(values)
        assert s.trace_tensor.producer.inputs == []
        # they should be the same underlying value
        assert s.trace_tensor == t.trace_tensor
        assert cp.from_dlpack(s).get().tolist() == values

    def test_constructor_preserve_device(self, values):
        t = tp.Tensor(values, device=tp.device("cpu"))
        s = tp.Shape(t)

        assert isinstance(s, tp.Shape)
        assert s.device.kind == "cpu"

    def test_as_tensor(self, values):
        s = tp.Shape(values)
        t = s.as_tensor()

        assert isinstance(s, tp.Shape)
        assert isinstance(t, tp.Tensor) and not isinstance(t, tp.Shape)
        assert cp.from_dlpack(s).tolist() == cp.from_dlpack(t).tolist()

        # reshape is permitted for an ordinary tensor but not a shape
        reshaped = tp.reshape(t, (len(values), 1))
        assert isinstance(reshaped, tp.Tensor)
        assert reshaped.rank == 2
        assert cp.from_dlpack(reshaped).tolist() == [[v] for v in values]

    def test_plus_override(self, values, other_values):
        from tripy.frontend.trace.ops.concatenate import Concatenate

        appended = [4, 5]
        s = tp.Shape(values)

        # conversion is implicit except for tp.Tensor
        lhs_shape = other_values if not isinstance(other_values, tp.Tensor) else tp.Shape(other_values)
        new_shape = s + lhs_shape
        assert isinstance(new_shape, tp.Shape)
        assert isinstance(new_shape.trace_tensor.producer, Concatenate)
        assert cp.from_dlpack(new_shape).get().tolist() == values + appended

    @pytest.mark.parametrize("constructor", [lambda x: x, tp.Tensor])
    @pytest.mark.parametrize("factor", [-1, 0, 1, 2])
    def test_mul_override(self, values, constructor, factor):
        s = tp.Shape(values)
        f = constructor(factor)
        new_shape = s * f
        rmul = f * s
        expected = values * factor
        assert cp.from_dlpack(new_shape).get().tolist() == expected
        assert cp.from_dlpack(rmul).get().tolist() == expected

    @pytest.mark.parametrize("constructor", [lambda x: x, tp.Tensor])
    @pytest.mark.parametrize("factor", [-1, 0, 1, 2])
    def test_mul_empty_shape(self, constructor, factor):
        s = tp.Shape([])
        for factor in range(3):
            new_shape = s * constructor(factor)
            assert cp.from_dlpack(new_shape).get().tolist() == []

    def test_len_concatenation(self, values):
        s = tp.Shape(values)
        # we are testing that the length is *inferred*, so do not execute the concatenation
        c = s + s
        assert len(c) == 2 * len(values)

    def test_explicit_addition(self, values):
        from tripy.frontend.trace.ops.binary_elementwise import BinaryElementwise

        s = tp.Shape(values)
        res = s.add(tp.Shape(values))
        assert isinstance(res, tp.Shape)
        assert isinstance(res.trace_tensor.producer, BinaryElementwise)
        assert res.trace_tensor.producer.kind == BinaryElementwise.Kind.SUM
        assert cp.from_dlpack(res).get().tolist() == [2 * v for v in values]

    def test_len_binary_op(self, values):
        s = tp.Shape(values)
        res = s.add(tp.Shape(values))
        assert len(res) == len(values)

        res = s.multiply(2)
        assert len(res) == len(values)

    def test_shape_op(self, values):
        from tripy.frontend.trace.ops.shape import Shape

        t = tp.Tensor(values)
        s = t.shape

        assert isinstance(s, tp.Shape)
        assert isinstance(s.trace_tensor.producer, Shape)
        assert cp.from_dlpack(s).get().tolist() == [len(values)]

    def test_len_shape_op(self, values):
        t = tp.Tensor(values)
        s = t.shape
        assert len(s) == 1

    def test_flip(self, values):
        from tripy.frontend.trace.ops.flip import Flip

        # this ensures that a default operator will wrap shapes
        flipped_shape = tp.flip(tp.Shape(values), dims=0)

        assert isinstance(flipped_shape, tp.Shape)
        assert isinstance(flipped_shape.trace_tensor.producer, Flip)
        assert cp.from_dlpack(flipped_shape).get().tolist() == values[::-1]

    def test_len_flip(self, values):
        s = tp.Shape(values)
        flipped = tp.flip(s, dims=0)
        assert len(flipped) == len(values)

    def test_expand(self):
        from tripy.frontend.trace.ops.expand import Expand

        s = tp.Shape([1])
        # rank-1 result, so it's wrapped
        expanded = tp.expand(s, (3,))

        assert isinstance(expanded, tp.Shape)
        assert isinstance(expanded.trace_tensor.producer, Expand)
        assert cp.from_dlpack(expanded).get().tolist() == [1, 1, 1]

    def test_len_expand(self):
        s = tp.Shape([1])
        expanded = tp.expand(s, (3,))
        assert len(expanded) == 3

    def test_gather(self, values):
        from tripy.frontend.trace.ops.gather import Gather

        s1 = tp.Shape(values)
        s2 = tp.gather(s1, 0, tp.Tensor([0, len(values) - 1]))
        assert isinstance(s2, tp.Shape)
        assert isinstance(s2.trace_tensor.producer, Gather)
        assert cp.from_dlpack(s2).get().tolist() == [values[0], values[-1]]

    def test_len_gather(self, values):
        s = tp.Shape(values)
        gathered = tp.gather(s, 0, tp.Tensor([0, len(values) - 1]))
        assert len(gathered) == 2

    def test_matmul(self, values):
        s1 = tp.Shape(values)
        s2 = tp.Shape(values)
        prod = s1 @ s2
        assert not isinstance(prod, tp.Shape)
        # matmul is multiple ops in FlatIR
        assert cp.from_dlpack(prod).get() == np.array(values) @ np.array(values)

    def test_multiply_by_scalar(self, values):
        from tripy.frontend.trace.ops.binary_elementwise import BinaryElementwise

        # binary elementwise ops are overridden to accept one shape
        s = tp.Shape(values)
        doubled = s.multiply(2)

        assert isinstance(doubled, tp.Shape)
        assert isinstance(doubled.trace_tensor.producer, BinaryElementwise)
        assert doubled.trace_tensor.producer.kind == " * "
        assert cp.from_dlpack(doubled).get().tolist() == list(map(lambda x: 2 * x, values))

    def test_squeeze(self):
        from tripy.frontend.trace.ops.reshape import Squeeze

        # Result of a squeeze is not a Shape, since squeezing a shape will be rank 0.
        # However, it may be desirable to obtain the value of a particular shape dimension,
        # such as when slicing.
        s = tp.Shape([1])
        v = tp.squeeze(s, 0)
        assert not isinstance(v, tp.Shape)
        assert isinstance(v.trace_tensor.producer, Squeeze)
        assert cp.from_dlpack(v).get() == 1

    def test_slice_single(self, values):
        s = tp.Shape(values)
        dim = s[0]
        assert not isinstance(dim, tp.Shape)
        # note: the parent is not actually Slice because there is a squeeze afterwards,
        # but the test should not necessarily reflect that implementation detail
        assert cp.from_dlpack(dim).get() == values[0]

    def test_slice_range(self, values):
        from tripy.frontend.trace.ops.slice import Slice

        s = tp.Shape(values)
        dims = s[1:]
        assert isinstance(dims, tp.Shape)
        assert isinstance(dims.trace_tensor.producer, Slice)
        assert cp.from_dlpack(dims).get().tolist() == values[1:]

    @pytest.mark.parametrize(
        "slice_value",
        [
            slice(0, 2),
            slice(0, 1),
            slice(1, 3),
            slice(0, 3, 2),
            slice(1, 3, 2),
            slice(1, 4, 2),
            slice(1, 4, 3),  # should select only one
            slice(1, None, 200),  # selects only start point
            # some with negative strides
            slice(None, None, -1),
            slice(None, None, -2),
            slice(4, 0, -1),
            slice(2, 0, -1),
            slice(2, 1, -1),
            # check the clamping behavior
            slice(-10, 20),
            slice(10, -20, -1),
            # invalid bounds (length 0 result)
            slice(0, 4, -1),
            slice(4, 0),
            slice(2, 2),
        ],
    )
    def test_slice_len(self, slice_value):
        # checking consistency against Python list
        values = [1, 2, 3, 4]
        s1 = tp.Shape(values)
        assert len(s1[slice_value]) == len(values[slice_value])

    def test_iteration(self, values):
        s = tp.Shape(values)
        for i, v in enumerate(s):
            assert cp.from_dlpack(v).get() == values[i]

    def test_reduce(self, values):
        from tripy.frontend.trace.ops.reduce import Reduce

        s = tp.Shape(values)
        max_value = tp.max(s)
        assert not isinstance(max_value, tp.Shape)
        assert isinstance(max_value.trace_tensor.producer, Reduce)
        assert max_value.trace_tensor.producer.kind == Reduce.Kind.MAX
        assert cp.from_dlpack(max_value).get() == max(values)

    def test_right_addition(self, other_values):
        from tripy.frontend.trace.ops.concatenate import Concatenate

        values = [1, 2, 3]
        appended = [4, 5]
        s = tp.Shape(values)

        # conversion is implicit except for tp.Tensor
        rhs_shape = other_values if not isinstance(other_values, tp.Tensor) else tp.Shape(other_values)

        new_shape = rhs_shape + s
        assert isinstance(new_shape, tp.Shape)
        assert isinstance(new_shape.trace_tensor.producer, Concatenate)
        assert cp.from_dlpack(new_shape).get().tolist() == appended + values

    def test_reshape_identity(self, values):
        from tripy.frontend.trace.ops.reshape import Reshape

        s1 = tp.Shape(values)
        s2 = tp.reshape(s1, (len(values),))
        assert isinstance(s2, tp.Shape)
        assert isinstance(s2.trace_tensor.producer, Reshape)
        assert cp.from_dlpack(s1).get().tolist() == cp.from_dlpack(s2).get().tolist()

    def test_reshape_not_wrapped(self, values):
        from tripy.frontend.trace.ops.reshape import Reshape

        s1 = tp.Shape(values)
        s2 = tp.reshape(s1, (len(values), 1))
        assert not isinstance(s2, tp.Shape)
        assert isinstance(s2.trace_tensor.producer, Reshape)
        assert cp.from_dlpack(s2).get().tolist() == [[v] for v in values]

    def test_cast_identity(self, values):
        s1 = tp.Shape(values)
        c = tp.cast(s1, tp.int32)
        assert isinstance(c, tp.Shape)
        assert cp.from_dlpack(c).get().tolist() == cp.from_dlpack(s1).get().tolist()

    def test_cast_not_wrapped(self, values):
        tensor = tp.Tensor(values)
        shape = tp.Shape(tensor)
        c1 = tp.cast(shape, tp.float32)
        c2 = tp.cast(tensor, tp.float32)
        assert not isinstance(c1, tp.Shape)
        assert isinstance(c1, tp.Tensor)
        assert cp.from_dlpack(c1).get().tolist() == cp.from_dlpack(c2).get().tolist()

    def test_expand_higher_rank_not_wrapped(self):
        s = tp.Shape([1])
        e = tp.expand(s, [3, 1])
        assert not isinstance(e, tp.Shape)
        assert cp.from_dlpack(e).get().tolist() == [[1] for _ in range(3)]

    def test_cast_len(self, values):
        s = tp.Shape(values)
        cast = tp.cast(s, tp.int32)
        assert len(cast) == len(values)

    def test_split(self, values):
        s = tp.Shape(values)
        outputs = tp.split(s, len(values))
        assert len(outputs) == len(values)
        for i, output in enumerate(outputs):
            assert isinstance(output, tp.Shape)
            assert cp.from_dlpack(output).get().tolist() == [values[i]]

    def test_split_len(self, values):
        s = tp.Shape(values)
        outputs = tp.split(s, len(values))
        for output in outputs:
            assert len(output) == 1

    def test_split_len_intervals(self):
        s = tp.Shape([1, 2, 3, 4, 5])
        outputs = tp.split(s, [1, 4])
        assert len(outputs[0]) == 1  # 0:1
        assert len(outputs[1]) == 3  # 1:4
        assert len(outputs[2]) == 1  # 4:5

    def test_where(self, values):
        from tripy.frontend.trace.ops.where import Where

        zeros = [0 for _ in range(len(values))]
        s1 = tp.Shape(values)
        s2 = tp.Shape(zeros)
        cond = s1 >= 2

        res = tp.where(cond, s1, s2)
        assert isinstance(res, tp.Shape)
        assert isinstance(res.trace_tensor.producer, Where)
        assert cp.from_dlpack(res).get().tolist() == [0 if values[i] < 2 else values[i] for i in range(len(values))]

    def test_where_len(self, values):
        s1 = tp.Shape(values)
        s2 = tp.Shape([0 for _ in values])
        cond = tp.Tensor([i >= 1 for i in range(len(values))], dtype=tp.bool)
        res = tp.where(cond, s1, s2)
        assert len(res) == len(values)

    def test_invalid_input_dtype_tensor(self):
        with raises(
            tp.TripyException, match="Shape tensors must have int32 members, but input tensor has data type float32"
        ):
            _ = tp.Shape(tp.ones((3,)))

    def test_invalid_input_rank(self):
        with raises(
            tp.TripyException,
            match="Tensors used to represent shapes must be of rank 1, but given shape \\(2\\, 3\\) has rank 2",
        ):
            _ = tp.Shape(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))

    def test_invalid_input_rank_tensor(self):
        with raises(tp.TripyException, match="Shape tensors must be of rank 1, but input tensor is rank 2"):
            _ = tp.Shape(tp.ones((3, 2), dtype=tp.int32))

    def test_invalid_mul_rank(self, values):
        s = tp.Shape(values)
        with raises(
            tp.TripyException, match="Attempting to multiply a Tripy Shape by a tensor of rank >= 1, which is undefined"
        ):
            _ = s * values

    def test_invalid_plus_type(self, values):
        s = tp.Shape(values)
        t = tp.Tensor(values, dtype=tp.int32)
        with raises(
            tp.TripyException,
            match="Attempting to add a Tripy Tensor to a Tripy Shape, which is not allowed. Consider calling tp.Shape explicitly",
        ):
            s + t

    def test_invalid_right_addition_type(self, values):
        s = tp.Shape(values)
        t = tp.Tensor(values, dtype=tp.int32)
        with raises(
            tp.TripyException,
            match="Attempting to add a Tripy Tensor to a Tripy Shape, which is not allowed. Consider calling tp.Shape explicitly",
        ):
            t + s

    def test_dequantize_rejected(self, values):
        # the message will have to do with the datatype and not the fact it's a shape
        with raises(tp.TripyException):
            tp.dequantize(tp.Shape(values), 2, tp.int8, 0)

    def test_quantize_rejected(self, values):
        # the message will have to do with the datatype and not the fact it's a shape
        with raises(tp.TripyException):
            tp.quantize(tp.Shape(values), 2, tp.int8, 0)

    def test_binary_elementwise_broadcast_rejected(self, values):
        with raises(
            tp.TripyException, match="For binary elementwise operators on Shapes, all inputs must be of rank at most 1"
        ):
            tp.Shape(values).multiply(tp.Tensor([values, values]))

    def test_shape_equality(self, other_values):
        a = tp.Shape([4, 5])
        if isinstance(other_values, np.ndarray):
            pytest.skip("numpy array cannot be implicitly cast to Shape type")
        eq = a == other_values
        assert isinstance(eq, bool)
        assert eq

    def test_shape_inequality(self):
        a = tp.Shape([1, 2, 3])
        b = tp.Shape([1, 4, 5])
        assert a != b

    def test_shape_inequality_different_lengths(self):
        a = tp.Shape([1])
        b = tp.Shape([1, 2])
        assert a != b

        c = tp.Shape([1, 2, 3])
        assert a != c
        assert b != c
