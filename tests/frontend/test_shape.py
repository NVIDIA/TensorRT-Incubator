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


class TestShape:
    def test_shape(self, values):
        s = tp.Shape(values)

        assert isinstance(s, tp.Shape)
        assert s.trace_tensor.producer.inputs == []
        assert cp.from_dlpack(s).get().tolist() == values

    def test_empty_shape(self):
        s = tp.Shape([])

        assert isinstance(s, tp.Shape)
        assert s.trace_tensor.producer.inputs == []
        assert cp.from_dlpack(s).get().tolist() == []

    def test_constructor_from_tensor(self, values):
        t = tp.Tensor(values)
        s = tp.Shape(t)

        assert isinstance(s, tp.Shape)
        assert s.trace_tensor.producer.inputs == []
        # they should be the same underlying value
        assert s.trace_tensor == t.trace_tensor
        assert cp.from_dlpack(s).get().tolist() == values

    def test_constructor_preserve_device(self, values):
        t = tp.Tensor(values, device=tp.device("cpu"))
        s = tp.Shape(t)

        assert isinstance(s, tp.Shape)
        assert s.device.kind == "cpu"
        assert np.from_dlpack(s).tolist() == values

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

    def test_shape_jit(self, values):
        s = tp.Shape(values)

        @tp.jit
        def cat(a, b):
            return a + b

        new_shape = cat(s, s)
        assert isinstance(new_shape, tp.Shape)
        assert cp.from_dlpack(new_shape).get().tolist() == values + values

        # try with an ordinary tensor, using the ordinary tensor's definition of +
        t = tp.Tensor(values)
        new_t = cat(t, t)
        assert not isinstance(new_t, tp.Shape)
        expected_sum = [2 * v for v in values]
        assert cp.from_dlpack(new_t).get().tolist() == expected_sum

        # use with tensors of a different rank to ensure that there is recompilation,
        # since the cache uses shape information
        tt = tp.Tensor([values, values])
        new_tt = cat(tt, tt)
        assert not isinstance(new_tt, tp.Shape)
        assert cp.from_dlpack(new_tt).get().tolist() == [expected_sum, expected_sum]

        # call a second time to ensure that shape information is cached correctly
        new_shape = cat(s, s)
        assert isinstance(new_shape, tp.Shape)
        assert cp.from_dlpack(new_shape).get().tolist() == values + values

    def test_comparison_not_wrapped(self, values):
        from tripy.frontend.trace.ops.binary_elementwise import Comparison

        s = tp.Shape(values)
        eq_comparison = s == values
        assert not isinstance(eq_comparison, tp.Shape)
        assert isinstance(eq_comparison.trace_tensor.producer, Comparison)
        assert eq_comparison.trace_tensor.producer.kind == Comparison.Kind.EQUAL
        # TODO (#26): Currently these are returned as ints, not bools
        assert cp.from_dlpack(eq_comparison).get().tolist() == [1 for _ in values]

    def test_explicit_addition(self, values):
        from tripy.frontend.trace.ops.binary_elementwise import BinaryElementwise

        s = tp.Shape(values)
        res = s.add(tp.Shape(values))
        assert isinstance(res, tp.Shape)
        assert isinstance(res.trace_tensor.producer, BinaryElementwise)
        assert res.trace_tensor.producer.kind == BinaryElementwise.Kind.SUM
        assert cp.from_dlpack(res).get().tolist() == [2 * v for v in values]

    def test_shape_op(self, values):
        from tripy.frontend.trace.ops.shape import Shape

        t = tp.Tensor(values)
        s = t.shape

        assert isinstance(s, tp.Shape)
        assert isinstance(s.trace_tensor.producer, Shape)
        assert cp.from_dlpack(s).get().tolist() == [len(values)]

    def test_flip(self, values):
        from tripy.frontend.trace.ops.flip import Flip

        # this ensures that a default operator will wrap shapes
        flipped_shape = tp.flip(tp.Shape(values), dims=0)

        assert isinstance(flipped_shape, tp.Shape)
        assert isinstance(flipped_shape.trace_tensor.producer, Flip)
        assert cp.from_dlpack(flipped_shape).get().tolist() == values[::-1]

    def test_expand(self):
        from tripy.frontend.trace.ops.expand import Expand

        s = tp.Shape([1])
        # rank-1 result, so it's wrapped
        expanded = tp.expand(s, (3,))

        assert isinstance(expanded, tp.Shape)
        assert isinstance(expanded.trace_tensor.producer, Expand)
        assert cp.from_dlpack(expanded).get().tolist() == [1, 1, 1]

    def test_gather(self, values):
        from tripy.frontend.trace.ops.gather import Gather

        s1 = tp.Shape(values)
        s2 = tp.gather(s1, 0, tp.Tensor([0, len(values) - 1]))
        assert isinstance(s2, tp.Shape)
        assert isinstance(s2.trace_tensor.producer, Gather)
        assert cp.from_dlpack(s2).get().tolist() == [values[0], values[-1]]

    @pytest.mark.skip("#186: Fix test_matrix_multiplication.py hang for 1D tensor.")
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
        doubled = 2 * s

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

    def test_split(self, values):
        s = tp.Shape(values)
        outputs = tp.split(s, len(values))
        assert len(outputs) == len(values)
        for i, output in enumerate(outputs):
            assert isinstance(output, tp.Shape)
            assert cp.from_dlpack(output).get().tolist() == [values[i]]

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

    def test_rank_mismatch(self):
        with raises(tp.TripyException, match="Data has incorrect shape"):
            _ = tp.Shape(np.array([1, 2, 3], dtype=np.int32), num_dims=2)

    def test_invalid_input_dtype(self):
        with raises(tp.TripyException, match="Data has incorrect dtype"):
            _ = tp.Shape(np.array([2.0, 3.0], dtype=np.float32))

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
            tp.Shape(values) * tp.Tensor([values, values])

    def test_unary_elementwise_fails_at_run_time(self, values):
        v = tp.exp(tp.Shape(values))
        with raises(
            tp.TripyException,
            match=(
                "must be ranked tensor of f8E4M3B11FNUZ type or f8E4M3FN type or f8E4M3FNUZ type "
                "or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 "
                "type or complex type with 32-bit float or 64-bit float elements or 4/8/16/32-bit uniform "
                "quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values"
            ),
        ):
            v.eval()
