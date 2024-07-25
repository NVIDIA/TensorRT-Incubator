import cupy as cp

import tripy as tp
from tripy.frontend.utils import convert_inputs_to_tensors
from tests import helper


@convert_inputs_to_tensors()
def func(a):
    return a


@convert_inputs_to_tensors()
def multi_input(a, b, c):
    return a, b, c


@convert_inputs_to_tensors(sync_arg_types=[("a", "b", "c")])
def sync_arg_types(a, b, c):
    return a, b, c


@convert_inputs_to_tensors()
def variadic_positional_args(*args):
    return args


@convert_inputs_to_tensors()
def arg_before_variadic_positional_args(x, *args):
    return (x,) + args


@convert_inputs_to_tensors()
def kwarg_after_variadic_positional_args(*args, y):
    return args + (y,)


@convert_inputs_to_tensors(unpack_argument=["xs"])
def convert_list_input(xs):
    return xs


@convert_inputs_to_tensors(sync_arg_types=[("xs",)], unpack_argument=["xs"])
def sync_within_list(xs):
    return xs


@convert_inputs_to_tensors(sync_arg_types=[("x", "ys")], unpack_argument=["ys"])
def sync_single_type_to_list(x, ys):
    return x, ys


@convert_inputs_to_tensors(sync_arg_types=[("xs", "y")], unpack_argument=["xs"])
def sync_list_type_to_single(xs, y):
    return xs, y


@convert_inputs_to_tensors(sync_arg_types=[("xs", "ys")], unpack_argument=["xs", "ys"])
def sync_list_types(xs, ys):
    return xs, ys


@convert_inputs_to_tensors(shape_argument=["s"])
def convert_shape(s):
    return s


class TestConvertInputsToTensors:
    def test_args(self):
        assert isinstance(func(0), tp.Tensor)

    def test_kwargs(self):
        assert isinstance(func(a=0), tp.Tensor)

    def test_convert_list_into_tensor(self):
        t1 = func([1, 2, 3])
        assert isinstance(t1, tp.Tensor)
        assert t1.shape == (3,)

        t2 = func([[1, 2], [3, 4]])
        assert t2.shape == (2, 2)

    def test_convert_list_input(self):
        xs = convert_list_input([1.0, 2.0, 3.0, 4.0])
        assert len(xs) == 4
        for x in xs:
            assert isinstance(x, tp.Tensor)
        assert not convert_list_input([])

    def test_convert_tuple_input(self):
        xs = convert_list_input((1.0, 2.0))
        assert isinstance(xs, tuple)
        assert len(xs) == 2
        assert isinstance(xs[0], tp.Tensor)
        assert isinstance(xs[1], tp.Tensor)

    def test_variadic_positional_args(self):
        x, y = variadic_positional_args(1.0, 2.0)
        assert isinstance(x, tp.Tensor)
        assert isinstance(y, tp.Tensor)

    def test_arg_before_variadic_positional_args(self):
        x, y = arg_before_variadic_positional_args(1.0, 2.0)
        assert isinstance(x, tp.Tensor)
        assert isinstance(y, tp.Tensor)

    def test_kwarg_after_variadic_positional_args(self):
        x, y = kwarg_after_variadic_positional_args(1.0, y=2.0)
        assert isinstance(x, tp.Tensor)
        assert isinstance(y, tp.Tensor)

    def test_convert_shape_basic(self):
        s = convert_shape([1, 2, 3])
        assert isinstance(s, tp.Shape)
        assert s.shape == (3,)

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

    # When we convert arguments to tensors, we should preserve the column range
    # of the original non-Tensor argument.
    def test_includes_column_range_for_non_tensors(self):
        tensor = func(3.0)

        # Column offset of the `3.0` above.
        assert tensor.stack_info[tensor.stack_info.get_first_user_frame_index()].column_range == (22, 25)

    def test_includes_column_range_for_non_tensors_multiple_inputs(self):
        a, b, c = multi_input(1, 2.0, 3)

        # Column offsets of the arguments above.
        assert a.stack_info[a.stack_info.get_first_user_frame_index()].column_range == (30, 31)
        assert b.stack_info[b.stack_info.get_first_user_frame_index()].column_range == (33, 36)
        assert c.stack_info[c.stack_info.get_first_user_frame_index()].column_range == (38, 39)

    def test_includes_column_range_for_non_tensors_multiple_inputs_with_kwargs(self):
        a, b, c = multi_input(1, b=2.0, c=3)

        # Column offsets of the arguments above.
        assert a.stack_info[a.stack_info.get_first_user_frame_index()].column_range == (30, 31)
        assert b.stack_info[b.stack_info.get_first_user_frame_index()].column_range == (33, 38)
        assert c.stack_info[c.stack_info.get_first_user_frame_index()].column_range == (40, 43)

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

    def test_includes_column_range_for_list_elements(self):
        xs = convert_list_input([1.0, 2.0])
        assert xs[0].stack_info[xs[0].stack_info.get_first_user_frame_index()].column_range == (33, 36)
        assert xs[1].stack_info[xs[1].stack_info.get_first_user_frame_index()].column_range == (38, 41)

    def test_includes_column_range_for_tuple_elements(self):
        xs = convert_list_input((1.0, 2.0))
        assert xs[0].stack_info[xs[0].stack_info.get_first_user_frame_index()].column_range == (33, 36)
        assert xs[1].stack_info[xs[1].stack_info.get_first_user_frame_index()].column_range == (38, 41)

    def test_sync_arg_type_includes_non_tensor_column_range(self):
        x, y, z = sync_arg_types(tp.Tensor(3.0, dtype=tp.float16), 3, 4.0)

        assert y.dtype == tp.float16
        assert z.dtype == tp.float16
        assert y.stack_info[y.stack_info.get_first_user_frame_index()].column_range == (67, 68)
        assert z.stack_info[z.stack_info.get_first_user_frame_index()].column_range == (70, 73)

    def test_sync_arg_type_includes_non_tensor_column_range_with_kwargs(self):
        x, y, z = sync_arg_types(tp.Tensor(3.0, dtype=tp.float16), b=3, c=4.0)

        assert y.dtype == tp.float16
        assert z.dtype == tp.float16
        assert y.stack_info[y.stack_info.get_first_user_frame_index()].column_range == (67, 70)
        assert z.stack_info[z.stack_info.get_first_user_frame_index()].column_range == (72, 77)

    def test_sync_arg_type_not_applied_to_tensors(self):
        x, y, z = sync_arg_types(
            tp.Tensor(3.0),
            tp.Tensor(3, dtype=tp.int32),
            tp.Tensor(4, dtype=tp.float16),
        )

        assert x.dtype == tp.float32
        assert y.dtype == tp.int32
        assert z.dtype == tp.float16

    def test_sync_arg_type_within_list(self):
        xs = sync_within_list([1.0, tp.Tensor(3, dtype=tp.float16), 5])

        assert xs[0].dtype == tp.float16
        assert xs[1].dtype == tp.float16
        assert xs[2].dtype == tp.float16

    def test_sync_single_arg_type_to_list(self):
        _, ys = sync_single_type_to_list(tp.Tensor(5, dtype=tp.int32), [2.0, 3.0, 4.0])

        assert ys[0].dtype == tp.int32
        assert ys[1].dtype == tp.int32
        assert ys[2].dtype == tp.int32

    def test_sync_list_arg_type_to_single_arg(self):
        xs, y = sync_list_type_to_single([1.0, tp.Tensor(5, dtype=tp.int32), 4.0], 1.0)

        assert xs[0].dtype == tp.int32
        assert xs[2].dtype == tp.int32
        assert y.dtype == tp.int32

    def test_sync_list_arg_types(self):
        xs, ys = sync_list_types([1.0, 2.0, 3.0], [3, 4, tp.Tensor(6, dtype=tp.int32)])

        for x in xs:
            assert x.dtype == tp.int32
        for y in ys:
            assert y.dtype == tp.int32

    def test_sync_arg_type_list_not_applied_to_tensors(self):
        xs = sync_within_list(
            [tp.Tensor(1.0, dtype=tp.int32), tp.Tensor(3, dtype=tp.float16), tp.Tensor(5, dtype=tp.float32)]
        )

        assert xs[0].dtype == tp.int32
        assert xs[1].dtype == tp.float16
        assert xs[2].dtype == tp.float32

    def test_sync_arg_type_invalid(self):
        with helper.raises(
            tp.TripyException,
            match=r"At least one of the arguments: \('a', 'b', 'c'\) must be a \`tripy.Tensor\`.",
        ):
            x, y, z = sync_arg_types(3.0, 3, 4)
