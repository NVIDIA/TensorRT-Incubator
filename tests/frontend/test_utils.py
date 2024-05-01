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


class TestConverInputsToTensors:
    def test_args(self):
        assert isinstance(func(0), tp.Tensor)

    def test_kwargs(self):
        assert isinstance(func(a=0), tp.Tensor)

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

    def test_sync_arg_type_invalid(self):
        with helper.raises(
            tp.TripyException,
            match=r"At least one of the arguments: \('a', 'b', 'c'\) must be a \`tripy.Tensor\`.",
        ):
            x, y, z = sync_arg_types(3.0, 3, 4)
