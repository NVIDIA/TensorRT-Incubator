import tripy as tp
from tripy.frontend.utils import convert_inputs_to_tensors


@convert_inputs_to_tensors()
def func(a):
    return a


@convert_inputs_to_tensors()
def multi_input(a, b, c):
    return a, b, c


class TestConverInputsToTensors:
    def test_args(self):
        assert isinstance(func(0), tp.Tensor)

    def test_kwargs(self):
        assert isinstance(func(a=0), tp.Tensor)

    # When we convert arguments to tensors, we should preserve the column range
    # of the original non-Tensor argument.
    def test_includes_column_range_for_non_tensors(self):
        tensor = func(3.0)

        # Column offset of the `3.0` above.
        assert tensor._stack_info[tensor._stack_info.get_first_user_frame_index()].column_range == (22, 25)

    def test_includes_column_range_for_non_tensors_multiple_inputs(self):
        a, b, c = multi_input(1, 2.0, 3)

        # Column offsets of the arguments above.
        assert a._stack_info[a._stack_info.get_first_user_frame_index()].column_range == (30, 31)
        assert b._stack_info[b._stack_info.get_first_user_frame_index()].column_range == (33, 36)
        assert c._stack_info[c._stack_info.get_first_user_frame_index()].column_range == (38, 39)

    def test_includes_column_range_for_non_tensors_multiple_inputs_with_kwargs(self):
        a, b, c = multi_input(1, b=2.0, c=3)

        # Column offsets of the arguments above.
        assert a._stack_info[a._stack_info.get_first_user_frame_index()].column_range == (30, 31)
        assert b._stack_info[b._stack_info.get_first_user_frame_index()].column_range == (33, 38)
        assert c._stack_info[c._stack_info.get_first_user_frame_index()].column_range == (40, 43)

    def test_includes_column_range_for_non_tensors_for_magic_methods(self):
        c = tp.ones((2, 3)) + 3

        stack_info = c.op.inputs[1].stack_info

        # Column offset of the `3` above.
        assert stack_info[stack_info.get_first_user_frame_index()].column_range == (30, 31)

    def test_includes_column_range_for_non_tensors_for_magic_methods_with_kwargs(self):
        c = tp.ones((2, 3)).__add__(other=3)

        stack_info = c.op.inputs[1].stack_info

        # Column offset of the `3` above.
        assert stack_info[stack_info.get_first_user_frame_index()].column_range == (36, 43)
