import tripy as tp
from tests import helper
from tripy.frontend.trace.ops.expand import Expand


class TestExpand:
    def test_func_op(self):
        a = tp.ones((2, 1))
        a = a.expand((2, 2))
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.op, Expand)

    def test_invalid_small_size(self):
        a = tp.ones((2, 1, 1))
        b = a.expand((2, 2))

        with helper.raises(
            tp.TripyException,
            match="The number of sizes must be greater or equal to input tensor's rank.",
            has_stack_info_for=[a, b],
        ):
            b.eval()

    def test_invalid_mismatch_size(self):
        a = tp.ones((2, 1))
        b = a.expand((4, 2))

        with helper.raises(
            tp.TripyException,
            match="The expanded size must match the existing size at non-singleton dimension.",
            has_stack_info_for=[a, b],
        ):
            b.eval()