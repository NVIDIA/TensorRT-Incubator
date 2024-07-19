import pytest

import tripy as tp
from tests import helper
from tripy.frontend.trace.ops import Expand


class TestExpand:
    def test_func_op(self):
        a = tp.ones((2, 1))
        a = tp.expand(a, (2, 2))
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Expand)

    def test_invalid_small_size(self):
        a = tp.ones((2, 1, 1))

        with helper.raises(
            tp.TripyException,
            match="The shape of size tensor must be greater or equal to input tensor's rank",
            has_stack_info_for=[a],
        ):
            b = tp.expand(a, (2, 2))
            b.eval()

    def test_invalid_mismatch_size(self):
        a = tp.ones((2, 1))
        b = tp.expand(a, (4, 2))

        with helper.raises(
            tp.TripyException,
            match=r"size of operand dimension 0 \(2\) is not compatible with size of result dimension 0 \(4\)",
            has_stack_info_for=[a, b],
        ):
            b.eval()

    def test_infer_rank(self):
        a = tp.ones((2, 1))
        a = tp.expand(a, (2, 2))
        assert a.trace_tensor.rank == 2
