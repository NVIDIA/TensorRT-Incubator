import numpy as np
import pytest

import tripy as tp
from tests import helper
from tripy.frontend.trace.ops import Permute, Transpose


class TestPermute:
    def test_op_func(self):
        a = tp.Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
        a = tp.permute(a, (1, 0))
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.op, Permute)

    @pytest.mark.parametrize("perm", [(0,), (0, 1, 2)])
    def test_mistmatched_permutation_fails(self, perm):
        a = tp.ones((2, 3), dtype=tp.float32)
        b = tp.permute(a, perm)

        with helper.raises(
            tp.TripyException, match="Incorrect number of elements in permutation.", has_stack_info_for=[a, b]
        ):
            b.eval()


class TestTranspose:
    def test_op_func(self):
        a = tp.Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
        a = tp.transpose(a, 0, 1)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.op, Transpose)

    def test_incorrect_number_of_arguments(self):
        a = tp.ones((2, 3))

        with helper.raises(tp.TripyException, match="Function expects 3 parameters, but 4 arguments were provided."):
            b = tp.transpose(a, 1, 2, 3)
