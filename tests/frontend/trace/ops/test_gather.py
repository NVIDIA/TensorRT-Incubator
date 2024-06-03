import numpy as np
import pytest

import tripy as tp
from tests import helper
from tripy.frontend.trace.ops import Gather


class TestGather:
    def test_op_func_all_partial(self):
        a = tp.Tensor([1, 2, 3, 4])
        index = tp.Tensor(np.zeros(1, dtype=np.int32))
        out = tp.gather(a, 0, index)
        assert isinstance(out, tp.Tensor)
        assert isinstance(out.trace_tensor.producer, Gather)

    def test_incorrect_dtype(self):
        a = tp.Tensor([[1, 2], [3, 4]], shape=(2, 2))
        index = tp.Tensor(np.zeros(1, dtype=np.float32))
        with helper.raises(
            tp.TripyException,
            match="Index tensor for gather operation should be of int32 type.",
            has_stack_info_for=[a, index],
        ):
            b = tp.gather(a, 0, index)

    @pytest.mark.parametrize("index_shape", [(1,), (2, 2)])
    def test_infer_rank(self, index_shape):
        a = tp.Tensor([1, 2, 3, 4])
        index = tp.Tensor(np.zeros(index_shape, dtype=np.int32))
        out = tp.gather(a, 0, index)
        assert out.trace_tensor.rank == a.rank + len(index_shape) - 1
