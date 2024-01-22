import pytest
import numpy as np

import tripy as tp
from tripy.frontend.ops import Gather


class TestGather:
    def test_op_func_all_partial(self):
        a = tp.Tensor([1, 2, 3, 4])
        index = tp.Tensor(np.zeros(1, dtype=np.int32))
        out = tp.gather(a, index, axis=0)
        assert isinstance(out, tp.Tensor)
        assert isinstance(out.op, Gather)

    def test_incorrect_dtype(self):
        a = tp.Tensor([[1, 2], [3, 4]], shape=(2, 2))
        index = tp.Tensor(np.zeros(1, dtype=np.float32))
        a = tp.gather(a, index, 0)
        with pytest.raises(
            tp.TripyException, match="Index tensor for gather operation should be of int32 type."
        ) as exc:
            a.eval()
        print(str(exc.value))
