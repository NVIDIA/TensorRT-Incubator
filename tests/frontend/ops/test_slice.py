import pytest

import tripy as tp
from tripy.frontend.ops.slice import Slice


class TestSlice:
    def test_op_func_all_partial(self):
        a = tp.Tensor([1, 2, 3, 4])
        a = a[:2]
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.op, Slice)

    def test_incorrect_index_size(self):
        a = tp.Tensor([[1, 2], [3, 4]], shape=(2, 2))
        a = a[:, :, 0:1]

        with pytest.raises(tp.TripyException, match="Too many indices for array.") as exc:
            a.eval()
        print(str(exc.value))
