from tests import helper

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
        b = a[:, :, 0:1]

        with helper.raises(
            tp.TripyException,
            match=r"Too many indices for input tensor[\.a-zA-Z:|/_0-9-=+,\[\]\s]*?Input tensor has a rank of 2 but was attempted to be sliced with 3 indices.",
            has_stack_info_for=[a, b],
        ) as exc:
            b.eval()
