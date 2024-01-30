from textwrap import dedent

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

        with pytest.raises(
            tp.TripyException,
            match=dedent(
                rf"""
                Too many indices for input tensor.
                    For expression:

                    | {__file__}:[0-9]+
                    | ------------------------------------------
                    |         a = a[:, :, 0:1]

                    Input tensor has a rank of 2 but was attempted to be sliced with 3 indices.

                    Input 0 was:

                    | {__file__}:[0-9]+
                    | -----------------------------------------------------
                    |         a = tp.Tensor([[1, 2], [3, 4]], shape=(2, 2))
            """
            ).strip(),
        ) as exc:
            a.eval()
        print(str(exc.value))
