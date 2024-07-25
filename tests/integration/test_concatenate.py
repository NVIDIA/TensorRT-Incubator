import numpy as np
import cupy as cp
import pytest
import tripy as tp

from tests import helper


class TestConcatenate:
    @pytest.mark.parametrize(
        "tensor_shapes, dim",
        [
            ([(2, 3, 4), (2, 4, 4)], 1),
            ([(2, 3, 4), (2, 3, 2)], -1),
            ([(2, 3, 4), (2, 3, 4)], 0),
            ([(2, 3, 4)], 0),
        ],
    )
    def test_concat(self, tensor_shapes, dim):
        tensors = [tp.ones(shape) for shape in tensor_shapes]
        out = tp.concatenate(tensors, dim=dim)
        assert np.array_equal(
            cp.from_dlpack(out).get(), np.concatenate([np.ones(shape) for shape in tensor_shapes], axis=dim)
        )

    @pytest.mark.parametrize(
        "tensor_shapes, dim",
        [([(2, 3, 4), (2, 4, 4)], 0), ([(4, 5, 6), (4, 1, 6)], -1)],
    )
    def test_negative_concat(self, tensor_shapes, dim):
        tensors = [tp.ones(shape) for shape in tensor_shapes]
        with helper.raises(tp.TripyException, match=f"not compatible at non-concat index"):
            out = tp.concatenate(tensors, dim=dim)
            print(out)
