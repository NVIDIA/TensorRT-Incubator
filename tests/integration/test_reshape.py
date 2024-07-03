import numpy as np
import cupy as cp
import pytest
import tripy as tp

from tests import helper


class TestReshape:
    @pytest.mark.parametrize(
        "shape, new_shape",
        [
            ((2, 4), (1, 8)),
            ((2, 4, 8, 9), (8, 8, 9)),
            ((2, 4), (8,)),  # change rank of output
            ((2, 4), (1, -1)),  # check negative dim
        ],
    )
    def test_static_reshape(self, shape, new_shape):
        cp_a = cp.random.rand(*shape).astype(np.float32)
        a = tp.Tensor(cp_a, shape=shape, device=tp.device("gpu"))
        b = tp.reshape(a, new_shape)
        if -1 in new_shape:
            new_shape = tuple(np.prod(shape) // -np.prod(new_shape) if d == -1 else d for d in new_shape)
        assert np.array_equal(cp.from_dlpack(b).get(), cp_a.reshape(new_shape).get())

    def test_dynamic_reshape(self):
        dim = tp.dynamic_dim(runtime_value=4, min=3, opt=5, max=6)
        a_np = np.ones((4, 5, 6, 7), dtype=np.float32)
        a = tp.Tensor(a_np, shape=(dim, 5, 6, 7))
        a = tp.reshape(a, (20, -1, 14))
        assert np.array_equal(cp.from_dlpack(a).get(), a_np.reshape((20, -1, 14)))

    def test_invalid_neg_dim_reshape(self):
        shape = (1, 30)
        new_shape = (-1, -1)
        with helper.raises(tp.TripyException, match="Reshape operation size operand can have only one dimension as -1"):
            a = tp.reshape(tp.ones(shape), new_shape)
            print(a)
