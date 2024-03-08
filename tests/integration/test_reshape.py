import numpy as np
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
        np_a = np.random.rand(*shape).astype(np.float32)
        a = tp.Tensor(np_a, shape=shape, device=tp.device("gpu"))
        b = tp.reshape(a, new_shape)
        if -1 in new_shape:
            new_shape = tuple(np.prod(shape) // -np.prod(new_shape) if d == -1 else d for d in new_shape)
        assert np.array_equal(b.numpy(), np_a.reshape(new_shape))

    def test_dynamic_reshape(self):
        dim = tp.Dim(runtime_value=4, min=3, opt=5, max=6)
        a = tp.ones((dim, 5, 6, 7))
        with helper.raises(NotImplementedError, match="Dynamic reshape is not supported"):
            a = tp.reshape(a, (20, 3, 14))
            print(a)

    def test_invalid_neg_dim_reshape(self):
        shape = (1, 30)
        new_shape = (-1, -1)
        with helper.raises(tp.TripyException, match="Only one dimension can be -1."):
            a = tp.reshape(tp.ones(shape), new_shape)
            print(a)
