import numpy as np
import torch
import pytest

import tripy as tp
from tripy.frontend import Tensor
from tripy.frontend.tensor_ops import ones, where


class TestReduceOp:
    @pytest.mark.parametrize(
        "x_shape, axis, keepdim",
        [
            ((2, 3), 1, True),
            ((2, 3, 4), (1, 2), True),
            ((2, 3), 1, False),
            ((2, 3, 4), (1, 2), False),
            ((2, 3, 4), None, False),
        ],
    )
    def test_mean(self, x_shape, axis, keepdim: bool):
        rand_x = np.random.uniform(
            low=-2.0,
            high=2.0,
            size=x_shape,
        ).astype(np.float32)

        a = Tensor(rand_x)
        out = a.mean(dim=axis, keepdim=keepdim)
        assert np.allclose(out.numpy(), np.array(rand_x.mean(axis=axis, keepdims=keepdim)))

    @pytest.mark.parametrize(
        "x_shape, axis, keepdim",
        [
            ((2, 3), 1, True),
            ((2, 3, 4), (1, 2), True),
            ((2, 3, 4), None, False),
            # #90 will fix this test.
            # ((2,3), 1, False),
            # ((2,3,4), (1,2), False),
        ],
    )
    def test_var(self, x_shape, axis, keepdim: bool):
        rand_x = np.random.uniform(
            low=-2.0,
            high=2.0,
            size=x_shape,
        ).astype(np.float32)

        a = Tensor(rand_x)
        out = a.var(dim=axis, keepdim=keepdim)
        torch_tensor = torch.Tensor(rand_x)
        assert np.allclose(out.numpy(), torch_tensor.var(dim=axis, keepdim=keepdim))
