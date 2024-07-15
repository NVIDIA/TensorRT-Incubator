import cupy as cp
import numpy as np
import pytest
import torch

import tripy as tp


class TestReduceOp:
    @pytest.mark.parametrize(
        "x_shape, axis, keepdim",
        [
            ((2, 3), 1, True),
            ((2, 3, 4), (1, 2), True),
            ((2, 3), 1, False),
            ((2, 3, 4), (1, 2), False),
            ((2, 3, 4), None, False),
            ((2, 3, 4), None, True),
        ],
    )
    def test_mean(self, x_shape, axis, keepdim: bool):
        rand_x = np.random.uniform(
            low=-2.0,
            high=2.0,
            size=x_shape,
        ).astype(np.float32)

        a = tp.Tensor(rand_x)
        out = tp.mean(a, dim=axis, keepdim=keepdim)
        assert np.allclose(cp.from_dlpack(out).get(), np.array(rand_x.mean(axis=axis, keepdims=keepdim)))

    @pytest.mark.parametrize(
        "x_shape, axis, keepdim",
        [
            ((2, 3), 1, True),
            ((2, 3, 4), (1, 2), True),
            ((2, 3, 4), None, False),
            ((2, 3, 4), None, True),
            ((2, 3), 1, False),
            ((2, 3, 4), (1, 2), False),
        ],
    )
    def test_var(self, x_shape, axis, keepdim: bool):
        rand_x = np.random.uniform(
            low=-2.0,
            high=2.0,
            size=x_shape,
        ).astype(np.float32)

        a = tp.Tensor(rand_x)
        out = tp.var(a, dim=axis, keepdim=keepdim)
        torch_tensor = torch.Tensor(rand_x)
        assert np.allclose(cp.from_dlpack(out).get(), torch_tensor.var(dim=axis, keepdim=keepdim))

    @pytest.mark.parametrize(
        "x_shape, axis, keepdim",
        [
            ((2, 3), 1, True),
            ((2, 3, 4), 2, True),
            ((2, 3), 1, False),
            ((2, 3, 4), 2, False),
            ((2, 3, 4), None, False),
            ((2, 3, 4), None, True),
        ],
    )
    def test_argmax(self, x_shape, axis, keepdim: bool):
        rand_x = np.random.uniform(
            low=-2.0,
            high=2.0,
            size=x_shape,
        ).astype(np.float32)

        a = tp.Tensor(rand_x)
        out = tp.argmax(a, dim=axis, keepdim=keepdim)
        assert np.array_equal(cp.from_dlpack(out).get(), np.array(rand_x.argmax(axis=axis, keepdims=keepdim)))

    @pytest.mark.parametrize(
        "x_shape, axis, keepdim",
        [
            ((2, 3), 1, True),
            ((2, 3, 4), 2, True),
            ((2, 3), 1, False),
            ((2, 3, 4), 2, False),
            ((2, 3, 4), None, False),
            ((2, 3, 4), None, True),
        ],
    )
    def test_argmin(self, x_shape, axis, keepdim: bool):
        rand_x = np.random.uniform(
            low=-2.0,
            high=2.0,
            size=x_shape,
        ).astype(np.float32)

        a = tp.Tensor(rand_x)
        out = tp.argmin(a, dim=axis, keepdim=keepdim)
        assert np.array_equal(cp.from_dlpack(out).get(), np.array(rand_x.argmin(axis=axis, keepdims=keepdim)))
