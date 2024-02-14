import numpy as np
import pytest

import tripy as tp


class TestIota:

    def _compute_ref_iota(self, shape, dim):
        if dim is None:
            dim = 0
        elif dim < 0:
            dim += len(shape)
        expected = np.arange(0, shape[dim], dtype=np.float32)
        if dim < len(shape) - 1:
            expand_dims = [1 + i for i in range(len(shape) - 1 - dim)]
            expected = np.expand_dims(expected, expand_dims)
        expected = np.broadcast_to(expected, shape)
        return expected

    @pytest.mark.parametrize(
        "shape, dim",
        [
            ((2, 3), 1),
            ((2, 3), None),
            ((2, 3), -1),
            ((2, 3, 4), 2),
        ],
    )
    def test_iota(self, shape, dim):
        if dim:
            output = tp.iota(shape, dim)
        else:
            output = tp.iota(shape)

        assert np.array_equal(output.numpy(), self._compute_ref_iota(shape, dim))

    @pytest.mark.parametrize(
        "shape, dim",
        [
            ((2, 3), 1),
            ((2, 3), None),
            ((2, 3), -1),
            ((2, 3, 4), 2),
        ],
    )
    def test_iota_like(self, shape, dim):
        if dim:
            output = tp.iota_like(tp.ones(shape), dim)
        else:
            output = tp.iota_like(tp.ones(shape))

        assert np.array_equal(output.numpy(), self._compute_ref_iota(shape, dim))
