import cupy as cp
import numpy as np
import pytest
import tripy as tp


class TestFlip:
    @pytest.mark.parametrize(
        "dims",
        [0, 1, None, [0, 1], [1, 0], -1, -2, [0, -1], [-2, 1]],
    )
    def test_flip(self, dims):
        cp_a = cp.arange(16).reshape((4, 4)).astype(cp.float32)
        a = tp.Tensor(cp_a, device=tp.device("gpu"))
        f = tp.flip(a, dims=dims)
        assert np.array_equal(cp.from_dlpack(f).get(), np.flip(cp_a.get(), axis=dims))

        # also ensure that flipping a second time restores the original value
        f2 = tp.flip(f, dims=dims)
        assert cp.array_equal(cp.from_dlpack(f2), cp_a)

    def test_no_op(self):
        cp_a = cp.arange(16).reshape((4, 4)).astype(cp.float32)
        a = tp.Tensor(cp_a, device=tp.device("gpu"))
        f = tp.flip(a, dims=[])
        assert cp.array_equal(cp.from_dlpack(a), cp.from_dlpack(f))

    def test_zero_rank(self):
        t = tp.Tensor(1)
        f = tp.flip(t)
        assert cp.array_equal(cp.from_dlpack(t), cp.from_dlpack(f))

    @pytest.mark.parametrize(
        "dims1, dims2",
        [(0, -2), (1, -1), ([0, 1], None), ([0, 1], [1, 0]), ([0, 1], [-2, -1])],
    )
    def test_equivalences(self, dims1, dims2):
        cp_a = cp.arange(16).reshape((4, 4)).astype(cp.float32)
        a = tp.Tensor(cp_a, device=tp.device("gpu"))
        f1 = tp.flip(a, dims=dims1)
        f2 = tp.flip(a, dims=dims2)
        assert cp.array_equal(cp.from_dlpack(f1), cp.from_dlpack(f2))
