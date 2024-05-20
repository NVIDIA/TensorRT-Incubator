import numpy as np
import pytest
import tripy as tp


class TestFlip:
    @pytest.mark.parametrize(
        "dims",
        [0, 1, None, [0, 1], [1, 0], -1, -2, [0, -1], [-2, 1]],
    )
    def test_flip(self, dims):
        np_a = np.random.rand(4, 4).astype(np.float32)
        a = tp.Tensor(np_a, device=tp.device("gpu"))
        f = tp.flip(a, dims=dims)
        assert np.array_equal(f.numpy(), np.flip(np_a, axis=dims))

        # also ensure that flipping a second time restores the original value
        f2 = tp.flip(f, dims=dims)
        assert np.array_equal(f2.numpy(), np_a)

    def test_no_op(self):
        np_a = np.random.rand(4, 4).astype(np.float32)
        a = tp.Tensor(np_a, device=tp.device("gpu"))
        f = tp.flip(a, dims=[])
        assert np.array_equal(a.numpy(), f.numpy())

    def test_zero_rank(self):
        t = tp.Tensor(1)
        f = tp.flip(t)
        assert np.array_equal(t.numpy(), f.numpy())

    @pytest.mark.parametrize(
        "dims1, dims2",
        [(0, -2), (1, -1), ([0, 1], None), ([0, 1], [1, 0]), ([0, 1], [-2, -1])],
    )
    def test_equivalences(self, dims1, dims2):
        np_a = np.random.rand(4, 4).astype(np.float32)
        a = tp.Tensor(np_a, device=tp.device("gpu"))
        f1 = tp.flip(a, dims=dims1)
        f2 = tp.flip(a, dims=dims2)
        assert np.array_equal(f1.numpy(), f2.numpy())
