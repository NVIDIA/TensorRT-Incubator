import pytest

import tripy as tp
from tests import helper
from tripy.frontend.trace.ops import Flip


class TestFlip:
    @pytest.mark.parametrize(
        "dims",
        [None, [], 0, 1, -1, [0], [1], [-1], [0, 1]],
    )
    def test_flip_properties(self, dims):
        t = tp.Tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        f = tp.flip(t, dims=dims)
        assert isinstance(f, tp.Tensor)
        assert isinstance(f.trace_tensor.producer, Flip)
        assert f.trace_tensor.rank == 2
        assert f.shape == (2, 5)

    def test_flip_0_rank(self):
        t = tp.Tensor(1)
        f = tp.flip(t)
        assert isinstance(f, tp.Tensor)
        assert isinstance(f.trace_tensor.producer, Flip)
        assert f.trace_tensor.rank == 0

    def test_out_of_range_dim(self):
        t = tp.Tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        with helper.raises(
            tp.TripyException,
            match=r"All dimensions for flip must be in the range \[-2, 2\), but dimension 3 is out of range",
        ):
            tp.flip(t, dims=3)

    def test_repeated_dim(self):
        t = tp.Tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        with helper.raises(
            tp.TripyException, match="All dimensions for flip must be unique but dimension 0 is repeated"
        ):
            tp.flip(t, dims=[0, 1, 0])

    def test_out_of_range_negative_dim(self):
        t = tp.Tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        with helper.raises(
            tp.TripyException,
            match=r"All dimensions for flip must be in the range \[-2, 2\), but dimension -3 is out of range",
        ):
            tp.flip(t, dims=-3)

    def test_repeated_negative_dim(self):
        t = tp.Tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        with helper.raises(
            tp.TripyException, match=r"All dimensions for flip must be unique but dimension 1 \(-1\) is repeated"
        ):
            tp.flip(t, dims=[0, 1, -1])

    def test_flip_rank_0_with_dims(self):
        t = tp.Tensor(1)
        with helper.raises(tp.TripyException, match="It is not possible to flip a rank-0 tensor"):
            tp.flip(t, dims=0)
