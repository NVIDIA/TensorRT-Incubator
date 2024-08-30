from tests import helper
import tripy as tp


class TestRepeat:
    def test_invalid_dim_fails(self):
        a = tp.ones((2, 2))
        with helper.raises(tp.TripyException, "Dimension argument is out of bounds."):
            tp.repeat(a, 2, dim=4)

    def test_negative_repeats_fails(self):
        a = tp.ones((2, 2))
        with helper.raises(tp.TripyException, "`repeats` value must be non-negative."):
            tp.repeat(a, -1, dim=0)
