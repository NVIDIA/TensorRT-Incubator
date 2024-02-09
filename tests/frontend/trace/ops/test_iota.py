import tripy as tp
from tests import helper
from tripy.frontend.trace.ops.iota import Iota


class TestIota:
    def test_iota(self):
        a = tp.iota([2, 3])
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.op, Iota)

    def test_iota_like(self):
        t = tp.Tensor([1, 2, 3])
        a = tp.iota_like(t)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.op, Iota)

    def test_invalid_dim(self):
        with helper.raises(tp.TripyException, match="Invalid iota dim."):
            tp.iota([2, 3], dim=3)
