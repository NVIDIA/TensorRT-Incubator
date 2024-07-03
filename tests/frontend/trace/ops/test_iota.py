import tripy as tp
from tests import helper
from tripy.frontend.trace.ops import Iota


class TestIota:
    def test_iota(self):
        a = tp.iota([2, 3])
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Iota)

    def test_iota_like(self):
        t = tp.Tensor([1, 2, 3])
        a = tp.iota_like(t)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Iota)

    def test_invalid_dim(self):
        a = tp.iota([2, 3], dim=3)
        with helper.raises(tp.TripyException, match="iota dimension cannot go beyond the output rank"):
            a.eval()

    def test_infer_rank(self):
        a = tp.iota((2, 3, 4))
        assert a.trace_tensor.rank == 3


class TestIotaLike:
    def test_iota_like(self):
        t = tp.Tensor([1, 2, 3])
        a = tp.iota_like(t)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Iota)

    def test_infer_rank(self):
        t = tp.Tensor([1, 2, 3])
        a = tp.iota_like(t)
        assert a.trace_tensor.rank == 1
