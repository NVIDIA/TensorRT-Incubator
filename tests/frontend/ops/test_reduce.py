import tripy as tp
from tripy.frontend.ops.reduce import Reduce


class TestReduce:
    def test_sum(self):
        a = tp.ones((2, 3))
        a = a.sum(0)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.op, Reduce)

    def test_max(self):
        a = tp.ones((2, 3))
        a = a.max(0)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.op, Reduce)
