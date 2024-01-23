import tripy as tp
from tripy.frontend.ops.unsqueeze import Unsqueeze


class TestUnsqueeze:
    def test_func_op(self):
        a = tp.ones((2, 1))
        a = a.unsqueeze(0)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.op, Unsqueeze)
