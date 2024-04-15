import tripy as tp
from tripy.frontend.trace.ops import Unsqueeze


class TestUnsqueeze:
    def test_func_op(self):
        a = tp.ones((2, 1))
        a = tp.unsqueeze(a, 0)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Unsqueeze)
