import tripy as tp
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import ExpOp


class TestExpOp:
    def test_str(self):
        out = tp.Tensor([1.0, 1.0]).exp()

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        exp = flat_ir.ops[-1]
        assert isinstance(exp, ExpOp)
        assert str(exp) == "t1: [shape=(2,), dtype=(float32), loc=(gpu:0)] = ExpOp(t0)"
