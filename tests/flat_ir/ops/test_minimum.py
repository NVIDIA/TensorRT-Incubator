import tripy as tp
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import MinOp


class TestMinOp:
    def test_str(self):
        a = tp.Tensor([3.0, 4.0], name="a")
        b = tp.Tensor([5.0, 2.0], name="b")
        out = a.minimum(b)
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        min_op = flat_ir.ops[-1]
        assert isinstance(min_op, MinOp)
        assert str(min_op) == "out: [shape=(2,), dtype=(float32), loc=(gpu:0)] = MinOp(a, b)"
