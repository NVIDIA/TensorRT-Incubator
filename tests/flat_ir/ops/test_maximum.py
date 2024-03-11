import tripy as tp
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import MaxOp


class TestMaxOp:
    def test_str(self):
        a = tp.Tensor([3.0, 4.0], name="a")
        b = tp.Tensor([5.0, 2.0], name="b")
        out = tp.maximum(a, b)
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        max_op = flat_ir.ops[-1]
        assert isinstance(max_op, MaxOp)
        assert str(max_op) == "out: [shape=(2,), dtype=(float32), loc=(gpu:0)] = MaxOp(a, b)"
