import tripy as tp
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import SubtractOp


class TestSubtractOp:
    def test_str(self):
        a = tp.Tensor([3.0, 4.0], name="a")
        b = tp.Tensor([1.0, 2.0], name="b")
        out = a - b
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        sub = flat_ir.ops[-1]
        assert isinstance(sub, SubtractOp)
        assert str(sub) == "out: [shape=(2,), dtype=(float32), loc=(gpu:0)] = SubtractOp(a, b)"
