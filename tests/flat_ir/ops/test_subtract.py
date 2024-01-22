import tripy as tp
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import SubtractOp


class TestSubtractOp:
    def test_str(self):
        a = tp.Tensor([3.0, 4.0])
        b = tp.Tensor([1.0, 2.0])
        out = a - b

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        sub = flat_ir.ops[-1]
        assert isinstance(sub, SubtractOp)
        assert str(sub) == "t2: [shape=(2,), dtype=(float32), loc=(gpu:0)] = SubtractOp(t0, t1)"
