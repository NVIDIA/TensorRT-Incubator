import tripy as tp
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import ConstantOp


class TestConstantOp:
    def test_str(self):
        out = tp.Tensor([2.0, 3.0], dtype=tp.float32, name="out")

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        const = flat_ir.ops[-1]
        assert isinstance(const, ConstantOp)
        assert str(const) == "out: [shape=(2,), dtype=(float32), loc=(gpu:0)] = ConstantOp(data=[2.0, 3.0])"
