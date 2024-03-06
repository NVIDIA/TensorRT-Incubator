import tripy as tp
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import DequantizeOp


class TestQuantizeOp:
    def test_str(self):
        a = tp.quantize(tp.Tensor([1.0, 2.0]), tp.int8, 1.0)
        a.name = "a"
        out = tp.dequantize(a, tp.float32)
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        dequant_op = flat_ir.ops[-1]
        assert isinstance(dequant_op, DequantizeOp)
        assert str(dequant_op) == "out: [shape=(2,), dtype=(float32), loc=(gpu:0)] = DequantizeOp(a)"
