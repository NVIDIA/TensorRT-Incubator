import re
import tripy as tp
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import DequantizeOp


class TestDequantizeOp:
    def test_str(self):
        a = tp.Tensor([2, 4], dtype=tp.int8, name="a")
        scale = tp.Tensor(0.9, name="scale")
        out = tp.dequantize(a, scale, tp.float32)
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        dequant_op = flat_ir.ops[-1]
        assert isinstance(dequant_op, DequantizeOp)
        assert re.match(
            r"out: \[shape=\(2,\), dtype=\(float32\), loc=\(gpu:0\)\] = DequantizeOp\(a, t[0-9]+, axis=None\)",
            str(dequant_op),
        )

    def test_per_channel_str(self):
        a = tp.Tensor([[2, 4], [3, 5]], dtype=tp.int8, name="a")
        scale = tp.Tensor([0.9, 0.9], name="scale")
        out = tp.dequantize(a, scale, tp.float32, dim=0)
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        dequant_op = flat_ir.ops[-1]
        assert isinstance(dequant_op, DequantizeOp)
        assert str(dequant_op) == "out: [shape=(2, 2,), dtype=(float32), loc=(gpu:0)] = DequantizeOp(a, scale, axis=0)"
