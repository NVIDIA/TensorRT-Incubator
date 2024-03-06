import tripy as tp
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import QuantizeOp


class TestQuantizeOp:
    def test_str(self):
        a = tp.Tensor([1.0, 2.0], name="a")
        out = tp.quantize(a, tp.int8, 1.0)
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        quantize_op = flat_ir.ops[-1]
        assert isinstance(quantize_op, QuantizeOp)
        assert (
            str(quantize_op)
            == "out: [shape=(2,), dtype=(int8), loc=(gpu:0)] = QuantizeOp(a, scale=1.0, zero_point=0, storage_min=-128, storage_max=127)"
        )
