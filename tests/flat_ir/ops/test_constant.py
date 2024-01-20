import tripy as tp
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import ConstantOp


class TestConstantOp:
    def test_str(self):
        out = tp.Tensor([2.0, 3.0], dtype=tp.float32)

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        const = flat_ir.ops[-1]
        assert isinstance(const, ConstantOp)
        assert (
            str(const)
            == "t0: [shape=(2,), dtype=(float32), loc=(gpu:0)] = ConstantOp(data=[2. 3.], dtype=float32, device=gpu:0)"
        )