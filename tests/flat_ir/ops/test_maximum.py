import re
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
        broadcast_a = flat_ir.ops[-3]
        broadcast_b = flat_ir.ops[-2]

        assert isinstance(max_op, MaxOp)
        assert re.match(
            r"t_inter[0-9]+: \[rank=\(1\), dtype=\(float32\), loc=\(gpu:0\)\] = DynamicBroadcastOp\(a, t_inter[0-9]+, broadcast_dim=\[0\]\)",
            str(broadcast_a),
        )
        assert re.match(
            r"t_inter[0-9]+: \[rank=\(1\), dtype=\(float32\), loc=\(gpu:0\)\] = DynamicBroadcastOp\(b, t_inter[0-9]+, broadcast_dim=\[0\]\)",
            str(broadcast_b),
        )
        assert re.match(
            r"out: \[rank=\(1\), dtype=\(float32\), loc=\(gpu:0\)\] = MaxOp\(t_inter[0-9]+, t_inter[0-9]+\)",
            str(max_op),
        )
