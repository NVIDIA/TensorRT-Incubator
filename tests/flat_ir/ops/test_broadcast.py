import re
import tripy as tp
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import DynamicBroadcastOp


class TestBroadcastOp:
    def test_str(self):
        out = tp.ones([2, 3], dtype=tp.float32)
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        broadcast = flat_ir.ops[-1]
        assert isinstance(broadcast, DynamicBroadcastOp)
        assert re.match(
            r"out: \[rank=\(2\), dtype=\(float32\), loc=\(gpu:0\)\] = DynamicBroadcastOp\(t_inter2, t[0-9]+, broadcast_dim=\[\]\)",
            str(broadcast),
        )
