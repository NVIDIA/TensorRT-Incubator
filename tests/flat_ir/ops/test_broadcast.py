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
        assert (
            str(broadcast)
            == "out: [rank=(2), shape=(?, ?,), dtype=(float32), loc=(gpu:0)] = DynamicBroadcastOp(t_inter1, t_inter2, broadcast_dim=[])"
        )
