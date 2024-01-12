import tripy as tp
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import BroadcastOp


class TestBroadcastOp:
    def test_str(self):
        out = tp.ones([2, 3], dtype=tp.float32)

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        broadcast = flat_ir.ops[-1]
        assert isinstance(broadcast, BroadcastOp)
        assert (
            str(broadcast)
            == "t0: [shape=(2, 3,), dtype=(float32), loc=(gpu:0)] = BroadcastOp(t_inter1, broadcast_dim=[])"
        )
