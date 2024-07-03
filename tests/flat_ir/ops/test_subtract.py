import tripy as tp
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import SubtractOp


class TestSubtractOp:
    def test_str(self):
        a = tp.Tensor([3.0, 4.0], name="a")
        b = tp.Tensor([1.0, 2.0], name="b")
        out = a - b
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        sub = flat_ir.ops[-1]
        broadcast_a = flat_ir.ops[-3]
        broadcast_b = flat_ir.ops[-2]

        assert isinstance(sub, SubtractOp)
        assert (
            str(broadcast_a)
            == "t_inter3: [rank=(1), shape=(?,), dtype=(float32), loc=(gpu:0)] = DynamicBroadcastOp(a, t_inter4, broadcast_dim=[0])"
        )
        assert (
            str(broadcast_b)
            == "t_inter9: [rank=(1), shape=(?,), dtype=(float32), loc=(gpu:0)] = DynamicBroadcastOp(b, t_inter4, broadcast_dim=[0])"
        )
        assert str(sub) == "out: [rank=(1), shape=(?,), dtype=(float32), loc=(gpu:0)] = SubtractOp(t_inter3, t_inter9)"
