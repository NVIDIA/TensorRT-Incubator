import tripy as tp
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import ReduceOp, DivideOp, BroadcastOp, ConvertOp, MulOp


class TestReduceOp:
    def test_sum_str(self):
        out = tp.Tensor([[1, 2], [3, 4]])
        out = out.sum(0)

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        reduce = flat_ir.ops[-1]
        assert isinstance(reduce, ReduceOp)
        assert (
            str(reduce)
            == "t1: [shape=(2,), dtype=(int32), loc=(gpu:0)] = ReduceOp(t0, t_inter2, reduce_mode=sum, reduce_dims=[0])"
        )

    def test_max_str(self):
        out = tp.Tensor([[1, 2], [3, 4]])
        out = out.max(0)

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        reduce = flat_ir.ops[-1]
        assert isinstance(reduce, ReduceOp)
        assert (
            str(reduce)
            == "t1: [shape=(2,), dtype=(int32), loc=(gpu:0)] = ReduceOp(t0, t_inter2, reduce_mode=max, reduce_dims=[0])"
        )

    def test_mean_str(self):
        out = tp.Tensor([[1, 2], [3, 4]]).float()
        out = out.mean(0)

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        div = flat_ir.ops[-1]
        assert isinstance(div, DivideOp)
        assert str(div) == "t2: [shape=(2,), dtype=(float32), loc=(gpu:0)] = DivideOp(t0, t_inter11)"

        broadcast = flat_ir.ops[-2]
        assert isinstance(broadcast, BroadcastOp)
        assert (
            str(broadcast)
            == "t_inter11: [shape=(2,), dtype=(float32), loc=(gpu:0)] = BroadcastOp(t1, broadcast_dim=[])"
        )

        cast = flat_ir.ops[-3]
        assert isinstance(cast, ConvertOp)
        assert str(cast) == "t1: [shape=(), dtype=(float32), loc=(gpu:0)] = ConvertOp(t4)"

        add = flat_ir.ops[-4]
        assert isinstance(add, MulOp)
        assert str(add) == "t4: [shape=(), dtype=(int32), loc=(gpu:0)] = MulOp(t6, t7)"

        reduce = flat_ir.ops[-9]
        assert isinstance(reduce, ReduceOp)
        assert (
            str(reduce)
            == "t0: [shape=(2,), dtype=(float32), loc=(gpu:0)] = ReduceOp(t3, t_inter3, reduce_mode=sum, reduce_dims=[0])"
        )
