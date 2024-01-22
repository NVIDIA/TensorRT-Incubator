import tripy as tp
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import ReduceOp


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
