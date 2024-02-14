import tripy as tp
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import ArgMinMaxOp, ReduceOp, DivideOp, BroadcastOp, ConvertOp, MulOp
import re


class TestReduceOp:
    def test_sum_str(self):
        inp = tp.Tensor([[1, 2], [3, 4]], name="inp")
        out = inp.sum(0)
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        reduce = flat_ir.ops[-1]
        assert isinstance(reduce, ReduceOp)
        assert (
            str(reduce)
            == "out: [shape=(2,), dtype=(int32), loc=(gpu:0)] = ReduceOp(inp, t_inter2, reduce_mode=sum, reduce_dims=[0])"
        )

    def test_max_str(self):
        inp = tp.Tensor([[1, 2], [3, 4]], name="inp")
        out = inp.max(0)
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        reduce = flat_ir.ops[-1]
        assert isinstance(reduce, ReduceOp)
        assert (
            str(reduce)
            == "out: [shape=(2,), dtype=(int32), loc=(gpu:0)] = ReduceOp(inp, t_inter2, reduce_mode=max, reduce_dims=[0])"
        )

    def test_mean_str(self):
        inp = tp.Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=tp.float32, name="inp")
        out = inp.mean(0)
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        div = flat_ir.ops[-1]
        assert isinstance(div, DivideOp)
        assert re.match(
            r"out: \[shape=\(2,\), dtype=\(float32\), loc=\(gpu:0\)\] = DivideOp\(t[0-9]+, t_inter[0-9]+\)", str(div)
        )

        broadcast = flat_ir.ops[-2]
        assert isinstance(broadcast, BroadcastOp)
        assert re.match(
            r"t_inter[0-9]+: \[shape=\(2,\), dtype=\(float32\), loc=\(gpu:0\)\] = BroadcastOp\(t[0-9]+, broadcast_dim=\[\]\)",
            str(broadcast),
        )

        add = flat_ir.ops[-4]
        assert isinstance(add, MulOp)
        assert re.match(
            r"t[0-9]+: \[shape=\(\), dtype=\(int32\), loc=\(gpu:0\)\] = MulOp\(t[0-9]+, t[0-9]+\)", str(add)
        )

        reduce = flat_ir.ops[-9]
        assert isinstance(reduce, ReduceOp)
        assert re.match(
            r"t[0-9]+: \[shape=\(2,\), dtype=\(float32\), loc=\(gpu:0\)\] = ReduceOp\(inp, t_inter[0-9]+, reduce_mode=sum, reduce_dims=\[0\]\)",
            str(reduce),
        )

    def test_argmax_str(self):
        inp = tp.Tensor([[1, 2], [3, 4]], name="inp")
        out = inp.argmax(0)
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        reduce = flat_ir.ops[-1]
        assert isinstance(reduce, ArgMinMaxOp)
        assert re.match(
            r"out: \[shape=\(2,\), dtype=\(int32\), loc=\(gpu:0\)\] = ArgMinMaxOp\(inp, t[0-9]+, t_inter[0-9]+, t_inter[0-9]+, reduce_mode=argmax, reduce_dims=\[0\]\)",
            str(reduce),
        )

    def test_argmin_str(self):
        inp = tp.Tensor([[1, 2], [3, 4]], name="inp")
        out = inp.argmin(0)
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        reduce = flat_ir.ops[-1]
        assert isinstance(reduce, ArgMinMaxOp)
        assert re.match(
            r"out: \[shape=\(2,\), dtype=\(int32\), loc=\(gpu:0\)\] = ArgMinMaxOp\(inp, t[0-9]+, t_inter[0-9]+, t_inter[0-9]+, reduce_mode=argmin, reduce_dims=\[0\]\)",
            str(reduce),
        )
