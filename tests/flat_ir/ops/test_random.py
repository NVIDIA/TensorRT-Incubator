import tripy as tp
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import RandomNormalOp, RandomUniformOp


class TestRandomOp:
    def test_uniform_str(self):
        out = tp.rand((2, 3))
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        random = flat_ir.ops[-1]
        assert isinstance(random, RandomUniformOp)
        assert (
            str(random)
            == "out: [rank=(2), shape=(?, ?,), dtype=(float32), loc=(gpu:0)] = RandomUniformOp(static_low=0.0, static_high=1.0)"
        )

    def test_normal_str(self):
        out = tp.randn((2, 3))
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        random = flat_ir.ops[-1]
        assert isinstance(random, RandomNormalOp)
        assert (
            str(random)
            == "out: [rank=(2), shape=(?, ?,), dtype=(float32), loc=(gpu:0)] = RandomNormalOp(static_mean=0.0, static_std=1.0)"
        )
