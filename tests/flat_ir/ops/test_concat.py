import tripy as tp
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import ConcatenateOp


class TestConcatOp:
    def test_str(self):
        a = tp.ones((2, 3))
        a.name = "a"
        b = tp.ones((3, 3))
        b.name = "b"
        out = tp.concatenate([a, b], dim=0)
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        concat = flat_ir.ops[-1]
        assert isinstance(concat, ConcatenateOp)
        assert str(concat) == "out: [rank=(2), dtype=(float32), loc=(gpu:0)] = ConcatenateOp(a, b, dim=0)"
