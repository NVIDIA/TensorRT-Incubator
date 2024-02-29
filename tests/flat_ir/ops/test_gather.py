import tripy as tp
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import GatherOp
from tripy import int32


class TestGatherOp:
    def test_gather_str(self):
        data = tp.Tensor([3.0, 4.0], name="data")
        index = tp.Tensor([0], dtype=int32, name="indices")
        out = data.gather(0, index)
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        gather = flat_ir.ops[-1]
        reshape = flat_ir.ops[-2]

        print(str(reshape))
        assert isinstance(gather, GatherOp)
        assert str(gather) == "out: [shape=(1,), dtype=(float32), loc=(gpu:0)] = GatherOp(data, indices, axis=0)"
