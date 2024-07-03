import tripy as tp
from tripy import int32

from tripy.flat_ir.ops import DynamicGatherOp
from tripy.frontend.trace import Trace


class TestGatherOp:
    def test_gather_str(self):
        data = tp.Tensor([3.0, 4.0], name="data")
        index = tp.Tensor([0], dtype=int32, name="indices")
        out = tp.gather(data, 0, index)
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        gather = flat_ir.ops[-1]
        reshape = flat_ir.ops[-2]

        print(str(reshape))
        assert isinstance(gather, DynamicGatherOp)
        assert (
            str(gather)
            == "out: [rank=(1), shape=(?,), dtype=(float32), loc=(gpu:0)] = DynamicGatherOp(data, indices, t_inter3, axis=0)"
        )
