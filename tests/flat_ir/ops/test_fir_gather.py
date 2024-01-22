import tripy as tp
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import GatherOp, ReshapeOp
from tripy import int32


class TestGatherOp:
    def test_gather_str(self):
        data = tp.Tensor([3.0, 4.0])
        index = tp.Tensor([0], dtype=int32)
        out = tp.gather(data, index, axis=0)

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        gather = flat_ir.ops[-1]
        reshape = flat_ir.ops[-2]

        print(str(reshape))
        assert isinstance(gather, GatherOp)
        assert (
            str(gather)
            == "t2: [shape=(1,), dtype=(float32), loc=(gpu:0)] = GatherOp(t0, t1, offset_dims=(), axis=0, slice_sizes=[1], index_vector_dim=1)"
        )
