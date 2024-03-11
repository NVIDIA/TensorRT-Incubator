import tripy as tp
from tripy.frontend.trace import Trace


class TestFlatIR:
    def test_tensor_connectivity(self):
        # When we build up a FlatIR with multiple layers, the tensors/ops
        # should be connected to each other - i.e. the producer/inputs fields
        # should let you walk through the entire FlatIR.
        inp = tp.Tensor([0])

        b = tp.tanh(inp)
        out = tp.tanh(b)

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        # Check that `b` is connected to `inp`
        assert flat_ir.ops[1].inputs[0].producer is flat_ir.ops[0]
        assert flat_ir.ops[1].inputs[0] is flat_ir.ops[0].outputs[0]

        # Check that `out` is connected to `b`
        assert flat_ir.ops[2].inputs[0].producer is flat_ir.ops[1]
        assert flat_ir.ops[2].inputs[0] is flat_ir.ops[1].outputs[0]
