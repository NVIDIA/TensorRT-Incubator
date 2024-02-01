from textwrap import dedent

import numpy as np

from tripy.common.device import device
from tripy.frontend import Tensor
from tripy.frontend.trace import Trace


class TestTrace:
    def test_single_layer_structure(self):
        a = Tensor([0])

        trace = Trace([a])

        assert len(trace.ops) == 1
        layer = trace.ops[0]

        assert layer == a.op
        assert layer.inputs == []
        assert layer.outputs[0].name == "t0"

    def test_trace_recurses_inputs(self):
        a = Tensor([0])
        b = Tensor([1])

        c = a + b

        trace = Trace([c])

        assert len(trace.ops) == 3
        names = {layer.outputs[0].name for layer in trace.ops}

        assert names == {"t0", "t1", "t2"}

    def test_layers_are_topologically_sorted(self):
        a = Tensor([0])
        b = Tensor([1])

        c = a + b

        trace = Trace([c])

        assert len(trace.ops) == 3

        # The final layer should be 'c'. The ordering of 'a' and 'b' doesn't matter.
        assert trace.ops[-1].outputs[0].name == "t2"

    def test_duplicate_traces_are_skipped(self):
        a = Tensor([0])
        b = Tensor([1])

        c = a + b
        # In a naive implementation, we might end up tracing the `c` expression twice.
        # Our implementation should not do that.
        d = c + c

        trace = Trace([d])

        # If we end up tracing `c` twice, we'll end up with 7 layers: [a, b, a, b, c, c, d].
        # Without duplication, we should just have [a, b, c, d].
        assert len(trace.ops) == 4

    def test_str(self):
        a = Tensor([0])
        b = Tensor([1])

        c = a + b

        trace = Trace([c])

        print(trace)  # Makes it easier to debug when the test fails.
        assert (
            str(trace)
            == dedent(
                """
                t0 = storage(data=[0], shape=(1,), dtype=int32, device=cpu:0)
                t1 = storage(data=[1], shape=(1,), dtype=int32, device=cpu:0)
                t2 = t0 + t1
                outputs:
                    t2: [shape=(1,), dtype=(int32), loc=(gpu:0)]
                """
            ).strip()
        )

    def test_infer_tensor_info(self):
        shape = (5, 5)
        a = Tensor(np.ones(shape, dtype=np.float32))
        b = Tensor(np.ones(shape, dtype=np.float32))

        c = a + b

        trace = Trace([c])

        assert trace.ops[-1].outputs[0].shape == shape
        assert trace.ops[-1].outputs[0].dtype == a.op.dtype
        assert trace.ops[-1].outputs[0].device == device("gpu")

    def test_multiple_outputs(self):
        shape = 1
        a = Tensor(np.ones(shape, dtype=np.float32))
        b = Tensor(np.ones(shape, dtype=np.float32))

        c = a + b
        d = c + c

        # The order c,d is important to test topological sort correctness, since if its d,c the dependencies are managed automatically.
        trace = Trace([c, d])
        print(trace)
        assert (
            str(trace)
            == dedent(
                """
                t0 = storage(data=[1.], shape=(1,), dtype=float32, device=cpu:0)
                t1 = storage(data=[1.], shape=(1,), dtype=float32, device=cpu:0)
                t2 = t0 + t1
                t3 = t2 + t2
                outputs:
                    t2: [shape=(1,), dtype=(float32), loc=(gpu:0)]
                    t3: [shape=(1,), dtype=(float32), loc=(gpu:0)]
                """
            ).strip()
        )

    def test_input_output(self):
        a = Tensor([1, 1])
        # a is an input
        trace = Trace([a], [a])
        assert len(trace.inputs) == 1
        assert len(trace.outputs) == 1
        assert len(trace.ops) == 0

    def test_all_inputs(self):
        shape = 1
        # Need explicit data type here since by default dtype is np.float64 which is not yet supported.
        # (38): Add cast operation to support unsupported backend types.
        a = Tensor(np.ones(shape, dtype=np.float32))
        b = Tensor(np.ones(shape, dtype=np.float32))
        c = a + b
        # a and b are inputs
        trace = Trace([c], [a, b])
        print(trace)
        assert (
            str(trace)
            == dedent(
                """
                inputs:
                    t0: [shape=(1,), dtype=(float32), loc=(gpu:0)]
                    t1: [shape=(1,), dtype=(float32), loc=(gpu:0)]
                t2 = t0 + t1
                outputs:
                    t2: [shape=(1,), dtype=(float32), loc=(gpu:0)]
                """
            ).strip()
        )

    def test_const_and_input(self):
        shape = 1
        a = Tensor(np.ones(shape, dtype=np.float32))
        b = Tensor(np.ones(shape, dtype=np.float32))
        c = a + b
        # a is an input
        trace = Trace([c], [a])
        print(trace)
        assert (
            str(trace)
            == dedent(
                """
                inputs:
                    t0: [shape=(1,), dtype=(float32), loc=(gpu:0)]
                t1 = storage(data=[1.], shape=(1,), dtype=float32, device=cpu:0)
                t2 = t0 + t1
                outputs:
                    t2: [shape=(1,), dtype=(float32), loc=(gpu:0)]
                """
            ).strip()
        )
