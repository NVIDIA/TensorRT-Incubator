from textwrap import dedent

import numpy as np

from tripy.frontend import Tensor
from tripy.flat_ir import FlatIR


class TestFlatIR:
    def test_single_layer_structure(self):
        a = Tensor([0])

        flat_ir = FlatIR([a])

        assert len(flat_ir.layers) == 1
        layer = flat_ir.layers[0]

        assert layer.op == a.op
        assert layer.inputs == []
        assert layer.outputs[0].name == "t0"

    def test_flat_ir_recurses_inputs(self):
        a = Tensor([0])
        b = Tensor([1])

        c = a + b

        flat_ir = FlatIR([c])

        assert len(flat_ir.layers) == 3
        names = {layer.outputs[0].name for layer in flat_ir.layers}

        assert names == {"t0", "t1", "t2"}

    def test_layers_are_topologically_sorted(self):
        a = Tensor([0])
        b = Tensor([1])

        c = a + b

        flat_ir = FlatIR([c])

        assert len(flat_ir.layers) == 3

        # The final layer should be 'c'. The ordering of 'a' and 'b' doesn't matter.
        assert flat_ir.layers[-1].outputs[0].name == "t2"

    def test_duplicate_traces_are_skipped(self):
        a = Tensor([0])
        b = Tensor([1])

        c = a + b
        # In a naive implementation, we might end up tracing the `c` expression twice.
        # Our implementation should not do that.
        d = c + c

        flat_ir = FlatIR([d])

        # If we end up tracing `c` twice, we'll end up with 7 layers: [a, b, a, b, c, c, d].
        # Without duplication, we should just have [a, b, c, d].
        assert len(flat_ir.layers) == 4

    # For a given program, we should generate identical FlatIRs each time.
    def test_ir_consistent_across_runs(self):
        def make_expr():
            a = Tensor([0])
            b = Tensor([1])

            c = a + b
            return c

        # We do this in a loop so that the stack traces are identical
        irs = []
        for _ in range(2):
            irs.append(FlatIR([make_expr()]))

        assert irs[0] == irs[1]

    def test_str(self):
        a = Tensor([0])
        b = Tensor([1])

        c = a + b

        flat_ir = FlatIR([c])

        print(flat_ir)  # Makes it easier to debug when the test fails.
        assert (
            str(flat_ir)
            == dedent(
                """
                t0 : data=([0.]), shape=(), stride=(), loc=(cpu:0)
                t1 : data=([1.]), shape=(), stride=(), loc=(cpu:0)
                t2 = t0 + t1
                outputs: t2
                """
            ).strip()
        )

    def test_infer_shapes(self):
        shape = (5, 5)
        a = Tensor(np.ones(shape))
        b = Tensor(np.ones(shape))

        c = a + b

        flat_ir = FlatIR([c])
        flat_ir.infer_shapes()

        assert flat_ir.layers[-1].outputs[0].shape == shape

    def test_multiple_outputs(self):
        shape = 1
        a = Tensor(np.ones(shape))
        b = Tensor(np.ones(shape))

        c = a + b
        d = c + c

        # The order c,d is important to test topological sort correctness, since if its d,c the dependencies are managed automatically.
        flat_ir = FlatIR([c, d])
        print(flat_ir)
        assert (
            str(flat_ir)
            == dedent(
                """
                t0 : data=([1.]), shape=(), stride=(), loc=(cpu:0)
                t1 : data=([1.]), shape=(), stride=(), loc=(cpu:0)
                t2 = t0 + t1
                t3 = t2 + t2
                outputs: t2, t3
                """
            ).strip()
        )
