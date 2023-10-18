from tripy.frontend import FlatIR, TensorExpression
from textwrap import dedent


class TestFlatIR:
    def test_single_layer_structure(self):
        a = TensorExpression.tensor([0])

        flat_ir = FlatIR([a])

        assert len(flat_ir.layers) == 1
        layer = flat_ir.layers[0]

        assert layer.params == a.params
        assert layer.inputs == []
        assert layer.output.name == "t0"

    def test_flat_ir_recurses_inputs(self):
        a = TensorExpression.tensor([0])
        b = TensorExpression.tensor([1])

        c = a + b

        flat_ir = FlatIR([c])

        assert len(flat_ir.layers) == 3
        names = {layer.output.name for layer in flat_ir.layers}

        assert names == {"t0", "t1", "t2"}

    def test_layers_are_topologically_sorted(self):
        a = TensorExpression.tensor([0])
        b = TensorExpression.tensor([1])

        c = a + b

        flat_ir = FlatIR([c])

        assert len(flat_ir.layers) == 3

        # The final layer should be 'c'. The ordering of 'a' and 'b' doesn't matter.
        assert flat_ir.layers[-1].output.name == "t2"

    def test_duplicate_traces_are_skipped(self):
        a = TensorExpression.tensor([0])
        b = TensorExpression.tensor([1])

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
            a = TensorExpression.tensor([0])
            b = TensorExpression.tensor([1])

            c = a + b
            return c

        # We do this in a loop so that the stack traces are identical
        irs = []
        for _ in range(2):
            irs.append(FlatIR([make_expr()]))

        assert irs[0] == irs[1]

    def test_str(self):
        a = TensorExpression.tensor([0])
        b = TensorExpression.tensor([1])

        c = a + b

        flat_ir = FlatIR([c])
        assert (
            str(flat_ir)
            == dedent(
                """
                Inputs: []
                Output: 't1 [[1]]'
                Parameters: ValueParameters(values=[1])

                Inputs: []
                Output: 't0 [[1]]'
                Parameters: ValueParameters(values=[0])

                Inputs: ['t0 [[1]]', 't1 [[1]]']
                Output: 't2 [[1]]'
                Parameters: BinaryElementwiseParameters(operation=<Operation.SUM: 0>)
                """
            ).strip()
        )
