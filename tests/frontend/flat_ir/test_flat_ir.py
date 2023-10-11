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
                Output: 't1'
                Parameters: ValueParameters(values=[1])

                Inputs: []
                Output: 't0'
                Parameters: ValueParameters(values=[0])

                Inputs: ['t0', 't1']
                Output: 't2'
                Parameters: BinaryElementwiseParameters(operation=<Operation.SUM: 0>)
                """
            ).strip()
        )
