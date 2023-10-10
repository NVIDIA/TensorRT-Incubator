from tripy.frontend import FlatIR, TensorExpression


class TestFlatIR:
    def test_single_layer_structure(self):
        a = TensorExpression.tensor([0])

        flat_ir = FlatIR([a])

        assert len(flat_ir.layers) == 1
        layer = flat_ir.layers[0]

        assert layer.params == a.params
        assert layer.inputs == []
        assert layer.output.id == id(a)

    def test_flat_ir_recurses_inputs(self):
        a = TensorExpression.tensor([0])
        b = TensorExpression.tensor([1])

        c = a + b

        flat_ir = FlatIR([c])

        assert len(flat_ir.layers) == 3
        ids = {layer.output.id for layer in flat_ir.layers}

        assert ids == {id(a), id(b), id(c)}

    def test_layers_are_topologically_sorted(self):
        a = TensorExpression.tensor([0])
        b = TensorExpression.tensor([1])

        c = a + b

        flat_ir = FlatIR([c])

        assert len(flat_ir.layers) == 3

        # The final layer should be 'c'. The ordering of 'a' and 'b' doesn't matter.
        assert flat_ir.layers[-1].output.id == id(c)
