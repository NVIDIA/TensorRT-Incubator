import tripy as tp
from tests import helper


class TestLinear:
    def test_linear_params(self):
        linear = tp.nn.Linear(20, 30)
        assert isinstance(linear, tp.nn.Linear)
        assert linear.weight.numpy().shape == (30, 20)
        assert linear.bias.numpy().shape == (30,)

    def test_mismatched_input_shapes(self):
        a = tp.ones((2, 3))
        linear = tp.nn.Linear(2, 128)
        out = linear(a)

        with helper.raises(tp.TripyException, match="Incompatible input shapes.", has_stack_info_for=[a]):
            out.eval()
