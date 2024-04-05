import pytest
import tripy as tp
from tests import helper


class TestLinear:
    def test_linear_params(self):
        linear = tp.Linear(20, 30)
        assert isinstance(linear, tp.Linear)
        assert linear.weight.numpy().shape == (30, 20)
        assert linear.bias.numpy().shape == (30,)

    def test_mismatched_input_shapes(self):
        a = tp.ones((2, 3))
        linear = tp.Linear(2, 128)
        out = linear(a)

        with helper.raises(tp.TripyException, match="Incompatible input shapes.", has_stack_info_for=[a]):
            out.eval()

    @pytest.mark.parametrize("weight_quant_dim", [None, 0, 1])
    def test_quantized_params(self, weight_quant_dim):
        qlinear = tp.Linear(
            20,
            30,
            quant_dtype=tp.int8,
            weight_quant_dim=weight_quant_dim,
        )
        assert isinstance(qlinear, tp.Linear)
        assert qlinear.dtype == tp.float32
        assert qlinear.quant_dtype == tp.int8
        assert qlinear.weight.numpy().shape == (30, 20)
        assert qlinear.weight_scale is None
        assert qlinear.input_scale is None
        assert qlinear.bias.numpy().shape == (30,)
        assert qlinear.weight_quant_dim == weight_quant_dim
