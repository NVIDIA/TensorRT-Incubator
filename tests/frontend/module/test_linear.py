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

    @pytest.mark.parametrize("quant_dtype", [tp.int8, tp.float8])
    @pytest.mark.parametrize("weight_quant_dim", [None, 0, 1])
    def test_quantized_params(self, quant_dtype, weight_quant_dim):
        qlinear = tp.Linear(
            20,
            30,
            quant_dtype=quant_dtype,
            weight_quant_dim=weight_quant_dim,
        )
        assert isinstance(qlinear, tp.Linear)
        assert qlinear.dtype == tp.float32
        assert qlinear.weight.numpy().shape == (30, 20)
        assert qlinear.bias.numpy().shape == (30,)
        assert qlinear.quant_dtype == quant_dtype
        assert not qlinear.weight_scale
        assert not qlinear.input_scale
        assert qlinear.weight_quant_dim == weight_quant_dim

    def test_load_quantized_params_from_state_dict(self):
        qlinear = tp.Linear(
            20,
            30,
            quant_dtype=tp.int8,
            weight_quant_dim=0,
        )

        qlinear.load_from_state_dict(
            {"weight_scale": tp.Parameter(tp.ones((20,))), "input_scale": tp.Parameter(tp.ones((20,)))}
        )
