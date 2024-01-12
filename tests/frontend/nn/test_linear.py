import pytest
import tripy as tp


class TestLinear:
    def test_linear_params(self):
        linear = tp.nn.Linear(20, 30)
        assert isinstance(linear, tp.nn.Linear)
        params = linear.parameters()
        assert len(params.keys()) == 2
        assert params["weight"].numpy().shape == (30, 20)
        assert params["bias"].numpy().shape == (1, 30)

    def test_mismatched_input_shapes(self):
        a = tp.ones((2, 3))
        linear = tp.nn.Linear(2, 128)
        out = linear(a)

        with pytest.raises(tp.TripyException, match="Incompatible input shapes.") as exc:
            out.eval()
        print(str(exc.value))
