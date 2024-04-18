import numpy as np
import pytest

import tripy as tp
from tests import helper
from tests.conftest import skip_if_older_than_sm89


def create_random_matrix(shape):
    return np.random.rand(*shape).astype(np.float32)


class TestLinear:
    @pytest.mark.parametrize(
        "use_jit",
        [False, True],
    )
    def test_linear_module(self, use_jit):
        class Network(tp.Module):
            def __init__(self):
                super().__init__()
                self.linear = tp.Linear(4, 2)

            def __call__(self, x):
                return self.linear(x)

        net = Network()
        a1 = tp.Tensor(np.ones((3, 4), dtype=np.float32), device=tp.device("gpu"))

        if use_jit:
            net = tp.jit(net)

        out = net(a1)

        np_out = np.ones((3, 4), dtype=np.float32) @ (np.ones((2, 4), dtype=np.float32).transpose()) + np.ones(
            (1, 2), dtype=np.float32
        )

        assert (out.numpy() == np.array(np_out)).all()


class TestQuantLinear:

    def _create_network(self, use_input_scale, quant_dtype, weight_quant_dim):
        out_feat = 2
        in_feat = 4

        def _get_dummy_scale(quant_dim):
            if quant_dim is None:
                scale = 1.0
            elif quant_dim == 0:
                scale = [1.0] * out_feat
            elif quant_dim == 1:
                scale = [1.0] * in_feat
            return tp.Parameter(scale)

        class Network(tp.Module):
            def __init__(self):
                super().__init__()
                self.linear = tp.Linear(
                    in_feat,
                    out_feat,
                    quant_dtype=quant_dtype,
                    weight_quant_dim=weight_quant_dim,
                )

            def __call__(self, x):
                return self.linear(x)

        net = Network()
        net.linear.weight_scale = _get_dummy_scale(weight_quant_dim)
        if use_input_scale:
            net.linear.input_scale = _get_dummy_scale(None)
        return net

    @pytest.mark.parametrize("use_jit", [False, True])
    @pytest.mark.parametrize("use_input_scale", [False, True])
    @pytest.mark.parametrize("quant_dtype", [tp.int8, pytest.param(tp.float8, marks=skip_if_older_than_sm89)])
    @pytest.mark.parametrize("weight_quant_dim", [None, 0, 1])
    def test_quant_linear(self, use_jit, use_input_scale, quant_dtype, weight_quant_dim):
        net = self._create_network(use_input_scale, quant_dtype, weight_quant_dim)
        if use_jit:
            net = tp.jit(net)

        a1 = tp.Tensor(np.ones((3, 4), dtype=np.float32), device=tp.device("gpu"))
        if use_input_scale and weight_quant_dim == 1:
            with helper.raises(
                tp.TripyException,
                match="Unsupported quantization parameters for Linear module.",
            ):
                out = net(a1)
        else:
            out = net(a1)

            np_out = np.ones((3, 4), dtype=np.float32) @ (np.ones((2, 4), dtype=np.float32).transpose()) + np.ones(
                (1, 2), dtype=np.float32
            )

            assert (out.numpy() == np.array(np_out)).all()

    @pytest.mark.parametrize("use_jit", [False, True])
    @pytest.mark.parametrize("quant_mode", ["block-wise", "per-tensor", "per-channel-0", "per-channel-1"])
    def test_quant_linear_int4_weight_only(self, use_jit, quant_mode):
        if quant_mode == "block-wise":
            weight_quant_dim = None
            scale = tp.Parameter(tp.ones((2, 4)))
        elif quant_mode == "per-tensor":
            weight_quant_dim = None
            scale = tp.Parameter(1.0)
        elif quant_mode.endswith("0"):
            weight_quant_dim = 0
            scale = tp.Parameter(tp.ones((8,)))
        elif quant_mode.endswith("1"):
            weight_quant_dim = 1
            scale = tp.Parameter(tp.ones((4,)))

        linear = tp.Linear(4, 8, quant_dtype=tp.int4, weight_quant_dim=weight_quant_dim)
        linear.weight_scale = scale
        if use_jit:
            linear = tp.jit(linear)

        input = tp.Tensor(np.ones((4, 4), dtype=np.float32), device=tp.device("gpu"))
        out = linear(input)

        np_out = np.ones((4, 4), dtype=np.float32) @ (np.ones((8, 4), dtype=np.float32).transpose()) + np.ones(
            (1, 8), dtype=np.float32
        )

        assert np.array_equal(out.numpy(), np_out)
