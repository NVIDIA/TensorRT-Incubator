import pytest

import numpy as np

import tripy as tp

_UNARY_OPS = {
    tp.exp: np.exp,
    tp.tanh: np.tanh,
    tp.rsqrt: lambda a: 1.0 / np.sqrt(a),
    tp.log: np.log,
    tp.sin: np.sin,
    tp.cos: np.cos,
}


class TestUnaryElementwise:
    @pytest.mark.parametrize("tp_func, np_func", [(tp_func, np_func) for tp_func, np_func in _UNARY_OPS.items()])
    def test_op_funcs(self, tp_func, np_func):
        input = tp.arange(1, 4, dtype=tp.float32)
        output = tp_func(input)
        assert np.allclose(output.numpy(), np_func(input.numpy()))
