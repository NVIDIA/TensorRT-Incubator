import pytest

import numpy as np

import tripy as tp

_UNARY_OPS = {
    lambda a: a.exp(): np.exp,
    lambda a: a.tanh(): np.tanh,
    lambda a: a.rsqrt(): lambda a: 1.0 / np.sqrt(a),
    lambda a: a.log(): np.log,
}


class TestUnaryElementwise:
    @pytest.mark.parametrize("tp_func, np_func", [(tp_func, np_func) for tp_func, np_func in _UNARY_OPS.items()])
    def test_op_funcs(self, tp_func, np_func):
        input = tp.arange(1, 4, dtype=tp.float32)
        output = tp_func(input)
        assert np.allclose(output.numpy(), np_func(input.numpy()))
