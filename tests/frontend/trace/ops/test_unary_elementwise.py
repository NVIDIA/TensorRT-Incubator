import pytest

import tripy as tp
from tripy.frontend.trace.ops import UnaryElementwise


_UNARY_OPS = {
    UnaryElementwise.Kind.EXP: tp.exp,
    UnaryElementwise.Kind.TANH: tp.tanh,
    UnaryElementwise.Kind.RSQRT: tp.rsqrt,
    UnaryElementwise.Kind.LOG: tp.log,
    UnaryElementwise.Kind.SQRT: tp.sqrt,
    UnaryElementwise.Kind.ABS: tp.abs,
}


class TestUnaryElementwise:
    @pytest.mark.parametrize("func, kind", [(func, kind) for kind, func in _UNARY_OPS.items()])
    def test_op_funcs(self, func, kind):
        a = tp.Tensor([1.0])

        out = func(a)
        assert isinstance(out, tp.Tensor)
        assert isinstance(out.trace_tensor.producer, UnaryElementwise)
        assert out.trace_tensor.producer.kind == kind

    @pytest.mark.parametrize("func, kind", [(func, kind) for kind, func in _UNARY_OPS.items()])
    def test_infer_rank(self, func, kind):
        a = tp.ones((2, 3))
        out = func(a)
        assert out.trace_tensor.rank == 2
