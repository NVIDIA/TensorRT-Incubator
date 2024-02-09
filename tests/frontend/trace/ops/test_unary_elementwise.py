import pytest

import tripy as tp
from tripy.frontend.trace.ops import UnaryElementwise


_UNARY_OPS = {
    UnaryElementwise.Kind.EXP: lambda a: a.exp(),
    UnaryElementwise.Kind.TANH: lambda a: a.tanh(),
    UnaryElementwise.Kind.RSQRT: lambda a: a.rsqrt(),
}


class TestUnaryElementwise:
    @pytest.mark.parametrize("func, kind", [(func, kind) for kind, func in _UNARY_OPS.items()])
    def test_op_funcs(self, func, kind):
        a = tp.Tensor([1.0])

        out = func(a)
        assert isinstance(out, tp.Tensor)
        assert isinstance(out.op, UnaryElementwise)
        assert out.op.kind == kind
