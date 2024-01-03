import pytest

import tripy
from tripy.frontend.ops import BinaryElementwise


_BINARY_OPS = {
    BinaryElementwise.Kind.SUM: lambda a, b: a + b,
    BinaryElementwise.Kind.LESS: lambda a, b: a < b,
    BinaryElementwise.Kind.LESS_EQUAL: lambda a, b: a <= b,
    BinaryElementwise.Kind.EQUAL: lambda a, b: a == b,
    BinaryElementwise.Kind.NOT_EQUAL: lambda a, b: a != b,
    BinaryElementwise.Kind.GREATER_EQUAL: lambda a, b: a >= b,
    BinaryElementwise.Kind.GREATER: lambda a, b: a > b,
}


@pytest.mark.parametrize("func, kind", [(func, kind) for kind, func in _BINARY_OPS.items()])
def test_binary_elementwise(func, kind):
    a = tripy.Tensor([1])
    b = tripy.Tensor([2])

    out = func(a, b)
    assert isinstance(out, tripy.Tensor)
    assert isinstance(out.op, BinaryElementwise)
    assert out.op.kind == kind
