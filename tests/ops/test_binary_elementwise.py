import pytest

from tripy.frontend import Tensor
from tripy.ops import BinaryElementwise


@pytest.mark.parametrize("func, kind", [(lambda a, b: a + b, BinaryElementwise.Kind.SUM)])
def test_binary_elementwise(func, kind):
    a = Tensor([1])
    b = Tensor([2])

    out = func(a, b)
    assert isinstance(out, Tensor)
    assert isinstance(out.op, BinaryElementwise)
    assert out.op.kind == kind
