from tripy.frontend import Tensor
from tripy.frontend.ops.fill import Fill, ones


def test_fill():
    a = ones([1, 2])
    assert isinstance(a, Tensor)
    assert isinstance(a.op, Fill)
