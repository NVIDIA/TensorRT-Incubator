from tripy.frontend import Tensor
from tripy.frontend.ops.iota import Iota, arange


def test_iota():
    a = arange([2, 3])
    assert isinstance(a, Tensor)
    assert isinstance(a.op, Iota)
