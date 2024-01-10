from tripy.frontend import Tensor
from tripy.frontend.ops.iota import Iota, arange, arange_like


def test_arange():
    a = arange([2, 3])
    assert isinstance(a, Tensor)
    assert isinstance(a.op, Iota)


def test_arange_like():
    t = Tensor([1, 2, 3])
    a = arange_like(t)
    assert isinstance(a, Tensor)
    assert isinstance(a.op, Iota)
