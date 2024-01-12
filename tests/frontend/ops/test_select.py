from tripy.frontend import Tensor
from tripy.frontend.ops.select import Select, where


def test_select():
    a = Tensor([0, 1, 0, 1])
    b = Tensor([0, 0, 1, 0])
    condition = a >= b
    a = where(condition, a, b)
    assert isinstance(a, Tensor)
    assert isinstance(a.op, Select)


def test_masked_fill():
    a = Tensor([0, 1, 0, 1])
    b = Tensor([0, 0, 1, 0])
    mask = a == b
    a = a.masked_fill(mask, -1)
    assert isinstance(a, Tensor)
    assert isinstance(a.op, Select)
