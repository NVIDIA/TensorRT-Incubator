from tripy.frontend import Tensor
from tripy.frontend.ops.select import Select, where


def test_select():
    a = Tensor([[0, 1], [0, 1]])
    b = Tensor([[0, 0], [1, 1]])
    condition = a >= b
    a = where(condition, a, b)
    assert isinstance(a, Tensor)
    assert isinstance(a.op, Select)
