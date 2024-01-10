from tripy.frontend import Tensor
from tripy.frontend.ops.fill import Fill, full, full_like


def test_full():
    a = full([1, 2], 1)
    assert isinstance(a, Tensor)
    assert isinstance(a.op, Fill)


def test_full_like():
    t = Tensor([[1, 2], [3, 4]])
    a = full_like(t, 1)
    assert isinstance(a, Tensor)
    assert isinstance(a.op, Fill)
