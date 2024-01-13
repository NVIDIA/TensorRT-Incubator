from tripy.frontend import Tensor
from tripy.frontend.ops.slice import Slice


def test_slice_all_partial():
    a = Tensor([1, 2, 3, 4])
    a = a[:2]
    assert isinstance(a, Tensor)
    assert isinstance(a.op, Slice)
