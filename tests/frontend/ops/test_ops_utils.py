from tripy.frontend.ops.utils import to_dims
from tripy.frontend.dim import Dim


def test_to_dims():
    assert to_dims((2, 3, 4)) == (Dim(2), Dim(3), Dim(4))
    assert to_dims((2, Dim(3), 4)) == (Dim(2), Dim(3), Dim(4))
    assert to_dims(None) == None
    assert to_dims(()) == ()
