import pytest

from tripy.frontend import Tensor
from tripy.frontend.tensor_ops import ones
from tripy.frontend.ops.shape import ShapeOf


def test_shape():
    a = ones((3, 4))
    shape_a = a.shape
    assert isinstance(a, Tensor)
    assert isinstance(shape_a, Tensor)
