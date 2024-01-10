import numpy as np
from tripy.frontend import Tensor
from tripy.frontend.ops.transpose import Transpose, transpose, permute


def test_transpose():
    a = Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    a = transpose(a, 0, 1)
    assert isinstance(a, Tensor)
    assert isinstance(a.op, Transpose)


def test_permute():
    a = Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    a = permute(a, (1, 0))
    assert isinstance(a, Tensor)
    assert isinstance(a.op, Transpose)
