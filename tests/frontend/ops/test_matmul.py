import pytest

import tripy
from tripy.common import device
from tripy.frontend.ops import MatrixMultiplication
import numpy as np


def test_matmul():
    a = tripy.Tensor(np.random.rand(2, 3).astype(np.float32))
    b = tripy.Tensor(np.random.rand(3, 2).astype(np.float32))

    out = a @ b
    assert isinstance(a, tripy.Tensor)
    assert isinstance(out.op, MatrixMultiplication)
