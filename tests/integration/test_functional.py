import numpy as np
import pytest

from tripy.frontend import Tensor


class TestFunctional:
    def test_add_two_tensors(self):
        arr = np.array([2, 3], dtype=np.float32)
        a = Tensor(arr)
        b = Tensor(np.ones(2, dtype=np.float32))

        c = a + b
        out = c + c
        assert (out.eval() == np.array([6.0, 8.0])).all()
