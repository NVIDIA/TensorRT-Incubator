import pytest

import numpy as np
import tripy as tp
from tripy.frontend.ops.reshape import Reshape, Squeeze


class TestReshape:
    def test_op_func(self):
        a = tp.Tensor([1, 2, 3, 4])
        a = a.reshape((1, 1, 4))
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.op, Reshape)


class TestSqueeze:
    def test_op_func(self):
        a = tp.Tensor(np.ones((1, 1, 4), dtype=np.int32))
        a = a.squeeze()
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.op, Squeeze)

    def test_incorrect_dims(self):
        a = tp.Tensor(np.ones((1, 1, 4), dtype=np.int32))
        a = a.squeeze(2)

        with pytest.raises(
            tp.TripyException, match="Cannot select an axis to squeeze out which has size not equal to one"
        ) as exc:
            a.eval()
        print(str(exc.value))
