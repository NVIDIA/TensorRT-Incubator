import numpy as np
import pytest

import tripy as tp
from tripy.frontend.ops import MatrixMultiplication


class TestMatMul:
    def test_op_func(self):
        a = tp.Tensor(np.random.rand(2, 3).astype(np.float32))
        b = tp.Tensor(np.random.rand(3, 2).astype(np.float32))

        out = a @ b
        assert isinstance(a, tp.Tensor)
        assert isinstance(out.op, MatrixMultiplication)

    def test_0d_matrix_fails(self):
        a = tp.ones(tuple(), dtype=tp.float32)
        b = tp.ones((2,), dtype=tp.float32)
        c = a @ b

        with pytest.raises(tp.TripyException, match="Input tensors must have at least 1 dimension.") as exc:
            c.eval()
        print(str(exc.value))

    def test_mismatched_dtypes_fails(self):
        a = tp.ones((2, 3), dtype=tp.float32)
        b = tp.ones((3, 2), dtype=tp.float16)
        c = a @ b

        with pytest.raises(tp.TripyException, match="Mismatched input data types.") as exc:
            c.eval()
        print(str(exc.value))
