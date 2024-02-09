import numpy as np

import tripy as tp
from tests import helper
from tripy.frontend.trace.ops import MatrixMultiplication


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

        with helper.raises(
            tp.TripyException, match="Input tensors must have at least 1 dimension.", has_stack_info_for=[a, b, c]
        ):
            c.eval()

    def test_mismatched_dtypes_fails(self):
        a = tp.ones((2, 3), dtype=tp.float32)
        b = tp.ones((3, 2), dtype=tp.float16)
        c = a @ b

        with helper.raises(tp.TripyException, match="Incompatible input data types.", has_stack_info_for=[a, b, c]):
            c.eval()

    def test_incompatible_1d_shapes_fails(self):
        a = tp.ones((2,), dtype=tp.float32)
        b = tp.ones((3,), dtype=tp.float32)
        c = a @ b

        with helper.raises(tp.TripyException, match="Incompatible input shapes.", has_stack_info_for=[a, b, c]):
            c.eval()

    def test_incompatible_2d_shapes_fails(self):
        a = tp.ones((2, 4), dtype=tp.float32)
        b = tp.ones((3, 6), dtype=tp.float32)
        c = a @ b

        with helper.raises(tp.TripyException, match="Incompatible input shapes.", has_stack_info_for=[a, b, c]):
            c.eval()
