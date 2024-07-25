import pytest

import numpy as np

import tripy as tp
from tests import helper
from tripy.frontend.trace.ops import MatrixMultiplication


class TestMatMul:
    def test_op_func(self):
        a = tp.Tensor(np.arange(6).reshape((2, 3)).astype(np.float32))
        b = tp.Tensor(np.arange(6).reshape((2, 3))[::-1].astype(np.float32))
        out = a @ b
        assert isinstance(a, tp.Tensor)
        assert isinstance(out.trace_tensor.producer, MatrixMultiplication)

    def test_0d_matrix_fails(self):
        a = tp.ones(tuple(), dtype=tp.float32)
        b = tp.ones((2,), dtype=tp.float32)

        with helper.raises(
            tp.TripyException, match="Input tensors must have at least 1 dimension.", has_stack_info_for=[a, b]
        ):
            c = a @ b
            c.eval()

    def test_mismatched_dtypes_fails(self):
        a = tp.ones((2, 3), dtype=tp.float32)
        b = tp.ones((3, 2), dtype=tp.float16)

        with helper.raises(tp.TripyException, match="Incompatible input data types.", has_stack_info_for=[a, b]):
            c = a @ b

    @pytest.mark.skip(
        "https://gitlab-master.nvidia.com/initialdl/mlir-tensorrt/-/issues/860 fixes dynamic broadcast issue."
    )
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

        with helper.raises(
            tp.TripyException, match="contracting dimension sizes must match for lhs/rhs", has_stack_info_for=[a, b, c]
        ):
            c.eval()

    @pytest.mark.parametrize(
        "a, b, expected_rank",
        [
            (
                tp.ones((2,)),
                tp.ones((2,)),
                0,
            ),
            (tp.ones((2, 3)), tp.ones((3, 2)), 2),
            (tp.ones((4, 2, 3)), tp.ones((3, 2)), 3),
        ],
    )
    def test_infer_rank(self, a, b, expected_rank):
        out = a @ b
        assert out.trace_tensor.rank == expected_rank
