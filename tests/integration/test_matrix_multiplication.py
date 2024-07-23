import cupy as cp
import numpy as np
import pytest

import tripy as tp
import tripy.common.datatype


class TestMatrixMultiplication:
    def test_2d_tensors(self):
        a_np = np.arange(6).reshape((2, 3)).astype(np.float32)
        b_np = np.arange(6).reshape((3, 2)).astype(np.float32)
        a = tp.Tensor(a_np)
        b = tp.Tensor(b_np)

        out = a @ b
        assert np.allclose(cp.from_dlpack(out).get(), a_np @ b_np)

    def test_1d_tensors(self):
        a_np = np.arange(64).astype(np.float32)  # 1D Tensor
        b_np = np.arange(64).astype(np.float32)  # 1D Tensor
        a = tripy.Tensor(cp.asanyarray(a_np))
        b = tripy.Tensor(cp.asanyarray(b_np))

        out = a @ b
        assert np.allclose(cp.from_dlpack(out).get(), a_np @ b_np, atol=1e-2)

    @pytest.mark.parametrize(
        "shape_a, shape_b",
        [
            ((3,), (3, 2)),  # 1D Tensor and 2D tensor
            ((3, 2), (2,)),  # 2D Tensor and 1D tensor
            ((2, 3, 4), (4, 2)),  # 3D tensor and 2D tensor
            ((3, 2, 3, 4), (4, 2)),  # 4D tensor and 2D tensor
            ((3, 2, 3), (1, 3, 2)),  # Broadcasting batch dimension
            ((1, 2, 3), (0, 0, 3, 2)),  # Broadcasting batch dims with 0
        ],
    )
    def test_broadcast_gemm(self, shape_a, shape_b):
        a_np = np.arange(np.prod(shape_a)).reshape(shape_a).astype(np.float32)
        b_np = np.arange(np.prod(shape_b)).reshape(shape_b).astype(np.float32)
        a = tp.Tensor(a_np)
        b = tp.Tensor(b_np)

        out = a @ b
        assert np.allclose(cp.from_dlpack(out).get(), a_np @ b_np)
