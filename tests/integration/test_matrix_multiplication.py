import numpy as np
import pytest

import tripy.common.datatype


def create_random_matrix(shape):
    return np.random.rand(*shape).astype(np.float32)


class TestMatrixMultiplication:
    def test_2d_tensors(self):
        a_np = create_random_matrix((2, 3))
        b_np = create_random_matrix((3, 2))
        a = tripy.Tensor(a_np)
        b = tripy.Tensor(b_np)

        out = a @ b
        assert np.allclose(out.numpy(), a_np @ b_np)

    def test_1d_tensors(self):
        a_np = create_random_matrix((3,))  # 1D Tensor
        b_np = create_random_matrix((3,))  # 1D Tensor
        a = tripy.Tensor(a_np)
        b = tripy.Tensor(b_np)

        out = a @ b
        assert np.allclose(out.numpy(), a_np @ b_np)

    @pytest.mark.parametrize(
        "shape_a, shape_b",
        [
            ((3,), (3, 2)),  # 1D Tensor and 2D tensor
            ((2, 3, 4), (3, 2)),  # 3D Tensor and 2D Tensor
            ((2, 3), (4, 3, 2)),  # 2D Tensor and 3D Tensor
        ],
    )
    def test_invalid_dimensions(self, shape_a, shape_b):
        a_np = create_random_matrix(shape_a)
        b_np = create_random_matrix(shape_b)
        a = tripy.Tensor(a_np)
        b = tripy.Tensor(b_np)

        with pytest.raises(Exception):
            out = a @ b
            out.numpy()
