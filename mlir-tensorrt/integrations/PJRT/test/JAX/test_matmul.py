# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import numpy as np
import jax.numpy as jnp
from jax import jit


# Test case 1: N-D mat @ N-D mat (batched matrix multiply)
@jit
def jit_matmul_batched(x, y):
    return jnp.matmul(x, y)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_matmul_batched(dtype):
    """Test batched matrix multiplication (N-D @ N-D)."""
    x = np.arange(4 * 12).reshape(4, 3, 4).astype(dtype)
    y = np.arange(4 * 4).reshape(4, 4, 1).astype(dtype)
    np_result = np.matmul(x, y)
    jax_result = jit_matmul_batched(x, y)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-6, atol=1e-6)


# Test case 2: 2D mat @ 2D mat (conventional matrix multiply)
@jit
def jit_matmul_2d(x, y):
    return jnp.matmul(x, y)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_matmul_2d(dtype):
    """Test conventional 2D matrix multiplication."""
    x = np.arange(12).reshape(4, 3).astype(dtype)
    y = np.arange(6).reshape(3, 2).astype(dtype)
    np_result = np.matmul(x, y)
    jax_result = jit_matmul_2d(x, y)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-6, atol=1e-6)


# Test case 3: 1D vec @ 2D mat
@jit
def jit_matmul_vec_mat(x, y):
    return jnp.matmul(x, y)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_matmul_vec_mat(dtype):
    """Test vector @ matrix multiplication."""
    x = np.arange(4).astype(dtype)
    y = np.arange(12).reshape(4, 3).astype(dtype)
    np_result = np.matmul(x, y)
    jax_result = jit_matmul_vec_mat(x, y)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-6, atol=1e-6)


# Test case 4: 2D mat @ 1D vec
@jit
def jit_matmul_mat_vec(x, y):
    return jnp.matmul(x, y)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_matmul_mat_vec(dtype):
    """Test matrix @ vector multiplication."""
    x = np.arange(6).reshape(3, 2).astype(dtype)
    y = np.arange(2).astype(dtype)
    np_result = np.matmul(x, y)
    jax_result = jit_matmul_mat_vec(x, y)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
