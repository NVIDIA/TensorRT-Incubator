# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import numpy as np
import jax.numpy as jnp
from jax import jit


# Test equation: 'i'
@jit
def jit_einsum_i(x):
    return jnp.einsum("i", x)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_einsum_i(dtype):
    """Test einsum with equation 'i'."""
    x = np.arange(5).astype(dtype)
    np_result = np.einsum("i", x)
    jax_result = jit_einsum_i(x)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-6, atol=1e-6)


# Test equation: 'i->'
@jit
def jit_einsum_i_reduce(x):
    return jnp.einsum("i->", x)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_einsum_i_reduce(dtype):
    """Test einsum with equation 'i->'."""
    x = np.arange(5, 12).astype(dtype)
    np_result = np.einsum("i->", x)
    jax_result = jit_einsum_i_reduce(x)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-6, atol=1e-6)


# Test equation: 'ii'
@jit
def jit_einsum_ii(x):
    return jnp.einsum("ii", x)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_einsum_ii(dtype):
    """Test einsum with equation 'ii' (trace)."""
    x = np.arange(2 * 2).reshape(2, 2).astype(dtype)
    np_result = np.einsum("ii", x)
    jax_result = jit_einsum_ii(x)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-6, atol=1e-6)


# Test equation: 'ij, ij -> ij'
@jit
def jit_einsum_ij_ij_ij(x, y):
    return jnp.einsum("ij, ij -> ij", x, y)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_einsum_ij_ij_ij(dtype):
    """Test einsum with equation 'ij, ij -> ij' (element-wise multiply)."""
    x = np.arange(2 * 5).reshape(2, 5).astype(dtype)
    y = np.arange(2 * 5).reshape(2, 5).astype(dtype)
    np_result = np.einsum("ij, ij -> ij", x, y)
    jax_result = jit_einsum_ij_ij_ij(x, y)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-6, atol=1e-6)


# Test equation: 'i,i'
@jit
def jit_einsum_i_i(x, y):
    return jnp.einsum("i,i", x, y)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_einsum_i_i(dtype):
    """Test einsum with equation 'i,i' (dot product)."""
    x = np.arange(5).astype(dtype)
    y = np.arange(5).astype(dtype)
    np_result = np.einsum("i,i", x, y)
    jax_result = jit_einsum_i_i(x, y)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-6, atol=1e-6)


# Test equation: 'i,j'
@jit
def jit_einsum_i_j(x, y):
    return jnp.einsum("i,j", x, y)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_einsum_i_j(dtype):
    """Test einsum with equation 'i,j' (outer product)."""
    x = np.arange(5).astype(dtype)
    y = np.arange(5, 12).astype(dtype)
    np_result = np.einsum("i,j", x, y)
    jax_result = jit_einsum_i_j(x, y)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-6, atol=1e-6)


# Test equation: 'ij'
@jit
def jit_einsum_ij(x):
    return jnp.einsum("ij", x)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_einsum_ij(dtype):
    """Test einsum with equation 'ij' (identity)."""
    x = np.arange(2 * 5).reshape(2, 5).astype(dtype)
    np_result = np.einsum("ij", x)
    jax_result = jit_einsum_ij(x)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-6, atol=1e-6)


# Test equation: 'ii->i'
@jit
def jit_einsum_ii_i(x):
    return jnp.einsum("ii->i", x)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_einsum_ii_i(dtype):
    """Test einsum with equation 'ii->i' (diagonal)."""
    x = np.arange(2 * 2).reshape(2, 2).astype(dtype)
    np_result = np.einsum("ii->i", x)
    jax_result = jit_einsum_ii_i(x)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-6, atol=1e-6)


# Test equation: 'ij->ji'
@jit
def jit_einsum_ij_ji(x):
    return jnp.einsum("ij->ji", x)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_einsum_ij_ji(dtype):
    """Test einsum with equation 'ij->ji' (transpose)."""
    x = np.arange(2 * 5).reshape(2, 5).astype(dtype)
    np_result = np.einsum("ij->ji", x)
    jax_result = jit_einsum_ij_ji(x)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-6, atol=1e-6)


# Test equation: 'ij->i'
@jit
def jit_einsum_ij_i(x):
    return jnp.einsum("ij->i", x)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_einsum_ij_i(dtype):
    """Test einsum with equation 'ij->i' (sum over columns)."""
    x = np.arange(2 * 5).reshape(2, 5).astype(dtype)
    np_result = np.einsum("ij->i", x)
    jax_result = jit_einsum_ij_i(x)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-6, atol=1e-6)


# Test equation: 'ij,jk->ik'
@jit
def jit_einsum_ij_jk_ik(x, y):
    return jnp.einsum("ij,jk->ik", x, y)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_einsum_ij_jk_ik(dtype):
    """Test einsum with equation 'ij,jk->ik' (matrix multiply)."""
    x = np.arange(2 * 2).reshape(2, 2).astype(dtype)
    y = np.arange(2 * 5).reshape(2, 5).astype(dtype)
    np_result = np.einsum("ij,jk->ik", x, y)
    jax_result = jit_einsum_ij_jk_ik(x, y)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-6, atol=1e-6)


# Test equation: 'ij,jk->ki'
@jit
def jit_einsum_ij_jk_ki(x, y):
    return jnp.einsum("ij,jk->ki", x, y)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_einsum_ij_jk_ki(dtype):
    """Test einsum with equation 'ij,jk->ki' (matrix multiply with transpose)."""
    x = np.arange(2 * 2).reshape(2, 2).astype(dtype)
    y = np.arange(2 * 5).reshape(2, 5).astype(dtype)
    np_result = np.einsum("ij,jk->ki", x, y)
    jax_result = jit_einsum_ij_jk_ki(x, y)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-6, atol=1e-6)


# Test equation: 'ij,jh'
@jit
def jit_einsum_ij_jh(x, y):
    return jnp.einsum("ij,jh", x, y)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_einsum_ij_jh(dtype):
    """Test einsum with equation 'ij,jh' (implicit output)."""
    x = np.arange(2 * 2).reshape(2, 2).astype(dtype)
    y = np.arange(2 * 5).reshape(2, 5).astype(dtype)
    np_result = np.einsum("ij,jh", x, y)
    jax_result = jit_einsum_ij_jh(x, y)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-6, atol=1e-6)


# Test equation: 'ijkl->jilk'
@jit
def jit_einsum_ijkl_jilk(x):
    return jnp.einsum("ijkl->jilk", x)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_einsum_ijkl_jilk(dtype):
    """Test einsum with equation 'ijkl->jilk' (4D transpose)."""
    x = np.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4).astype(dtype)
    np_result = np.einsum("ijkl->jilk", x)
    jax_result = jit_einsum_ijkl_jilk(x)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-6, atol=1e-6)


# Test equation: 'bij,bjk->bik'
@jit
def jit_einsum_bij_bjk_bik(x, y):
    return jnp.einsum("bij,bjk->bik", x, y)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_einsum_bij_bjk_bik(dtype):
    """Test einsum with equation 'bij,bjk->bik' (batched matrix multiply)."""
    x = np.arange(2 * 2 * 3).reshape(2, 2, 3).astype(dtype)
    y = np.arange(2 * 3 * 2).reshape(2, 3, 2).astype(dtype)
    np_result = np.einsum("bij,bjk->bik", x, y)
    jax_result = jit_einsum_bij_bjk_bik(x, y)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-6, atol=1e-6)


# Test equation: 'ijk,jil->kl'
@jit
def jit_einsum_ijk_jil_kl(x, y):
    return jnp.einsum("ijk,jil->kl", x, y)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_einsum_ijk_jil_kl(dtype):
    """Test einsum with equation 'ijk,jil->kl' (complex contraction)."""
    x = np.arange(1 * 2 * 3).reshape(1, 2, 3).astype(dtype)
    y = np.arange(2 * 1 * 4).reshape(2, 1, 4).astype(dtype)
    np_result = np.einsum("ijk,jil->kl", x, y)
    jax_result = jit_einsum_ijk_jil_kl(x, y)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
