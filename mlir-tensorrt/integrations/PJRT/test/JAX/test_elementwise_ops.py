# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import numpy as np
import jax.numpy as jnp
from jax import jit


# Unary float operations
INPUT_TYPE_1 = np.linspace(0.1, 4, num=8, dtype=np.float32)


@jit
def jit_exp(x):
    return jnp.exp(x)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_exp(dtype):
    """Test jnp.exp with float dtypes."""
    x = INPUT_TYPE_1.astype(dtype)
    np_result = np.exp(x)
    jax_result = jit_exp(x)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-3, atol=1e-3)


@jit
def jit_exp2(x):
    return jnp.exp2(x)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_exp2(dtype):
    """Test jnp.exp2 with float dtypes."""
    x = INPUT_TYPE_1.astype(dtype)
    np_result = np.exp2(x)
    jax_result = jit_exp2(x)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-3, atol=1e-3)


@jit
def jit_negative(x):
    return jnp.negative(x)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_negative(dtype):
    """Test jnp.negative with float dtypes."""
    x = INPUT_TYPE_1.astype(dtype)
    np_result = np.negative(x)
    jax_result = jit_negative(x)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-3, atol=1e-3)


@jit
def jit_log(x):
    return jnp.log(x)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_log(dtype):
    """Test jnp.log with float dtypes."""
    x = INPUT_TYPE_1.astype(dtype)
    np_result = np.log(x)
    jax_result = jit_log(x)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-3, atol=1e-3)


@jit
def jit_log10(x):
    return jnp.log10(x)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_log10(dtype):
    """Test jnp.log10 with float dtypes."""
    x = INPUT_TYPE_1.astype(dtype)
    np_result = np.log10(x)
    jax_result = jit_log10(x)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-3, atol=1e-3)


@jit
def jit_log2(x):
    return jnp.log2(x)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_log2(dtype):
    """Test jnp.log2 with float dtypes."""
    x = INPUT_TYPE_1.astype(dtype)
    np_result = np.log2(x)
    jax_result = jit_log2(x)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-3, atol=1e-3)


@jit
def jit_log1p(x):
    return jnp.log1p(x)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_log1p(dtype):
    """Test jnp.log1p with float dtypes."""
    x = INPUT_TYPE_1.astype(dtype)
    np_result = np.log1p(x)
    jax_result = jit_log1p(x)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-3, atol=1e-3)


@jit
def jit_sinc(x):
    return jnp.sinc(x)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sinc(dtype):
    """Test jnp.sinc with float dtypes."""
    x = INPUT_TYPE_1.astype(dtype)
    np_result = np.sinc(x)
    jax_result = jit_sinc(x)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-3, atol=1e-3)


@jit
def jit_tanh(x):
    return jnp.tanh(x)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_tanh(dtype):
    """Test jnp.tanh with float dtypes."""
    x = INPUT_TYPE_1.astype(dtype)
    np_result = np.tanh(x)
    jax_result = jit_tanh(x)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-3, atol=1e-3)


@jit
def jit_cos(x):
    return jnp.cos(x)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cos(dtype):
    """Test jnp.cos with float dtypes."""
    x = INPUT_TYPE_1.astype(dtype)
    np_result = np.cos(x)
    jax_result = jit_cos(x)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-3, atol=1e-3)


@jit
def jit_sin(x):
    return jnp.sin(x)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sin(dtype):
    """Test jnp.sin with float dtypes."""
    x = INPUT_TYPE_1.astype(dtype)
    np_result = np.sin(x)
    jax_result = jit_sin(x)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-3, atol=1e-3)


@jit
def jit_sqrt(x):
    return jnp.sqrt(x)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sqrt(dtype):
    """Test jnp.sqrt with float dtypes."""
    x = INPUT_TYPE_1.astype(dtype)
    np_result = np.sqrt(x)
    jax_result = jit_sqrt(x)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-3, atol=1e-3)


@jit
def jit_square(x):
    return jnp.square(x)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_square(dtype):
    """Test jnp.square with float dtypes."""
    x = INPUT_TYPE_1.astype(dtype)
    np_result = np.square(x)
    jax_result = jit_square(x)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-3, atol=1e-3)


@jit
def jit_floor(x):
    return jnp.floor(x)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_floor(dtype):
    """Test jnp.floor with float dtypes."""
    x = np.asarray([1.23, 3.97, 4.4467, 5.555]).astype(dtype)
    np_result = np.floor(x)
    jax_result = jit_floor(x)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-3, atol=1e-3)


@jit
def jit_floor_divide(x, y):
    return jnp.floor_divide(x, y)


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_floor_divide(dtype):
    """Test jnp.floor_divide with int dtypes."""
    x = np.arange(1, 17).astype(dtype)
    y = np.asarray([2.5]).astype(dtype)
    np_result = np.floor_divide(x, y)
    jax_result = jit_floor_divide(x, y)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-3, atol=1e-3)


# Binary operations with broadcasting
@jit
def jit_greater(x, y):
    return jnp.greater(x, y)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("broadcast", [False, True])
def test_greater(dtype, broadcast):
    """Test jnp.greater with various dtypes and broadcasting."""
    if broadcast:
        x = np.arange(1, 17).reshape(4, 4).astype(dtype)
        y = np.arange(1, 5).astype(dtype)
    else:
        x = np.asarray([4, 2]).astype(dtype)
        y = np.asarray([2, 2]).astype(dtype)
    np_result = np.greater(x, y)
    jax_result = jit_greater(x, y)
    np.testing.assert_array_equal(np_result, jax_result)


@jit
def jit_greater_equal(x, y):
    return jnp.greater_equal(x, y)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("broadcast", [False, True])
def test_greater_equal(dtype, broadcast):
    """Test jnp.greater_equal with various dtypes and broadcasting."""
    if broadcast:
        x = np.arange(1, 17).reshape(4, 4).astype(dtype)
        y = np.arange(1, 5).astype(dtype)
    else:
        x = np.asarray([4, 2]).astype(dtype)
        y = np.asarray([2, 2]).astype(dtype)
    np_result = np.greater_equal(x, y)
    jax_result = jit_greater_equal(x, y)
    np.testing.assert_array_equal(np_result, jax_result)


@jit
def jit_less(x, y):
    return jnp.less(x, y)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("broadcast", [False, True])
def test_less(dtype, broadcast):
    """Test jnp.less with various dtypes and broadcasting."""
    if broadcast:
        x = np.arange(1, 17).reshape(4, 4).astype(dtype)
        y = np.arange(1, 5).astype(dtype)
    else:
        x = np.asarray([4, 2]).astype(dtype)
        y = np.asarray([2, 2]).astype(dtype)
    np_result = np.less(x, y)
    jax_result = jit_less(x, y)
    np.testing.assert_array_equal(np_result, jax_result)


@jit
def jit_less_equal(x, y):
    return jnp.less_equal(x, y)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("broadcast", [False, True])
def test_less_equal(dtype, broadcast):
    """Test jnp.less_equal with various dtypes and broadcasting."""
    if broadcast:
        x = np.arange(1, 17).reshape(4, 4).astype(dtype)
        y = np.arange(1, 5).astype(dtype)
    else:
        x = np.asarray([4, 2]).astype(dtype)
        y = np.asarray([2, 2]).astype(dtype)
    np_result = np.less_equal(x, y)
    jax_result = jit_less_equal(x, y)
    np.testing.assert_array_equal(np_result, jax_result)


@jit
def jit_equal(x, y):
    return jnp.equal(x, y)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("broadcast", [False, True])
def test_equal(dtype, broadcast):
    """Test jnp.equal with various dtypes and broadcasting."""
    if broadcast:
        x = np.arange(1, 17).reshape(4, 4).astype(dtype)
        y = np.arange(1, 5).astype(dtype)
    else:
        x = np.asarray([4, 2]).astype(dtype)
        y = np.asarray([2, 2]).astype(dtype)
    np_result = np.equal(x, y)
    jax_result = jit_equal(x, y)
    np.testing.assert_array_equal(np_result, jax_result)


@jit
def jit_not_equal(x, y):
    return jnp.not_equal(x, y)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("broadcast", [False, True])
def test_not_equal(dtype, broadcast):
    """Test jnp.not_equal with various dtypes and broadcasting."""
    if broadcast:
        x = np.arange(1, 17).reshape(4, 4).astype(dtype)
        y = np.arange(1, 5).astype(dtype)
    else:
        x = np.asarray([4, 2]).astype(dtype)
        y = np.asarray([2, 2]).astype(dtype)
    np_result = np.not_equal(x, y)
    jax_result = jit_not_equal(x, y)
    np.testing.assert_array_equal(np_result, jax_result)


@jit
def jit_add(x, y):
    return jnp.add(x, y)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("broadcast", [False, True])
def test_add(dtype, broadcast):
    """Test jnp.add with various dtypes and broadcasting."""
    if broadcast:
        x = np.arange(1, 17).reshape(4, 4).astype(dtype)
        y = np.arange(1, 5).astype(dtype)
    else:
        x = np.asarray([4, 2]).astype(dtype)
        y = np.asarray([2, 2]).astype(dtype)
    np_result = np.add(x, y)
    jax_result = jit_add(x, y)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-3, atol=1e-3)


@jit
def jit_subtract(x, y):
    return jnp.subtract(x, y)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("broadcast", [False, True])
def test_subtract(dtype, broadcast):
    """Test jnp.subtract with various dtypes and broadcasting."""
    if broadcast:
        x = np.arange(1, 17).reshape(4, 4).astype(dtype)
        y = np.arange(1, 5).astype(dtype)
    else:
        x = np.asarray([4, 2]).astype(dtype)
        y = np.asarray([2, 2]).astype(dtype)
    np_result = np.subtract(x, y)
    jax_result = jit_subtract(x, y)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-3, atol=1e-3)


@jit
def jit_multiply(x, y):
    return jnp.multiply(x, y)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("broadcast", [False, True])
def test_multiply(dtype, broadcast):
    """Test jnp.multiply with various dtypes and broadcasting."""
    if broadcast:
        x = np.arange(1, 17).reshape(4, 4).astype(dtype)
        y = np.arange(1, 5).astype(dtype)
    else:
        x = np.asarray([4, 2]).astype(dtype)
        y = np.asarray([2, 2]).astype(dtype)
    np_result = np.multiply(x, y)
    jax_result = jit_multiply(x, y)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-3, atol=1e-3)


@jit
def jit_divide(x, y):
    return jnp.divide(x, y)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("broadcast", [False, True])
def test_divide(dtype, broadcast):
    """Test jnp.divide with various dtypes and broadcasting."""
    if broadcast:
        x = np.arange(1, 17).reshape(4, 4).astype(dtype)
        y = np.arange(1, 5).astype(dtype)
    else:
        x = np.asarray([4, 2]).astype(dtype)
        y = np.asarray([2, 2]).astype(dtype)
    np_result = np.divide(x, y)
    jax_result = jit_divide(x, y)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-3, atol=1e-3)


# Logical operations
@jit
def jit_logical_and(x, y):
    return jnp.logical_and(x, y)


def test_logical_and():
    """Test jnp.logical_and with bool dtype."""
    x = np.asarray([True, False])
    y = np.asarray([False, False])
    np_result = np.logical_and(x, y)
    jax_result = jit_logical_and(x, y)
    np.testing.assert_array_equal(np_result, jax_result)


@jit
def jit_logical_or(x, y):
    return jnp.logical_or(x, y)


def test_logical_or():
    """Test jnp.logical_or with bool dtype."""
    x = np.asarray([True, False])
    y = np.asarray([False, False])
    np_result = np.logical_or(x, y)
    jax_result = jit_logical_or(x, y)
    np.testing.assert_array_equal(np_result, jax_result)


@jit
def jit_logical_xor(x, y):
    return jnp.logical_xor(x, y)


def test_logical_xor():
    """Test jnp.logical_xor with bool dtype."""
    x = np.asarray([True, False])
    y = np.asarray([False, False])
    np_result = np.logical_xor(x, y)
    jax_result = jit_logical_xor(x, y)
    np.testing.assert_array_equal(np_result, jax_result)


@jit
def jit_logical_not(x):
    return jnp.logical_not(x)


def test_logical_not():
    """Test jnp.logical_not with bool dtype."""
    x = np.asarray([True, False])
    np_result = np.logical_not(x)
    jax_result = jit_logical_not(x)
    np.testing.assert_array_equal(np_result, jax_result)


# NaN and inf operations
INPUT_TYPE_4 = np.asarray([jnp.inf, 5, jnp.nan, 6, 9.10, -jnp.inf, 7.0, 8.0])


@jit
def jit_isinf(x):
    return jnp.isinf(x)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_isinf(dtype):
    """Test jnp.isinf with float dtypes."""
    x = INPUT_TYPE_4.astype(dtype)
    np_result = np.isinf(x)
    jax_result = jit_isinf(x)
    np.testing.assert_array_equal(np_result, jax_result)


@jit
def jit_isnan(x):
    return jnp.isnan(x)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_isnan(dtype):
    """Test jnp.isnan with float dtypes."""
    x = INPUT_TYPE_4.astype(dtype)
    np_result = np.isnan(x)
    jax_result = jit_isnan(x)
    np.testing.assert_array_equal(np_result, jax_result)


@jit
def jit_isposinf(x):
    return jnp.isposinf(x)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_isposinf(dtype):
    """Test jnp.isposinf with float dtypes."""
    x = INPUT_TYPE_4.astype(dtype)
    np_result = np.isposinf(x)
    jax_result = jit_isposinf(x)
    np.testing.assert_array_equal(np_result, jax_result)


@jit
def jit_isneginf(x):
    return jnp.isneginf(x)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_isneginf(dtype):
    """Test jnp.isneginf with float dtypes."""
    x = INPUT_TYPE_4.astype(dtype)
    np_result = np.isneginf(x)
    jax_result = jit_isneginf(x)
    np.testing.assert_array_equal(np_result, jax_result)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
