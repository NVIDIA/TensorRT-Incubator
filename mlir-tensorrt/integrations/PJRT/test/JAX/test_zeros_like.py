# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def jit_zeros_like(i_f32_1d, i_f32_2d, i_f16_1d, i_i8_1d):
    o_f32_1d = jnp.zeros_like(i_f32_1d)
    o_f32_2d = jnp.zeros_like(i_f32_2d)
    o_f16_1d = jnp.zeros_like(i_f16_1d)
    o_i8_1d = jnp.zeros_like(i_i8_1d)
    return (o_f32_1d, o_f32_2d, o_f16_1d, o_i8_1d)


def zeros_like_np(i_f32_1d, i_f32_2d, i_f16_1d, i_i8_1d):
    e_f32_1d = np.zeros_like(i_f32_1d)
    e_f32_2d = np.zeros_like(i_f32_2d)
    e_f16_1d = np.zeros_like(i_f16_1d)
    e_i8_1d = np.zeros_like(i_i8_1d)
    return (e_f32_1d, e_f32_2d, e_f16_1d, e_i8_1d)


def test_zeros_like():
    """Test jnp.zeros_like with various dtypes."""
    i_f32_1d = np.asarray([-1.2, 2.4, 4.0, 5.6], dtype=jnp.float32)
    i_f32_2d = np.asarray([[-1.2, -2.4], [4.0, -5.6]], dtype=jnp.float32)
    i_f16_1d = np.asarray([-1.2, 2.4, 4.0, 5.6], dtype=jnp.float16)
    i_i8_1d = np.asarray([-1, 2, 4, 5], dtype=jnp.int8)

    i_f32_1d_np = np.array([-1.2, 2.4, 4.0, 5.6], dtype=np.float32)
    i_f32_2d_np = np.array([[-1.2, -2.4], [4.0, -5.6]], dtype=np.float32)
    i_f16_1d_np = np.asarray([-1.2, 2.4, 4.0, 5.6], dtype=np.float16)
    i_i8_1d_np = np.asarray([-1, 2, 4, 5], dtype=np.int8)

    o = jit_zeros_like(i_f32_1d, i_f32_2d, i_f16_1d, i_i8_1d)
    e = zeros_like_np(i_f32_1d_np, i_f32_2d_np, i_f16_1d_np, i_i8_1d_np)

    np.testing.assert_allclose(o[0], np.asarray(e[0]), rtol=1e-04, atol=1e-06)
    np.testing.assert_allclose(o[1], np.asarray(e[1]), rtol=1e-04, atol=1e-06)
    np.testing.assert_allclose(o[2], np.asarray(e[2]), rtol=1e-04, atol=1e-06)
    np.testing.assert_array_equal(o[3], np.asarray(e[3]))


if __name__ == "__main__":
    pytest.main(["-v", __file__])
