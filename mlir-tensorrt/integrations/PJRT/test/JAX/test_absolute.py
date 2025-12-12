# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def jit_absolute(i_f32_1d, i_f32_2d, i_f16_1d, i_f16_2d, i_i8_2d):
    o_f32_1d = jnp.absolute(i_f32_1d)
    o_f32_2d = jnp.absolute(i_f32_2d)
    o_f16_1d = jnp.absolute(i_f16_1d)
    o_f16_2d = jnp.absolute(i_f16_2d)
    o_i8_2d = jnp.absolute(i_i8_2d)
    return (o_f32_1d, o_f32_2d, o_f16_1d, o_f16_2d, o_i8_2d)


def test_absolute():
    """Test jnp.absolute with various dtypes.

    Compilation: jnp.numpy.absolute -> stablehlo.abs -> tensorrt.unary
    tensorrt.unary inputs needs to ND where N >= 1
    """
    i_f32_1d = np.asarray([-1.2, 2.4, 4.0, 5.6], dtype=np.float32)
    i_f32_2d = np.asarray([[-1.2, -2.4], [4.0, -5.6]], dtype=np.float32)
    i_f16_1d = np.asarray([-1.2, 2.4, 4.0, 5.6], dtype=np.float16)
    i_f16_2d = np.asarray([[-1.2, -2.4], [4.0, -5.6]], dtype=np.float16)
    i_i8_2d = np.asarray([[-1, 2], [3, -4]], dtype=np.int8)

    o = jit_absolute(i_f32_1d, i_f32_2d, i_f16_1d, i_f16_2d, i_i8_2d)

    e_f32_1d = np.asarray([1.2, 2.4, 4.0, 5.6], dtype=np.float32)
    e_f32_2d = np.asarray([[1.2, 2.4], [4.0, 5.6]], dtype=np.float32)
    np.testing.assert_allclose(o[0], e_f32_1d, rtol=1e-03, atol=1e-2)
    np.testing.assert_allclose(o[1], e_f32_2d, rtol=1e-03, atol=1e-2)

    e_f16_1d = np.asarray([1.2, 2.4, 4.0, 5.6], dtype=np.float16)
    e_f16_2d = np.asarray([[1.2, 2.4], [4.0, 5.6]], dtype=np.float16)
    np.testing.assert_allclose(o[2], e_f16_1d, rtol=1e-03, atol=1e-2)
    np.testing.assert_allclose(o[3], e_f16_2d, rtol=1e-03, atol=1e-2)

    e_i8_2d = np.asarray([[1, 2], [3, 4]], dtype=np.int8)
    np.testing.assert_array_equal(o[4], e_i8_2d)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
