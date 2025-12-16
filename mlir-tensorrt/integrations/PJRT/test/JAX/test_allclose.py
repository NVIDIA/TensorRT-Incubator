# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def jit_allclose(
    a_true_f32,
    b_true_f32,
    a_false_f32,
    b_false_f32,
    a_true_f16,
    b_true_f16,
    a_false_f16,
    b_false_f16,
):
    o_true_f32 = jnp.allclose(a_true_f32, b_true_f32)
    o_false_f32 = jnp.allclose(a_false_f32, b_false_f32)
    o_true_f16 = jnp.allclose(a_true_f16, b_true_f16)
    o_false_f16 = jnp.allclose(a_false_f16, b_false_f16)
    return (o_true_f32, o_false_f32, o_true_f16, o_false_f16)


def test_allclose():
    """Test jnp.allclose with various precision levels."""
    # unlike Numpy, default float type is float32
    a_true_f32 = np.asarray([1e10, 1e-8])
    b_true_f32 = np.asarray([1e10, 1e-8])

    a_false_f32 = np.asarray([1e10, 1e-7])
    b_false_f32 = np.asarray([1.00001e10, 1e-8])

    a_true_f16 = np.asarray([3.456789, 4.6734], dtype=jnp.float16)
    b_true_f16 = np.asarray([3.456788, 4.6734], dtype=jnp.float16)

    a_false_f16 = np.asarray([3.456789, 4.6734], dtype=jnp.float16)
    b_false_f16 = np.asarray([3.491788, 4.6734], dtype=jnp.float16)

    e_true = np.asarray(1, dtype=bool)
    e_false = np.asarray(0, dtype=bool)
    o = jit_allclose(
        a_true_f32,
        b_true_f32,
        a_false_f32,
        b_false_f32,
        a_true_f16,
        b_true_f16,
        a_false_f16,
        b_false_f16,
    )
    np.testing.assert_array_equal(o[0], e_true)
    np.testing.assert_array_equal(o[1], e_false)
    np.testing.assert_array_equal(o[2], e_true)
    np.testing.assert_array_equal(o[3], e_false)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
