# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def jit_any(
    i_1d_i32_true,
    i_1d_i32_false,
    i_2d_i32_axis_0,
    i_2d_i32_axis_1,
    i_1d_f32_true,
    i_1d_f32_false,
    i_2d_f32_axis_0,
    i_2d_f32_axis_1,
    i_1d_f16_true,
    i_1d_f16_false,
    i_2d_f16_axis_0,
    i_2d_f16_axis_1,
    b_1d_true,
    b_2d_axis_0,
):
    o_1d_i32_true = jnp.any(i_1d_i32_true)
    o_1d_i32_false = jnp.any(i_1d_i32_false)
    o_2d_i32_axis_0 = jnp.any(i_2d_i32_axis_0, axis=0)
    o_2d_i32_axis_1 = jnp.any(i_2d_i32_axis_1, axis=1)

    o_1d_f32_true = jnp.any(i_1d_f32_true)
    o_1d_f32_false = jnp.any(i_1d_f32_false)
    o_2d_f32_axis_0 = jnp.any(i_2d_f32_axis_0, axis=0)
    o_2d_f32_axis_1 = jnp.any(i_2d_f32_axis_1, axis=1)

    o_1d_f16_true = jnp.any(i_1d_f16_true)
    o_1d_f16_false = jnp.any(i_1d_f16_false)
    o_2d_f16_axis_0 = jnp.any(i_2d_f16_axis_0, axis=0)
    o_2d_f16_axis_1 = jnp.any(i_2d_f16_axis_1, axis=1)

    o_1d_b_true = jnp.any(b_1d_true)
    o_2d_b_axis_0 = jnp.any(b_2d_axis_0, axis=0)
    return (
        o_1d_i32_true,
        o_1d_i32_false,
        o_2d_i32_axis_0,
        o_2d_i32_axis_1,
        o_1d_f32_true,
        o_1d_f32_false,
        o_2d_f32_axis_0,
        o_2d_f32_axis_1,
        o_1d_f16_true,
        o_1d_f16_false,
        o_2d_f16_axis_0,
        o_2d_f16_axis_1,
        o_1d_b_true,
        o_2d_b_axis_0,
    )


def test_any():
    """Test jnp.any with various dtypes and axes."""
    # 1d i32 numbers true and false
    i_1d_i32_true = np.asarray([0, 0, 3, 0])
    i_1d_i32_false = np.asarray([0, 0, 0, 0])

    # 2d i32 numbers with axis
    i_2d_i32_axis_0 = np.asarray([[1, 0], [3, 0]])
    i_2d_i32_axis_1 = np.asarray([[0, 0], [2, 0]])

    # 1d f32 numbers true and false
    i_1d_f32_true = np.asarray([0, 0, 3, 0], dtype=jnp.float32)
    i_1d_f32_false = np.asarray([0, 0, 0, 0], dtype=jnp.float32)

    # 2d f32 numbers with axis
    i_2d_f32_axis_0 = np.asarray([[1, 0], [3, 0]], dtype=jnp.float32)
    i_2d_f32_axis_1 = np.asarray([[0, 0], [2, 0]], dtype=jnp.float32)

    # 1d f16 numbers true and false
    i_1d_f16_true = np.asarray([0, 0, 3, 0], dtype=jnp.float16)
    i_1d_f16_false = np.asarray([0, 0, 0, 0], dtype=jnp.float16)

    # 2d f16 numbers with axis
    i_2d_f16_axis_0 = np.asarray([[1, 0], [3, 0]], dtype=jnp.float16)
    i_2d_f16_axis_1 = np.asarray([[0, 0], [2, 0]], dtype=jnp.float16)

    e_1d_true = np.asarray(1, dtype=bool)
    e_1d_false = np.asarray(0, dtype=bool)
    e_o_2d_axis_0 = np.asarray([1, 0], dtype=bool)
    e_o_2d_axis_1 = np.asarray([0, 1], dtype=bool)

    # 1d boolean and 2d boolean
    b_1d_true = np.asarray([True, True])
    b_2d_axis_0 = np.asarray([[False, False], [False, False]])

    e_b_2d_axis_0 = np.asarray([0, 0], dtype=bool)

    o = jit_any(
        i_1d_i32_true,
        i_1d_i32_false,
        i_2d_i32_axis_0,
        i_2d_i32_axis_1,
        i_1d_f32_true,
        i_1d_f32_false,
        i_2d_f32_axis_0,
        i_2d_f32_axis_1,
        i_1d_f16_true,
        i_1d_f16_false,
        i_2d_f16_axis_0,
        i_2d_f16_axis_1,
        b_1d_true,
        b_2d_axis_0,
    )
    expected = (
        e_1d_true,
        e_1d_false,
        e_o_2d_axis_0,
        e_o_2d_axis_1,
        e_1d_true,
        e_1d_false,
        e_o_2d_axis_0,
        e_o_2d_axis_1,
        e_1d_true,
        e_1d_false,
        e_o_2d_axis_0,
        e_o_2d_axis_1,
        e_1d_true,
        e_b_2d_axis_0,
    )
    for actual, expect in zip(o, expected):
        np.testing.assert_array_equal(actual, expect)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
