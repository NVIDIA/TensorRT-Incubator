# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def jit_all(
    i_i32_1d_true,
    i_i32_1d_false,
    i_i32_2d_axis_0,
    i_i32_2d_axis_1,
    i_f32_1d_true,
    i_f32_1d_false,
    i_f32_2d_axis_0,
    i_f32_2d_axis_1,
    i_f16_1d_true,
    i_f16_1d_false,
    i_f16_2d_axis_0,
    i_f16_2d_axis_1,
    b_1d,
    b_2d_axis_0,
):
    o_i_i32_1d_true = jnp.all(i_i32_1d_true)
    o_i_i32_1d_false = jnp.all(i_i32_1d_false)
    o_i_i32_2d_axis_0 = jnp.all(i_i32_2d_axis_0, axis=0)
    o_i_i32_2d_axis_1 = jnp.all(i_i32_2d_axis_1, axis=1)
    o_i_f32_1d_true = jnp.all(i_f32_1d_true)
    o_i_f32_1d_false = jnp.all(i_f32_1d_false)
    o_i_f32_2d_axis_0 = jnp.all(i_f32_2d_axis_0, axis=0)
    o_i_f32_2d_axis_1 = jnp.all(i_f32_2d_axis_1, axis=1)
    o_i_f16_1d_true = jnp.all(i_f16_1d_true)
    o_i_f16_1d_false = jnp.all(i_f16_1d_false)
    o_i_f16_2d_axis_0 = jnp.all(i_f16_2d_axis_0, axis=0)
    o_i_f16_2d_axis_1 = jnp.all(i_f16_2d_axis_1, axis=1)
    o_b_1d = jnp.all(b_1d)
    o_b_2d_axis_0 = jnp.all(b_2d_axis_0, axis=0)
    return (
        o_i_i32_1d_true,
        o_i_i32_1d_false,
        o_i_i32_2d_axis_0,
        o_i_i32_2d_axis_1,
        o_i_f32_1d_true,
        o_i_f32_1d_false,
        o_i_f32_2d_axis_0,
        o_i_f32_2d_axis_1,
        o_i_f16_1d_true,
        o_i_f16_1d_false,
        o_i_f16_2d_axis_0,
        o_i_f16_2d_axis_1,
        o_b_1d,
        o_b_2d_axis_0,
    )


def test_all():
    """Test jnp.all with various dtypes and axes."""
    # 1d int32 numbers true and false
    i_i32_1d_true = np.asarray([1, 2, 3, 4])
    i_i32_1d_false = np.asarray([1, 0, 2, 4])

    # 2d int32 numbers with axis
    i_i32_2d_axis_0 = np.asarray([[1, 2], [3, 0]])
    i_i32_2d_axis_1 = np.asarray([[1, 2], [0, 3]])

    # 1d fp32 numbers true and false
    i_f32_1d_true = np.asarray([1, 2, 3, 4], dtype=jnp.float32)
    i_f32_1d_false = np.asarray([1, 0, 2, 4], dtype=jnp.float32)

    # 2d fp32 numbers with axis
    i_f32_2d_axis_0 = np.asarray([[1, 2], [3, 0]], dtype=jnp.float32)
    i_f32_2d_axis_1 = np.asarray([[1, 2], [0, 3]], dtype=jnp.float32)

    # 1d fp16 numbers true and false
    i_f16_1d_true = np.asarray([1, 2, 3, 4], dtype=jnp.float16)
    i_f16_1d_false = np.asarray([1, 0, 2, 4], dtype=jnp.float16)

    # 2d fp16 numbers with axis
    i_f16_2d_axis_0 = np.asarray([[1, 2], [3, 0]], dtype=jnp.float16)
    i_f16_2d_axis_1 = np.asarray([[1, 2], [0, 3]], dtype=jnp.float16)

    e_o_i_1d_true = np.asarray(1, dtype=bool)
    e_o_i_1d_false = np.asarray(0, dtype=bool)
    e_o_i_2d_axis_0 = np.asarray([1, 0], dtype=bool)
    e_o_i_2d_axis_1 = np.asarray([1, 0], dtype=bool)

    # 1d boolean and 2d boolean
    b_1d = np.asarray([True, True])
    e_o_b_1d = np.asarray(1, dtype=bool)
    b_2d_axis_0 = np.asarray([[True, False], [True, True]])
    e_o_b_2d_axis_0 = np.asarray([1, 0], dtype=bool)

    jnp_out = jit_all(
        i_i32_1d_true,
        i_i32_1d_false,
        i_i32_2d_axis_0,
        i_i32_2d_axis_1,
        i_f32_1d_true,
        i_f32_1d_false,
        i_f32_2d_axis_0,
        i_f32_2d_axis_1,
        i_f16_1d_true,
        i_f16_1d_false,
        i_f16_2d_axis_0,
        i_f16_2d_axis_1,
        b_1d,
        b_2d_axis_0,
    )
    expected = (
        e_o_i_1d_true,
        e_o_i_1d_false,
        e_o_i_2d_axis_0,
        e_o_i_2d_axis_1,
        e_o_i_1d_true,
        e_o_i_1d_false,
        e_o_i_2d_axis_0,
        e_o_i_2d_axis_1,
        e_o_i_1d_true,
        e_o_i_1d_false,
        e_o_i_2d_axis_0,
        e_o_i_2d_axis_1,
        e_o_b_1d,
        e_o_b_2d_axis_0,
    )
    for actual, expect in zip(jnp_out, expected):
        np.testing.assert_array_equal(actual, expect)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
