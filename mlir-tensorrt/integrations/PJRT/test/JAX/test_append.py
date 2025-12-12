# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def jit_append(
    i_1d_1d_i32,
    i_1d_1d_f32,
    i_1d_1d_f16,
    i_1d_2d_i32,
    i_1d_2d_f32,
    i_1d_2d_f16,
    i_2d_2d_a0_i32,
    i_2d_2d_a0_f32,
    i_2d_2d_a0_f16,
    i_2d_2d_a1_i32,
    i_2d_2d_a1_f32,
    i_2d_2d_a1_f16,
):
    o_1d_1d_i32 = jnp.append(i_1d_1d_i32[0], i_1d_1d_i32[1])
    o_1d_1d_f32 = jnp.append(i_1d_1d_f32[0], i_1d_1d_f32[1])
    o_1d_1d_f16 = jnp.append(i_1d_1d_f16[0], i_1d_1d_f16[1])

    o_1d_2d_i32 = jnp.append(i_1d_2d_i32[0], i_1d_2d_i32[1])
    o_1d_2d_f32 = jnp.append(i_1d_2d_f32[0], i_1d_2d_f32[1])
    o_1d_2d_f16 = jnp.append(i_1d_2d_f16[0], i_1d_2d_f16[1])

    o_2d_2d_a0_i32 = jnp.append(i_2d_2d_a0_i32[0], i_2d_2d_a0_i32[1], axis=0)
    o_2d_2d_a0_f32 = jnp.append(i_2d_2d_a0_f32[0], i_2d_2d_a0_f32[1], axis=0)
    o_2d_2d_a0_f16 = jnp.append(i_2d_2d_a0_f16[0], i_2d_2d_a0_f16[1], axis=0)

    o_2d_2d_a1_i32 = jnp.append(i_2d_2d_a1_i32[0], i_2d_2d_a1_i32[1], axis=1)
    o_2d_2d_a1_f32 = jnp.append(i_2d_2d_a1_f32[0], i_2d_2d_a1_f32[1], axis=1)
    o_2d_2d_a1_f16 = jnp.append(i_2d_2d_a1_f16[0], i_2d_2d_a1_f16[1], axis=1)

    return (
        o_1d_1d_i32,
        o_1d_1d_f32,
        o_1d_1d_f16,
        o_1d_2d_i32,
        o_1d_2d_f32,
        o_1d_2d_f16,
        o_2d_2d_a0_i32,
        o_2d_2d_a0_f32,
        o_2d_2d_a0_f16,
        o_2d_2d_a1_i32,
        o_2d_2d_a1_f32,
        o_2d_2d_a1_f16,
    )


def test_append():
    """Test jnp.append with various dtypes and shapes."""
    # append 1d with 1d
    i_1d_1d_i32_0 = np.asarray([1, 2, 3], dtype=np.int32)
    i_1d_1d_i32_1 = np.asarray([100, 20, 40], dtype=np.int32)
    e_1d_1d_i32 = np.asarray([1, 2, 3, 100, 20, 40], dtype=np.int32)

    i_1d_1d_f32_0 = np.asarray([1, 2, 3], dtype=np.float32)
    i_1d_1d_f32_1 = np.asarray([100, 20, 40], dtype=np.float32)
    e_1d_1d_f32 = np.asarray([1, 2, 3, 100, 20, 40], dtype=np.float32)

    i_1d_1d_f16_0 = np.asarray([1, 2, 3], dtype=np.float16)
    i_1d_1d_f16_1 = np.asarray([100, 20, 40], dtype=np.float16)
    e_1d_1d_f16 = np.asarray([1, 2, 3, 100, 20, 40], dtype=np.float16)

    # append 1d with 2d
    i_1d_2d_i32_0 = np.asarray([10, 20, 30], dtype=np.int32)
    i_1d_2d_i32_1 = np.asarray([[1, 2], [3, 4]], dtype=np.int32)
    e_1d_2d_i32 = np.asarray([10, 20, 30, 1, 2, 3, 4], dtype=np.int32)

    i_1d_2d_f32_0 = np.asarray([10, 20, 30], dtype=np.float32)
    i_1d_2d_f32_1 = np.asarray([[1, 2], [3, 4]], dtype=np.float32)
    e_1d_2d_f32 = np.asarray([10, 20, 30, 1, 2, 3, 4], dtype=np.float32)

    i_1d_2d_f16_0 = np.asarray([10, 20, 30], dtype=np.float16)
    i_1d_2d_f16_1 = np.asarray([[1, 2], [3, 4]], dtype=np.float16)
    e_1d_2d_f16 = np.asarray([10, 20, 30, 1, 2, 3, 4], dtype=np.float16)

    # append 2d with 2d along axis=0
    i_2d_2d_a0_i32_0 = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    i_2d_2d_a0_i32_1 = np.asarray([[7, 8, 9]], dtype=np.int32)
    e_2d_2d_a0_i32 = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)

    i_2d_2d_a0_f32_0 = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    i_2d_2d_a0_f32_1 = np.asarray([[7, 8, 9]], dtype=np.float32)
    e_2d_2d_a0_f32 = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)

    i_2d_2d_a0_f16_0 = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float16)
    i_2d_2d_a0_f16_1 = np.asarray([[7, 8, 9]], dtype=np.float16)
    e_2d_2d_a0_f16 = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float16)

    # append 2d with 2d along axis=1
    i_2d_2d_a1_i32_0 = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    i_2d_2d_a1_i32_1 = np.asarray([[100], [200]], dtype=np.int32)
    e_2d_2d_a1_i32 = np.asarray([[1, 2, 3, 100], [4, 5, 6, 200]], dtype=np.int32)

    i_2d_2d_a1_f32_0 = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    i_2d_2d_a1_f32_1 = np.asarray([[100], [200]], dtype=np.float32)
    e_2d_2d_a1_f32 = np.asarray([[1, 2, 3, 100], [4, 5, 6, 200]], dtype=np.float32)

    i_2d_2d_a1_f16_0 = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float16)
    i_2d_2d_a1_f16_1 = np.asarray([[100], [200]], dtype=np.float16)
    e_2d_2d_a1_f16 = np.asarray([[1, 2, 3, 100], [4, 5, 6, 200]], dtype=np.float16)

    o = jit_append(
        (i_1d_1d_i32_0, i_1d_1d_i32_1),
        (i_1d_1d_f32_0, i_1d_1d_f32_1),
        (i_1d_1d_f16_0, i_1d_1d_f16_1),
        (i_1d_2d_i32_0, i_1d_2d_i32_1),
        (i_1d_2d_f32_0, i_1d_2d_f32_1),
        (i_1d_2d_f16_0, i_1d_2d_f16_1),
        (i_2d_2d_a0_i32_0, i_2d_2d_a0_i32_1),
        (i_2d_2d_a0_f32_0, i_2d_2d_a0_f32_1),
        (i_2d_2d_a0_f16_0, i_2d_2d_a0_f16_1),
        (i_2d_2d_a1_i32_0, i_2d_2d_a1_i32_1),
        (i_2d_2d_a1_f32_0, i_2d_2d_a1_f32_1),
        (i_2d_2d_a1_f16_0, i_2d_2d_a1_f16_1),
    )
    np.testing.assert_array_equal(o[0], e_1d_1d_i32)
    np.testing.assert_array_equal(o[1], e_1d_1d_f32)
    np.testing.assert_array_equal(o[2], e_1d_1d_f16)
    np.testing.assert_array_equal(o[3], e_1d_2d_i32)
    np.testing.assert_array_equal(o[4], e_1d_2d_f32)
    np.testing.assert_array_equal(o[5], e_1d_2d_f16)
    np.testing.assert_array_equal(o[6], e_2d_2d_a0_i32)
    np.testing.assert_array_equal(o[7], e_2d_2d_a0_f32)
    np.testing.assert_array_equal(o[8], e_2d_2d_a0_f16)
    np.testing.assert_array_equal(o[9], e_2d_2d_a1_i32)
    np.testing.assert_array_equal(o[10], e_2d_2d_a1_f32)
    np.testing.assert_array_equal(o[11], e_2d_2d_a1_f16)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
