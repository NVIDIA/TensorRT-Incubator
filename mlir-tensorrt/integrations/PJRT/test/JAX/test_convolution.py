# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


@jax.jit
def jit_conv(
    f32_2d_left,
    f32_2d_right,
    f32_2d_k3_right,
    f16_2d_left,
    f16_2d_right,
    f32_1d_left,
    f32_1d_right,
    f32_3d_left,
    f32_3d_right,
):
    # different data types
    # 2d conv, same padding, stride = 1, fp32
    o_f32_2d = lax.conv_general_dilated(f32_2d_left, f32_2d_right, [1, 1], "SAME")
    # 2d conv, same padding, stride = 1, fp16
    o_f16_2d = lax.conv_general_dilated(f16_2d_left, f16_2d_right, [1, 1], "SAME")

    # different strides
    # 2d conv, same padding, stride = 2, fp32
    o_f32_s3_2d = lax.conv_general_dilated(f32_2d_left, f32_2d_right, [2, 2], "SAME")

    # different paddings
    # 2d conv, kernel = 3, same padding, stride = 1, fp32
    o_f32_k3_2d = lax.conv_general_dilated(f32_2d_left, f32_2d_k3_right, [1, 1], "SAME")
    # 2d conv, kernel = 3, valid padding, stride = 1, fp32
    o_f32_k3_vp_2d = lax.conv_general_dilated(
        f32_2d_left, f32_2d_k3_right, [1, 1], "VALID"
    )

    # dilated convs
    o_f32_2d_dilated = lax.conv_general_dilated(
        f32_2d_left, f32_2d_k3_right, [1, 1], "SAME", rhs_dilation=[3, 3]
    )

    # 1d conv, same padding, stride = 1, fp32
    o_f32_1d = lax.conv_general_dilated(f32_1d_left, f32_1d_right, [1], "SAME")

    # 3d conv, same padding, stride = 1, fp32
    o_f32_3d = lax.conv_general_dilated(f32_3d_left, f32_3d_right, [1, 1, 1], "SAME")
    return (
        o_f32_2d,
        o_f16_2d,
        o_f32_s3_2d,
        o_f32_k3_2d,
        o_f32_k3_vp_2d,
        o_f32_2d_dilated,
        o_f32_1d,
        o_f32_3d,
    )


def test_convolution():
    """Test various convolution operations with different dtypes, strides, and paddings."""
    f32_2d_left = np.ones([1, 2, 3, 3], dtype=np.float32)
    f32_2d_right = np.ones([3, 2, 1, 1], dtype=np.float32)
    f32_2d_k3_right = np.ones([3, 2, 3, 3], dtype=np.float32)

    f16_2d_left = np.ones([1, 2, 3, 3], dtype=np.float16)
    f16_2d_right = np.ones([3, 2, 1, 1], dtype=np.float16)

    f32_1d_left = np.ones([1, 2, 3], dtype=np.float32)
    f32_1d_right = np.ones([3, 2, 1], dtype=np.float32)

    f32_3d_left = np.ones([1, 2, 3, 3, 3], dtype=np.float32)
    f32_3d_right = np.ones([3, 2, 1, 1, 1], dtype=np.float32)

    o = jit_conv(
        f32_2d_left,
        f32_2d_right,
        f32_2d_k3_right,
        f16_2d_left,
        f16_2d_right,
        f32_1d_left,
        f32_1d_right,
        f32_3d_left,
        f32_3d_right,
    )
    e_f32_2d = 2 * np.ones([1, 3, 3, 3], dtype=np.float32)
    e_f16_2d = 2 * np.ones([1, 3, 3, 3], dtype=np.float16)
    e_f32_s3_2d = 2 * np.ones([1, 3, 2, 2], dtype=np.float32)
    e_f32_k3_2d = np.asarray(
        [8, 12, 8, 12, 18, 12, 8, 12, 8] * 3, dtype=np.float32
    ).reshape([1, 3, 3, 3])
    e_f32_k3_vp_2d = 18 * np.ones([1, 3, 1, 1], dtype=np.float32)
    e_f32_2d_dilated = 2 * np.ones([1, 3, 3, 3], dtype=np.float32)
    e_f32_1d = 2 * np.ones([1, 3, 3], dtype=np.float32)
    e_f32_3d = 2 * np.ones([1, 3, 3, 3, 3], dtype=np.float32)

    np.testing.assert_allclose(o[0], np.asarray(e_f32_2d), rtol=1e-04)
    np.testing.assert_allclose(o[1], np.asarray(e_f16_2d), rtol=1e-04)
    np.testing.assert_allclose(o[2], np.asarray(e_f32_s3_2d), rtol=1e-04)
    np.testing.assert_allclose(o[3], np.asarray(e_f32_k3_2d), rtol=1e-04)
    np.testing.assert_allclose(o[4], np.asarray(e_f32_k3_vp_2d), rtol=1e-04)
    np.testing.assert_allclose(o[5], np.asarray(e_f32_2d_dilated), rtol=1e-04)
    np.testing.assert_allclose(o[6], np.asarray(e_f32_1d), rtol=1e-04)
    np.testing.assert_allclose(o[7], np.asarray(e_f32_3d), rtol=1e-04)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
