# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import jax
import numpy as np
from jax import lax


@jax.jit
def jit_deconv(
    f32_2d_left,
    f32_2d_right,
    f32_2d_k3_right,
    f32_group_right,
    f16_2d_left,
    f16_2d_right,
    f32_1d_left,
    f32_1d_right,
    f32_3d_left,
    f32_3d_right,
    f32_2d_left_unit_lhs_dilation,
    f32_2d_right_unit_lhs_dilation,
):
    # different data types
    # 2d deconv, no padding, kernel=2, stride = 1, lhs_dilation=[1,2], fp32
    o_f32_2d = lax.conv_general_dilated(
        f32_2d_left,
        f32_2d_right,
        window_strides=[1, 1],
        padding=[[0, 0], [0, 0]],
        lhs_dilation=[1, 2],
    )

    # 2d deconv, no padding, kernel=2, stride = 1, lhs_dilation=[2,1], fp16
    o_f16_2d = lax.conv_general_dilated(
        f16_2d_left,
        f16_2d_right,
        window_strides=[1, 1],
        padding=[[0, 0], [0, 0]],
        lhs_dilation=[2, 1],
    )

    # 2d grouped deconv, no padding, kernel=1, stride = 1, lhs_dilation=[1,2], fp32
    o_f32_2d_grouped = lax.conv_general_dilated(
        f32_2d_left,
        f32_group_right,
        window_strides=[1, 1],
        padding=[[0, 0], [0, 0]],
        lhs_dilation=[1, 2],
        feature_group_count=2,
    )

    # different paddings
    # 2d deconv, kernel = 3, 1 padding, stride = 1, lhs_dilation=[2, 2], fp32
    o_f32_k3_2d = lax.conv_general_dilated(
        f32_2d_left,
        f32_2d_k3_right,
        window_strides=[1, 1],
        padding=[[1, 1], [1, 1]],
        lhs_dilation=[2, 2],
    )
    # 2d deconv, kernel = 3, 2 padding, stride = 1, lhs_dilation=[2, 2], fp32
    o_f32_k3_vp_2d = lax.conv_general_dilated(
        f32_2d_left,
        f32_2d_k3_right,
        window_strides=[1, 1],
        padding=[[2, 2], [2, 2]],
        lhs_dilation=[2, 2],
    )

    # dilated deconvs
    o_f32_2d_dilated = lax.conv_general_dilated(
        f32_2d_left,
        f32_2d_k3_right,
        window_strides=[1, 1],
        padding=[[2, 2], [2, 2]],
        rhs_dilation=[2, 2],
        lhs_dilation=[1, 2],
    )

    # 1d deconv, padding = 2, stride = 1, fp32
    o_f32_1d = lax.conv_general_dilated(
        f32_1d_left,
        f32_1d_right,
        window_strides=[1],
        padding=[[2, 2]],
        lhs_dilation=[2],
    )

    # 3d deconv, padding = 0, stride = 1, fp32
    o_f32_3d = lax.conv_general_dilated(
        f32_3d_left,
        f32_3d_right,
        window_strides=[1, 1, 1],
        padding=[[0, 0], [0, 0], [0, 0]],
        lhs_dilation=[1, 2, 2],
    )

    # Unit lhs dilation but represents transpose convolution
    o_f32_unit_lhs_dilation_case_1 = lax.conv_general_dilated(
        f32_2d_left_unit_lhs_dilation,
        f32_2d_right_unit_lhs_dilation,
        window_strides=[1, 1],
        padding=[[-1, -1], [-1, -1]],
        lhs_dilation=[1, 1],
    )

    o_f32_unit_lhs_dilation_case_2 = lax.conv_general_dilated(
        f32_2d_left_unit_lhs_dilation,
        f32_2d_right_unit_lhs_dilation,
        window_strides=[1, 1],
        padding=[[-1, -1], [-1, -1]],
        lhs_dilation=[1, 1],
        rhs_dilation=[1, 2],
    )

    return (
        o_f32_2d,
        o_f16_2d,
        o_f32_2d_grouped,
        o_f32_k3_2d,
        o_f32_k3_vp_2d,
        o_f32_2d_dilated,
        o_f32_1d,
        o_f32_3d,
        o_f32_unit_lhs_dilation_case_1,
        o_f32_unit_lhs_dilation_case_2,
    )


def test_deconvolution():
    """Test various deconvolution operations with different dtypes, strides, and paddings."""
    f32_2d_left = np.random.uniform(0, 5, (1, 2, 3, 3))
    f32_2d_right = np.random.uniform(0, 5, (1, 2, 2, 2))

    f32_2d_k3_right = np.random.uniform(0, 5, (1, 2, 3, 3))
    f32_group_right = np.random.uniform(0, 5, (2, 1, 1, 1))

    f16_2d_left = np.random.uniform(0, 5, (1, 2, 2, 3)).astype(np.float16)
    f16_2d_right = np.random.uniform(0, 5, (3, 2, 1, 1)).astype(np.float16)

    f32_1d_left = np.random.uniform(0, 5, (1, 2, 3))
    f32_1d_right = np.random.uniform(0, 5, (3, 2, 3))

    f32_3d_left = np.random.uniform(0, 5, (1, 2, 3, 3, 3))
    f32_3d_right = np.random.uniform(0, 5, (3, 2, 1, 1, 1))

    f32_2d_left_unit_lhs_dilation = np.random.uniform(0, 5, (1, 2, 5, 5))
    f32_2d_right_unit_lhs_dilation = np.random.uniform(0, 5, (1, 2, 2, 2))

    o = jit_deconv(
        f32_2d_left,
        f32_2d_right,
        f32_2d_k3_right,
        f32_group_right,
        f16_2d_left,
        f16_2d_right,
        f32_1d_left,
        f32_1d_right,
        f32_3d_left,
        f32_3d_right,
        f32_2d_left_unit_lhs_dilation,
        f32_2d_right_unit_lhs_dilation,
    )

    with jax.default_device(jax.devices("cpu")[0]):
        expected = jit_deconv(
            f32_2d_left,
            f32_2d_right,
            f32_2d_k3_right,
            f32_group_right,
            f16_2d_left,
            f16_2d_right,
            f32_1d_left,
            f32_1d_right,
            f32_3d_left,
            f32_3d_right,
            f32_2d_left_unit_lhs_dilation,
            f32_2d_right_unit_lhs_dilation,
        )

    np.testing.assert_allclose(o[0], expected[0], rtol=1e-02, atol=1e-04)
    np.testing.assert_allclose(o[1], expected[1], rtol=1e-02, atol=1e-04)
    np.testing.assert_allclose(o[2], expected[2], rtol=1e-02, atol=1e-04)
    np.testing.assert_allclose(o[3], expected[3], rtol=1e-02, atol=1e-04)
    np.testing.assert_allclose(o[4], expected[4], rtol=1e-02, atol=1e-04)
    np.testing.assert_allclose(o[5], expected[5], rtol=1e-02, atol=1e-04)
    np.testing.assert_allclose(o[6], expected[6], rtol=1e-02, atol=1e-04)
    np.testing.assert_allclose(o[7], expected[7], rtol=1e-02, atol=1e-04)
    np.testing.assert_allclose(o[8], expected[8], rtol=1e-02, atol=1e-04)
    np.testing.assert_allclose(o[9], expected[9], rtol=1e-02, atol=1e-04)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
