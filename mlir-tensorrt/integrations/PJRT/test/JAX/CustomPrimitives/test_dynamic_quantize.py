# REQUIRES: all-gpus-support-fp4
# REQUIRES: has-support-for-ptx-gte-86
# RUN: %pick-one-gpu %mlir-trt-jax-py %s

"""Test for MLIR-TensorRT JAX dynamic quantize primitive.

NOTE: This test file contains extra setup code for CI testing without pip installation.
      With a properly installed mlir_tensorrt_jax package, users only need:

      >>> import mlir_tensorrt_jax as mtrt
      >>> output, scale = mtrt.mtrt_dynamic_quantize(input, ...)
"""

# ====================================================================================
# CI TEST HARNESS - Not needed for regular users with pip-installed package
# ====================================================================================

# Step 1: Add source directory to path (pip installation handles this automatically)
from pathlib import Path
import sys

mtrt_ops_path = (
    Path(__file__).parent.parent.parent.parent / "python" / "mlir_tensorrt_jax"
)
assert mtrt_ops_path.exists() and mtrt_ops_path.is_dir()
sys.path.append(str(mtrt_ops_path))

# Step 2: Force JAX plugin discovery (happens automatically with pip installation)
from jax._src.xla_bridge import _discover_and_register_pjrt_plugins

_discover_and_register_pjrt_plugins()

# Step 3: Manual imports and registration (automatic when installed via pip)
from mtrt_ops.dynamic_quantize import (
    mtrt_dynamic_quantize,
    register_dynamic_quantize_lowering,
)
from mtrt_ops.dequantize import (
    mtrt_dequantize,
    register_dequantize_lowering,
)

register_dynamic_quantize_lowering()
register_dequantize_lowering()

# ====================================================================================
# ACTUAL TEST CODE - This is what regular users write
# ====================================================================================

import jax
import numpy as np
import pytest


@jax.jit
def jit_dynamic_quantize_pt(x):
    """Per-tensor dynamic quantization test."""
    double_quant_scale = np.array(1.0)
    fp4_out, fp8_scale = mtrt_dynamic_quantize(x, double_quant_scale, axis=1)
    dequantized_scales = mtrt_dequantize(
        fp8_scale, double_quant_scale, mode="tensorrt.pt_dq", output_dtype=np.float32
    )
    dequantized_out = mtrt_dequantize(
        fp4_out, dequantized_scales, mode="tensorrt.block_dq", output_dtype=np.float32
    )
    return dequantized_out


@pytest.mark.requires_fp4
@pytest.mark.requires_minimum_ptx_version(version=86)
def test_dynamic_quantize_pt():
    """Test per-tensor dynamic quantization."""
    x = np.array(
        [
            [
                0.0,
                0.3,
                0.6,
                1.0,
                1.3,
                1.6,
                1.9,
                2.3,
                2.6,
                2.9,
                3.2,
                3.5,
                3.9,
                4.2,
                4.5,
                4.8,
                5.2,
                5.5,
                5.8,
                6.1,
                6.5,
                6.8,
                7.1,
                7.4,
                7.7,
                8.1,
                8.4,
                8.7,
                9.0,
                9.4,
                9.7,
                10.0,
            ],
            [
                3.0,
                3.3,
                3.6,
                4.0,
                3.3,
                3.6,
                3.9,
                3.3,
                2.6,
                2.9,
                3.2,
                3.5,
                3.9,
                4.2,
                4.5,
                4.8,
                -5.2,
                -5.5,
                -5.8,
                -6.1,
                -5.5,
                -5.8,
                5.1,
                5.4,
                5.7,
                5.1,
                5.4,
                5.7,
                6.0,
                6.4,
                4.7,
                6.0,
            ],
        ],
        dtype=np.float32,
    )
    expected = np.array(
        [
            [
                0.0,
                0.40625,
                0.40625,
                0.8125,
                1.21875,
                1.625,
                1.625,
                2.4375,
                2.4375,
                3.25,
                3.25,
                3.25,
                3.25,
                4.875,
                4.875,
                4.875,
                4.875,
                4.875,
                6.5,
                6.5,
                6.5,
                6.5,
                6.5,
                6.5,
                6.5,
                6.5,
                9.75,
                9.75,
                9.75,
                9.75,
                9.75,
                9.75,
            ],
            [
                3.25,
                3.25,
                3.25,
                3.25,
                3.25,
                3.25,
                3.25,
                3.25,
                2.4375,
                3.25,
                3.25,
                3.25,
                3.25,
                4.875,
                4.875,
                4.875,
                -4.5,
                -4.5,
                -6.75,
                -6.75,
                -4.5,
                -6.75,
                4.5,
                4.5,
                6.75,
                4.5,
                4.5,
                6.75,
                6.75,
                6.75,
                4.5,
                6.75,
            ],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        jit_dynamic_quantize_pt(x), expected, rtol=1e-4, atol=1e-6
    )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
