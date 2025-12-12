# REQUIRES: all-gpus-support-fp8
# REQUIRES: tensorrt-version-ge-10.12
# RUN: %pick-one-gpu %mlir-trt-jax-py %s

"""Test for MLIR-TensorRT JAX dequantize primitive.

NOTE: This test file contains extra setup code for CI testing without pip installation.
      With a properly installed mlir_tensorrt_jax package, users only need:

      >>> import mlir_tensorrt_jax as mtrt
      >>> output = mtrt.mtrt_dequantize(input, scale, ...)
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
from mtrt_ops.dequantize import mtrt_dequantize, register_dequantize_lowering

register_dequantize_lowering()

# ====================================================================================
# ACTUAL TEST CODE - This is what regular users write
# ====================================================================================

import jax
import numpy as np
import pytest
from ml_dtypes import float8_e4m3fn


@jax.jit
def jit_dequantize_pt(x):
    scale = np.array(0.5)
    return mtrt_dequantize(x, scale, mode="tensorrt.pt_dq", output_dtype=np.float32)


@jax.jit
def jit_dequantize_pc(x):
    scale = np.array([0.5, 0.1])
    return mtrt_dequantize(
        x, scale, mode="tensorrt.pc_dq", axis=0, output_dtype=np.float32
    )


@jax.jit
def jit_dequantize_block(x):
    scale = np.array([[2.0, 2.0, 2.0, 2.0]])
    return mtrt_dequantize(x, scale, mode="tensorrt.block_dq", output_dtype=np.float32)


def test_dequantize_pt():
    """Test per-tensor dequantization."""
    x_int8 = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.int8)
    expected = np.array(
        [[5.0, 10.0, 15.0, 20.0], [25.0, 30.0, 35.0, 40.0]], dtype=np.float32
    )
    np.testing.assert_allclose(
        jit_dequantize_pt(x_int8), expected, atol=1e-5, rtol=1e-7
    )


def test_dequantize_pc():
    """Test per-channel dequantization."""
    x_int8 = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.int8)
    expected = np.array(
        [[5.0, 10.0, 15.0, 20.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32
    )
    np.testing.assert_allclose(
        jit_dequantize_pc(x_int8), expected, atol=1e-5, rtol=1e-7
    )


@pytest.mark.requires_fp8
@pytest.mark.requires_trt_version("ge", major=10, minor=12)
def test_dequantize_block():
    """Test block-wise dequantization with float8."""
    x_fp8 = np.array(
        [[5.0, 10.0, 15.0, 20.0], [25.0, 30.0, 35.0, 40.0]], dtype=float8_e4m3fn
    )
    expected = np.array(
        [[10.0, 20.0, 30.0, 40.0], [48.0, 60.0, 72.0, 80.0]], dtype=np.float32
    )
    np.testing.assert_allclose(
        jit_dequantize_block(x_fp8), expected, atol=1e-5, rtol=1e-7
    )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
