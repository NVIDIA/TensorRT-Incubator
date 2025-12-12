# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def jit_add(x, y):
    return x + y


@pytest.mark.mlir_trt_flags(
    "-debug -debug-only=translate-to-tensorrt -tensorrt-workspace-memory-pool-limit=1gb"
)
def test_tensorrt_workspace():
    """Test that basic operations work with workspace memory pool limit set."""
    x = np.array([-1.2, 2.4, 4.0, 5.6], dtype=np.float32)
    y = np.array([1.2, 3.4, 0.8, -2.2], dtype=np.float32)
    expected = x + y
    result = jit_add(x, y)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
