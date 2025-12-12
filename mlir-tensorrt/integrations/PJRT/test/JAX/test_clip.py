# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import numpy as np
import jax.numpy as jnp
from jax import jit


@jit
def jit_clip(x, a_min, a_max):
    return jnp.clip(x, min=a_min, max=a_max)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_clip(dtype):
    """Test jnp.clip with various dtypes."""
    x = np.asarray([-1, 2, 3, 10]).astype(dtype)
    a_min = 1
    a_max = 6
    np_result = np.clip(x, min=a_min, max=a_max)
    jax_result = jit_clip(x, a_min, a_max)
    np.testing.assert_array_equal(np_result, jax_result)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
