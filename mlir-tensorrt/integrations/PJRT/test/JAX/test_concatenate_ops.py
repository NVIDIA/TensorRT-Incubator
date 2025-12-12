# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import numpy as np
import jax.numpy as jnp
from jax import jit


@jit
def jit_column_stack(x):
    return jnp.column_stack(x)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_column_stack(dtype):
    """Test jnp.column_stack with various dtypes."""
    x = np.asarray([[1, 2, 3], [4, 5, 6]]).astype(dtype)
    np_result = np.column_stack(x)
    jax_result = jit_column_stack(x)
    np.testing.assert_array_equal(np_result, jax_result)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
