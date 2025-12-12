# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import numpy as np
import jax.numpy as jnp
import jax
from functools import partial


@partial(jax.jit, static_argnums=(1,))
def jit_argmax(x, axis):
    return jnp.argmax(x, axis=axis)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_argmax(dtype, axis):
    """Test jnp.argmax with various dtypes and axes."""
    x = np.asarray([[1, -2, 3, 0], [-4, 8, -7, 0]]).astype(dtype)
    np_result = np.argmax(x, axis=axis)
    jax_result = jit_argmax(x, axis=axis)
    np.testing.assert_array_equal(np_result, jax_result)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
