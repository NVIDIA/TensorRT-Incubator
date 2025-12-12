# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import numpy as np
import jax.numpy as jnp
from jax import jit
from jax import lax


@jit
def jit_multi_reduce_einsum(i):
    # reduces the first and last dimensions
    return jnp.einsum("a...a", i, precision=lax.Precision.HIGHEST)


def test_einsum_regression():
    """Test einsum regression with multi-dimensional reduction."""
    i = np.random.rand(3, 2, 3)
    np_r = np.einsum("a...a", i)
    jax_r = jit_multi_reduce_einsum(i)
    np.testing.assert_allclose(np_r, jax_r, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
