# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import numpy as np
import jax.numpy as jnp
from jax import jit
from functools import partial


TEST_3D = np.arange(0, 128).reshape(2, 4, 16).astype(np.float32)
# For the product test, add 1.0 to keep result
# from being zero, normalize to prevent it from being too large.
TEST_3D_NORM = (TEST_3D + 1.0) / (TEST_3D.sum() + 1.0)


@partial(jit, static_argnums=[1])
def jit_apply_over_sum(data, axes):
    return jnp.apply_over_axes(jnp.sum, data, axes)


@pytest.mark.parametrize("axes", [(0, 2), (0, 1), (1, 2)])
def test_apply_over_axes_sum(axes):
    """Test jnp.apply_over_axes with sum function."""
    np_result = np.apply_over_axes(np.sum, TEST_3D, axes)
    jax_result = jit_apply_over_sum(TEST_3D, axes)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-4, atol=1e-4)


@partial(jit, static_argnums=[1])
def jit_apply_over_prod(data, axes):
    return jnp.apply_over_axes(jnp.prod, data, axes)


@pytest.mark.parametrize("axes", [(0, 2), (0, 1), (1, 2)])
def test_apply_over_axes_prod(axes):
    """Test jnp.apply_over_axes with prod function."""
    np_result = np.apply_over_axes(np.prod, TEST_3D_NORM, axes)
    jax_result = jit_apply_over_prod(TEST_3D_NORM, axes)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
