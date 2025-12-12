# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import numpy as np
import jax.numpy as jnp
from jax import jit
from functools import partial


TEST_3D = np.arange(0, 32).reshape(2, 4, 4).astype(np.float32)


def my_func(a):
    return (a[0] + a[-1]) * 0.5


def my_func2(a):
    return (a[0] * 2) + 5


@partial(jit, static_argnums=[1])
def jit_apply_along_func1(data, axis):
    return jnp.apply_along_axis(my_func, axis, data)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_apply_along_axis_func1(axis):
    """Test jnp.apply_along_axis with my_func."""
    np_result = np.apply_along_axis(my_func, axis, TEST_3D)
    jax_result = jit_apply_along_func1(TEST_3D, axis)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-4, atol=1e-4)


@partial(jit, static_argnums=[1])
def jit_apply_along_func2(data, axis):
    return jnp.apply_along_axis(my_func2, axis, data)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_apply_along_axis_func2(axis):
    """Test jnp.apply_along_axis with my_func2."""
    np_result = np.apply_along_axis(my_func2, axis, TEST_3D)
    jax_result = jit_apply_along_func2(TEST_3D, axis)
    np.testing.assert_allclose(np_result, jax_result, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
