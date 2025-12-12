# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def jit_arange():
    o1 = jnp.arange(start=0, stop=16)
    o2 = jnp.arange(start=0, stop=16, step=0.5)
    return [o1, o2]


def arange_np():
    e1 = np.arange(start=0, stop=16).astype(np.int32)
    e2 = np.arange(start=0, stop=16, step=0.5).astype(np.float32)
    return [e1, e2]


def test_arange():
    """Test jnp.arange with different steps."""
    result = jit_arange()
    expected = arange_np()
    for actual, expect in zip(result, expected):
        np.testing.assert_array_equal(actual, expect)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
