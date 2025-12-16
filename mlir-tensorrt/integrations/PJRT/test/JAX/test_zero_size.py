# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import jax
import jax.numpy as jnp


@jax.jit
def jit_add_tensors(a, b):
    return a + b


def test_zero_size():
    """Test operations on zero-sized tensors."""
    A = jnp.ones((128, 0), dtype=jnp.float32)
    B = jit_add_tensors(A, A)
    assert B.shape == (128, 0), "expected result shape (128, 0)"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
