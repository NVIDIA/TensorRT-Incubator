# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


@jax.jit
def jit_angle(real_data, imag_data):
    complex_data = lax.complex(real_data, imag_data)
    return jnp.angle(complex_data)


def test_angle():
    """Test jnp.angle with complex numbers."""
    np.random.seed(0)
    # TODO: without the positive biases in the test data, JAX and TRT runs
    # into incorrect divide-by-zero and produces NaNs. Remove the positive bias
    # when the issue is resolved.
    data = (
        np.fix(np.random.randn(32)) + 1.0 + np.fix(np.random.randn(32)) * 1j + 1j
    ).astype(np.complex64)
    result = jit_angle(
        np.real(data).astype(np.float32), np.imag(data).astype(np.float32)
    )
    expected = np.angle(data)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
