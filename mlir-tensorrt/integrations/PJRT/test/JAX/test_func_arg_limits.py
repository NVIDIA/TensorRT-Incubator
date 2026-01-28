# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import jax
import numpy as np


@jax.jit
def jit_add_many_args(*args):
    """JIT-compiled function that adds together many arguments."""
    # Sum all input arguments
    result = [args[0]]
    for arg in args[1:]:
        result.append(result[-1] + arg)
    return tuple(result)


def test_many_arguments_and_returns():
    """Test JAX JIT compilation with 300 arguments and 300 return values.

    This test verifies that the PJRT/TensorRT backend can handle functions
    with a very large number of arguments and return values, which stress-tests
    the ABI and argument passing mechanisms.
    """
    # Create 300 input arrays
    num_args = 300
    inputs = [np.array([float(i)], dtype=np.float32) for i in range(num_args)]

    # Call the JIT-compiled function
    results = jit_add_many_args(*inputs)

    # Verify we got 300 results back
    assert len(results) == 300, f"Expected 300 results, got {len(results)}"

    # Verify each result
    for i, result in enumerate(results):
        expected_value = sum(range(i + 1))
        np.testing.assert_allclose(
            result,
            np.array([expected_value], dtype=np.float32),
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Result {i} mismatch",
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
