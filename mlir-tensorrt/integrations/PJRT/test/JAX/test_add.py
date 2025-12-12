# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def jit_add(a, b):
    return jnp.add(a, b)


def test_add():
    """Test jnp.add with various dtypes and shapes."""
    test_data_points = [np.array([1, 2, 3, 4]), np.array([[1, 2], [3, 4]])]
    test_data_types = [np.float16, np.float32, np.int16, np.uint16, np.int32, np.uint32]
    for data_point in test_data_points:
        for data_type in test_data_types:
            lhs = data_point.astype(data_type)
            rhs = data_point.astype(data_type)
            np.testing.assert_equal(jit_add(lhs, rhs), np.add(lhs, rhs))


if __name__ == "__main__":
    pytest.main(["-v", __file__])
