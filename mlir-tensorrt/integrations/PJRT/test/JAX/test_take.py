# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax.numpy as jnp
import numpy as np
from jax import jit


@partial(jit, static_argnums=[2])
def jit_take(data, indices, axis):
    return jnp.take(data, indices, axis=axis)


@dataclass
class TestCase:
    __test__ = False

    data: np.ndarray
    indices: np.ndarray
    axis: Optional[int] = None


TEST_CASES = [
    TestCase(
        np.asarray([4, 3, 5, 7, 6, 8], dtype=np.int32),
        np.asarray([0, 1, 4], dtype=np.int32),
    ),
    TestCase(
        np.arange(0, 4 * 4 * 4, 1, dtype=np.int32).reshape((4, 4, 4)),
        np.asarray([[0, 1, 2], [1, 2, 3]]),
        axis=1,
    ),
]


def test_take():
    """Test jnp.take with various shapes and axes."""
    for test_case in TEST_CASES:
        expected = np.take(test_case.data, test_case.indices, test_case.axis)
        result = jit_take(test_case.data, test_case.indices, test_case.axis)
        np.testing.assert_equal(expected, result)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
