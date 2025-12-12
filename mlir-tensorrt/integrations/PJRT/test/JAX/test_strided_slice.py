# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import numpy as np
from jax import jit


@jit
def jit_strided_indexing(input):
    return input[:, ::2, ...], input[:, 1::2, ...], input[:, 1::2, :, 3::4, :]


def strided_indexing_np(input):
    return input[:, ::2, ...], input[:, 1::2, ...], input[:, 1::2, :, 3::4, :]


def test_strided_slice():
    """Test strided slicing with various patterns."""
    np.random.seed(0)
    data = np.random.randn(12, 128, 4, 12, 1).astype(np.float32)
    results = jit_strided_indexing(data)
    expected = strided_indexing_np(data)
    for res, exp in zip(results, expected):
        np.testing.assert_equal(res, exp)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
