# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np

base_data = np.fix(np.random.randn(128, 128) * 16).astype(np.float32)
test_inputs = list(
    product([base_data], [None, 0, 1], [jnp.int32, jnp.float32, jnp.float16])
)
argmin_axis = tuple(i[1] for i in test_inputs)


@jax.jit
def jit_argmin(inputs):
    return tuple(jnp.argmin(inp, axis=argmin_axis[i]) for i, inp in enumerate(inputs))


def argmin_np(input_list):
    return tuple(
        np.argmin(inp, axis=argmin_axis[i]).astype(np.int32)
        for i, inp in enumerate(input_list)
    )


def test_argmin():
    """Test jnp.argmin with various dtypes and axes."""
    data = tuple(i[0].astype(i[2]) for i in test_inputs)
    result = jit_argmin(data)
    expected = argmin_np(data)
    for x, y in zip(result, expected):
        np.testing.assert_array_equal(x, y)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
