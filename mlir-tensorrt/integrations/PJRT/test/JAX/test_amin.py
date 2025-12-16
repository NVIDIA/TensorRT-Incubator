# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import jax
import jax.numpy as jnp
import numpy as np

test_types = [jnp.int32, jnp.float32, jnp.float16]
test_inputs = [
    ([[1, -2, 3], [-4, 8, -7]], None),
    ([[1, -2, 3], [-4, 8, -7]], 0),
    ([[1, -2, 3], [-4, 8, -7]], 1),
]
amin_axis = tuple(i[1] for i in test_inputs)


@jax.jit
def jit_amin(input_list):
    o1 = jnp.amin(input_list[0], axis=amin_axis[0])
    o2 = jnp.amin(input_list[1], axis=amin_axis[0])
    o3 = jnp.amin(input_list[2], axis=amin_axis[0])
    o4 = jnp.amin(input_list[3], axis=amin_axis[1])
    o5 = jnp.amin(input_list[4], axis=amin_axis[1])
    o6 = jnp.amin(input_list[5], axis=amin_axis[1])
    o7 = jnp.amin(input_list[6], axis=amin_axis[2])
    o8 = jnp.amin(input_list[7], axis=amin_axis[2])
    o9 = jnp.amin(input_list[8], axis=amin_axis[2])
    return (o1, o2, o3, o4, o5, o6, o7, o8, o9)


def np_amin(input_list):
    res = [np.amin(e[0], e[1]) for e in input_list]
    return res


def amin_test(test_inputs, test_types):
    jax_input_list = [
        np.asarray(inp[0], dtype=test_type)
        for inp in test_inputs
        for test_type in test_types
    ]
    numpy_input_list = [
        (np.asarray(inp[0], dtype=test_type), inp[1])
        for inp in test_inputs
        for test_type in test_types
    ]
    jax_res = jit_amin(jax_input_list)
    np_res = np_amin(numpy_input_list)
    for x, y in zip(jax_res, np_res):
        np.testing.assert_allclose(x, y, rtol=1e-4)


def test_amin():
    """Test jnp.amin with various dtypes and axes."""
    amin_test(test_inputs, test_types)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
