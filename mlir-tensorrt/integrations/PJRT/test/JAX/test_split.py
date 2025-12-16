# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import jax
import jax.numpy as jnp
import numpy as np

test_types = [jnp.float32, jnp.float16, jnp.int32]


@jax.jit
def jit_split(input_1D_list, input_2D_list, input_3D_list):
    res_1D_int_index = [jnp.split(input_1D, 3) for input_1D in input_1D_list]
    res_1D_array_indices = [jnp.split(input_1D, [1, 3]) for input_1D in input_1D_list]
    res_2D_int_index = [jnp.split(input_2D, 3) for input_2D in input_2D_list]
    res_2D_array_indices = [
        jnp.split(input_2D, [1, 2], 1) for input_2D in input_2D_list
    ]
    res_3D = [jnp.split(input_3D, [0], 1) for input_3D in input_3D_list]
    return (
        res_1D_int_index,
        res_1D_array_indices,
        res_2D_int_index,
        res_2D_array_indices,
        res_3D,
    )


def split_np(input_1D_list, input_2D_list, input_3D_list):
    res_1D_int_index = [np.split(input_1D, 3) for input_1D in input_1D_list]
    res_1D_array_indices = [np.split(input_1D, [1, 3]) for input_1D in input_1D_list]
    res_2D_int_index = [np.split(input_2D, 3) for input_2D in input_2D_list]
    res_2D_array_indices = [np.split(input_2D, [1, 2], 1) for input_2D in input_2D_list]
    res_3D = [np.split(input_3D, [0], 1) for input_3D in input_3D_list]
    return (
        res_1D_int_index,
        res_1D_array_indices,
        res_2D_int_index,
        res_2D_array_indices,
        res_3D,
    )


def test_split():
    """Test jnp.split with various shapes and dtypes."""
    # test 1-D array
    input_1D = [1, 2, 3, 4, 5, 6]
    # test 2-D array
    input_2D = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # test empty array
    input_3D = np.asarray([]).reshape(2, 0, 4)

    input_1D_list = [np.asarray(input_1D, test_type) for test_type in test_types]
    input_2D_list = [np.asarray(input_2D, test_type) for test_type in test_types]
    input_3D_list = [input_3D.astype(test_type) for test_type in test_types]

    jax_res = jit_split(input_1D_list, input_2D_list, input_3D_list)
    np_res = split_np(
        np.array(input_1D_list), np.array(input_2D_list), np.array(input_3D_list)
    )
    for x, y in zip(jax_res, np_res):
        for xi, yi in zip(x, y):
            for x_arr, y_arr in zip(xi, yi):
                np.testing.assert_allclose(x_arr, y_arr, rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
