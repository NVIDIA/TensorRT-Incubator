# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def jit_simple_loop(index, accum):
    def body_fun_simple(args):
        index, accum = args
        return index + 1, accum + jnp.asarray(1, dtype=accum.dtype)

    def cond_fun_simple(args):
        index, accum = args
        return index < 10

    return jax.lax.while_loop(cond_fun_simple, body_fun_simple, (index, accum))


@jax.jit
def jit_sum_items_in_tensor(input_tensor):
    """This function sums individual scalars in a tensor by looping over
    elements in the tensor and using a loop carried index. This is unsupported
    by TRT <= 8.6, so it exercises our host code pipeline.
    """

    def condition_fn(args):
        index, accum = args
        return index < input_tensor.shape[0]

    def body_fn(args):
        index, accum = args
        accum = accum + input_tensor[index]
        index = index + 1
        return index, accum

    init = (np.asarray(0), np.asarray(0.0))
    return jax.lax.while_loop(condition_fn, body_fn, init)


def test_loops():
    """Test while loops with various patterns."""
    init_data = (np.asarray(0, dtype=np.int32), np.arange(0, 128, dtype=np.float32))
    np.testing.assert_array_equal(jit_simple_loop(*init_data)[1], init_data[1] + 10)

    init_data = (
        np.asarray(0, dtype=np.int32),
        np.arange(0, 128, dtype=np.float32) * 0.01,
    )
    result = jit_sum_items_in_tensor(init_data[1])[1]
    expected = np.sum(init_data[1])
    np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
