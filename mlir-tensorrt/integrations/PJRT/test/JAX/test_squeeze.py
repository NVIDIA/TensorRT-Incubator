# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import jax
import jax.numpy as jnp
import numpy as np

test_types = [jnp.int32, jnp.float32, jnp.float16]


@jax.jit
def jit_squeeze(input_list):
    res = [jnp.squeeze(i) for i in input_list]
    return res


def test_squeeze():
    """Test jnp.squeeze to remove dimensions of size 1."""
    # <2x1x3> tensor => <2x3> tensor
    input = [[[1, 2, 3]], [[4, 5, 6]]]
    input_list = [np.asarray(input, dtype=test_type) for test_type in test_types]
    jax_res = jit_squeeze(input_list)
    for res_tensor in jax_res:
        np.testing.assert_allclose(
            res_tensor, [[1, 2, 3], [4, 5, 6]], rtol=1e-5, atol=1e-7
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
