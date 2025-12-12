# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def jit_where(arg_groups):
    return tuple(jnp.where(*group) for group in arg_groups)


def where_np(arg_groups):
    return tuple(np.where(*group) for group in arg_groups)


def test_where():
    """Test jnp.where with various dtypes."""
    arg_groups = (
        (
            np.array([True, False, True, False]),
            np.array([-1.2, 2.4, 4.0, 5.6], dtype=np.float32),
            np.array([1.2, 3.4, 0.8, -2.2], dtype=np.float32),
        ),
        (
            np.array([[True, False], [True, False]]),
            np.array([[-1.2, -2.4], [4.0, -5.6]], dtype=np.float32),
            np.array([[1.2, 3.4], [0.8, -2.2]], dtype=np.float32),
        ),
        (
            np.array([True, False, True, False]),
            np.array([-1.2, 2.4, 4.0, 5.6], dtype=np.float16),
            np.array([1.2, 3.4, 0.8, -2.2], dtype=np.float16),
        ),
    )

    # TODO: (TRT bug) We are creating a single TRT engine with mixed precision
    # types. The particular underlying TRT backend (Myelin) for
    # `tensorrt.select` will ignore the FP32 precision constraints, even with
    # "obey-precision-constraints" enabled. A warning is not given. We need to
    # raise the tolerance until this bug is fixed.
    for out, expected in zip(jit_where(arg_groups), where_np(arg_groups)):
        np.testing.assert_allclose(out, expected, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
