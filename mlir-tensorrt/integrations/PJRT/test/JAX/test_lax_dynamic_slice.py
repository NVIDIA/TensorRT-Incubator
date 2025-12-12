# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def jit_dynamic_slice(input, offt0, offt1):
    """Regression test for shape tensor handling in dynamic slices.

    The offsets in the slice are regarded by TensorRT as shape tensors and must be
    copied to the host. When the shape tensors are coming directly from input arguments
    of the main JIT-compiled function, they will be owned by the caller (PJRT) and thus
    will be 'views' from the runtime's perspective. Since they will be copied, the
    size information must be propagated correctly. Previously there was a bug that
    resulted in a crash.
    """
    offt0 = offt0.reshape((1,))
    offt1 = offt1.reshape((1,))
    offsets = jax.lax.concatenate((offt0, offt1), dimension=0)
    # We must directly bind the XLA generator rather than using
    # `lax.dynamic_slice` since the LAX version inserts index normalization
    # logic that will hide the things we are trying to test.
    jax.lax.dynamic_slice
    return jax.lax.dynamic_slice_p.bind(input, *offsets, *[], slice_sizes=(1, 1))


def test_lax_dynamic_slice():
    """Test lax.dynamic_slice with shape tensor handling."""
    # Create an array of all ones, but set the value to be sliced to '42'.
    inp = np.ones([10, 10], dtype=np.float32)
    inp[2, 4] = 42.0
    expected = np.asarray(42.0, dtype=np.float32).reshape(1, 1)
    offt0 = jnp.array(2, dtype=np.uint32)
    offt1 = jnp.array(4, dtype=np.uint32)
    result = jit_dynamic_slice(inp, offt0, offt1)
    np.testing.assert_equal(result, expected)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
