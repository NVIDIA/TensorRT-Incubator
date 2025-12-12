# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def jit_arccos(i_1d_f32, i_2d_f32):
    o_1d_f32 = jnp.arccos(i_1d_f32)
    o_2d_f32 = jnp.arccos(i_2d_f32)
    return (o_1d_f32, o_2d_f32)


def test_arccos():
    """Test jnp.arccos with float32."""
    np_1d_f32 = np.asarray([1, -1], dtype=np.float32)
    jnp_1d_f32 = np.asarray([1, -1], dtype=jnp.float32)

    np_2d_f32 = np.asarray([[0.34, 0.86], [0.543, 0.123]], dtype=np.float32)
    jnp_2d_f32 = np.asarray([[0.34, 0.86], [0.543, 0.123]], dtype=np.float32)

    # TODO (sagar): Add F16 test cases. On hold due to unsupported `mhlo.atan2` conversion

    o = jit_arccos(jnp_1d_f32, jnp_2d_f32)
    np.testing.assert_allclose(o[0], np.arccos(np_1d_f32), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(o[1], np.arccos(np_2d_f32), rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
