# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def jit_arctan2(i_f32_1d, i_f32_2d):
    o_f32_1d = jnp.arctan2(i_f32_1d[0], i_f32_1d[1])
    o_f32_2d = jnp.arctan2(i_f32_2d[0], i_f32_2d[1])
    return (o_f32_1d, o_f32_2d)


def test_arctan2():
    """Test jnp.arctan2 with float32."""
    value_1d_0 = [0.26, 0.39]
    value_1d_1 = [0.56, -1]

    value_2d_0 = [[-1, 1], [0.33, 0.97]]
    value_2d_1 = [[-1, -1], [0.23, 0.45]]

    # TODO: (sagar) Add FP16 tests after `mhlo.atan2` support is added
    o = jit_arctan2(
        (
            np.asarray(value_1d_0, dtype=jnp.float32),
            np.asarray(value_1d_1, dtype=jnp.float32),
        ),
        (
            np.asarray(value_2d_0, dtype=jnp.float32),
            np.asarray(value_2d_1, dtype=jnp.float32),
        ),
    )

    np.testing.assert_allclose(
        o[0],
        np.arctan2(
            np.asarray(value_1d_0, dtype=np.float32),
            np.asarray(value_1d_1, dtype=np.float32),
        ),
        rtol=1e-4,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        o[1],
        np.arctan2(
            np.asarray(value_2d_0, dtype=np.float32),
            np.asarray(value_2d_1, dtype=np.float32),
        ),
        rtol=1e-4,
        atol=1e-6,
    )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
