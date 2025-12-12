# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
from functools import partial

import numpy as np
from jax import jit, random

# There's no equivalent numpy command for `jax.random.PRNGKey`, so we directly
# encode the expected values here. If JAX every changes the underlying
# implementation (e.g. currently it is threefry) then we will need to update
# these numbers.
expected_values = np.asarray(
    [
        [1797259609, 2579123966],
        [928981903, 3453687069],
        [507451445, 1853169794],
        [1948878966, 4237131848],
        [1821159224, 3364244817],
        [637334850, 3278974502],
    ],
    dtype=np.uint32,
).reshape(3, 2, 2)


@partial(jit, static_argnums=(0,))
def jit_get_random_key(seed):
    key = random.PRNGKey(seed)
    key1, key2 = random.split(key)
    return key1, key2


@jit
def jit_random_funcs(key1, key2):
    return random.normal(key1, shape=(4, 4)), random.uniform(key2, shape=(4, 4))


def test_random():
    """Test JAX random key generation and random functions."""
    # Test random key generation for a couple seeds
    for i in range(3):
        key1, key2 = jit_get_random_key(i)
        # For whatever reason, JAX/XLA is interpreting our result as int32.
        # TODO: fix this at PJRT level?
        np.testing.assert_array_equal(
            np.asarray(key1, dtype=np.uint32), expected_values[i, 0]
        )
        np.testing.assert_array_equal(
            np.asarray(key2, dtype=np.uint32), expected_values[i, 1]
        )

    # Test `random.normal`/`random.uniform`. These functions use JAX's pseudo-random
    # number generator, so there's no simple numpy alternative to specify. Use the
    # hard-coded expected results.
    key1, key2 = jit_get_random_key(0)
    result = jit_random_funcs(key1, key2)
    expected = [
        np.asarray(
            [
                [1.0040143, -0.9063372, -0.7481722, -1.1713669],
                [-0.8712328, 0.58883816, 0.72392994, -1.0255982],
                [1.661628, -1.8910251, -1.2889339, 0.13360691],
                [-1.1530392, 0.23929629, 1.7448071, 0.5050189],
            ],
            dtype=np.float32,
        ),
        np.asarray(
            [
                [0.00729382, 0.02089119, 0.5814265, 0.36183798],
                [0.22303772, 0.11928833, 0.12543893, 0.61683],
                [0.0950073, 0.9834225, 0.4248221, 0.8324801],
                [0.0966121, 0.22702086, 0.3545748, 0.67104137],
            ],
            dtype=np.float32,
        ),
    ]
    for res, exp in zip(result, expected):
        np.testing.assert_allclose(res, exp, rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
