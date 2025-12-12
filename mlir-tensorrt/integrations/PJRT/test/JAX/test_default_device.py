# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
import jax
import jax.numpy as jnp


def test_default_device():
    """Test jax.default_device with multi-device systems."""
    # This test is just for multi-device systems
    if jax.device_count() < 2:
        return  # Skip test on single-device systems

    # Perform a computation on a single device
    # that is not the first device.
    with jax.default_device(jax.devices()[-1]):
        assert jnp.add(1, 1).devices() == {
            jax.devices()[-1]
        }, "Expected the result's devices to be the last device"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
