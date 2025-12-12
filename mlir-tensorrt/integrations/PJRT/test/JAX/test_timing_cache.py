# This test verifies that the timing cache flag causes the cache to be created
# after the TRT engine is built. Note that autotuning is only done if builder level>0.
# RUN: %pick-one-gpu %mlir-trt-jax-py %s


import pytest
import jax
import jax.numpy as jnp
import numpy as np
import tempfile
from pathlib import Path
import os
import shutil


@jax.jit
def jit_computation(arg):
    return jnp.absolute(arg) * arg + arg / (arg + 1.0)


@pytest.fixture
def timing_cache_setup():
    """Set up timing cache file before test runs."""
    # Create temp directory and empty cache file
    timing_cache_file_path = Path(tempfile.mkdtemp()) / "test.timing-cache"
    timing_cache_file_path.touch()

    # Set flags BEFORE test runs
    flags = f"--tensorrt-timing-cache-path={timing_cache_file_path} --tensorrt-builder-opt-level=1"
    original_flags = os.environ.get("MLIR_TRT_FLAGS", "")
    os.environ["MLIR_TRT_FLAGS"] = flags

    yield timing_cache_file_path

    if original_flags:
        os.environ["MLIR_TRT_FLAGS"] = original_flags
    else:
        os.environ.pop("MLIR_TRT_FLAGS", None)
    shutil.rmtree(timing_cache_file_path.parent)


def test_timing_cache(timing_cache_setup, is_batch_mode):
    """Test that timing cache works with computations.

    This test sets runtime MLIR_TRT_FLAGS and requires process isolation.
    Skipped when running in batch mode; works when run individually or via LIT.
    """
    if is_batch_mode:
        pytest.skip(
            "Test requires process isolation. Run individually: pytest -k test_timing_cache"
        )

    timing_cache_file_path = timing_cache_setup

    input_data = np.zeros((1, 128, 128, 3), dtype=np.float32)
    result = jit_computation(input_data)
    expected = np.absolute(input_data) * input_data + input_data / (input_data + 1.0)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-7)

    # Verify the cache file was updated (size should be > 0 after compilation)
    assert timing_cache_file_path.exists(), "Timing cache file was not created."
    assert timing_cache_file_path.stat().st_size > 0, "Timing cache file is empty."


if __name__ == "__main__":
    pytest.main(["-v", __file__])
