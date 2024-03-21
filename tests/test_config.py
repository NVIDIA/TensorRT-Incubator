import numpy as np
import tripy as tp
import pytest


def test_timing_cache(tmp_path):
    # Create a dummy file using pytest fixture for timing cache.
    dummy_file = tmp_path / "dummy.txt"
    tp.config.timing_cache_file_path = str(dummy_file)

    @tp.jit
    def func(a, b):
        c = a + b
        return c

    a = tp.Tensor(np.array([2, 3], dtype=np.float32), device=tp.device("gpu"))
    b = tp.Tensor(np.ones(2, dtype=np.float32), device=tp.device("gpu"))

    c = func(a, b).eval()

    assert dummy_file.parent.exists(), "Cache directory was not created."
    assert dummy_file.exists(), "Cache file was not created."
    assert tp.config.timing_cache_file_path == str(
        dummy_file
    ), f"get_timing_cache_file() path does not match the user provided path {str(dummy_file)}"

    dummy_file = tmp_path / "dummy1.txt"
    tp.config.timing_cache_file_path = str(dummy_file)

    # Check if timing cache file was incorrectly generated (without recompiling).
    c = func(a, b).eval()
    assert (
        not dummy_file.exists()
    ), "Cache file created, but should not have been created since jitted func was used and no recompilation was triggered."

    # Check if new timing cache file was created.
    a = tp.Tensor(
        np.array(
            [
                2,
            ],
            dtype=np.float32,
        ),
        device=tp.device("gpu"),
    )
    b = tp.Tensor(np.ones(1, dtype=np.float32), device=tp.device("gpu"))
    c = func(a, b).eval()
    assert dummy_file.exists(), "Cache file was not created."
    assert tp.config.timing_cache_file_path == str(
        dummy_file
    ), f"get_timing_cache_file() path does not match the user provided path {str(dummy_file)}"
