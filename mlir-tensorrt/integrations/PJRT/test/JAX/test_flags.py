# REQUIRES: no-asan
# REQUIRES: debug-print
# RUN: %pick-one-gpu %mlir-trt-jax-py %s

import pytest
from jax import jit
import numpy as np
import os
import tempfile
import shutil
from pathlib import Path


@jit
def jit_dummy_func(x):
    return x + x


@pytest.fixture
def ir_dump_setup():
    """Set up IR dump directory and flags before test runs."""
    ir_dump_dir = Path(tempfile.mkdtemp())

    flags = (
        f"-tensorrt-builder-opt-level=1 -debug -debug-only=translate-to-tensorrt "
        f"--mlir-print-ir-tree-dir={ir_dump_dir} --mlir-print-ir-after-all"
    )
    original_flags = os.environ.get("MLIR_TRT_FLAGS", "")
    os.environ["MLIR_TRT_FLAGS"] = flags

    yield ir_dump_dir

    # Cleanup after test
    if original_flags:
        os.environ["MLIR_TRT_FLAGS"] = original_flags
    else:
        os.environ.pop("MLIR_TRT_FLAGS", None)
    shutil.rmtree(ir_dump_dir)


@pytest.mark.requires_no_asan
@pytest.mark.debug_print
def test_flags(ir_dump_setup, is_batch_mode):
    """Test that operations work correctly with various flags set and that IR is dumped.

    This test sets runtime MLIR_TRT_FLAGS and requires process isolation.
    Skipped when running in batch mode; works when run individually or via LIT.
    """
    if is_batch_mode:
        pytest.skip(
            "Test requires process isolation. Run individually: pytest -k test_flags"
        )

    ir_dump_dir = ir_dump_setup

    input_data = np.ones((16, 16), dtype=np.float32)
    result = jit_dummy_func(input_data)
    expected = input_data + input_data
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-7)

    # Verify that IR files were created in the directory (including subdirectories)
    ir_files = list(ir_dump_dir.glob("**/*.mlir"))
    assert len(ir_files) > 0, f"Expected IR files to be created in {ir_dump_dir}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
