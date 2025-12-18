"""Pytest configuration for MLIR-TensorRT JAX tests.

This provides hardware and version requirement checking similar to LIT's REQUIRES directives.
Uses the same gpu_tools module that LIT uses for consistency.
"""

import pytest
import os
import sys
import shutil
from pathlib import Path


def load_gpu_tools():
    """Load the gpu_tools module."""
    try:
        # Use the same gpu_tools.py that LIT uses
        gpu_tools_path = (
            Path(__file__).parent.parent.parent.parent.parent
            / "integrations"
            / "python"
            / "mlir_tensorrt_tools"
            / "mlir_tensorrt"
            / "tools"
            / "gpu_tools.py"
        )

        if not gpu_tools_path.exists():
            # Try alternative path
            gpu_tools_path = (
                Path(__file__).parent.parent.parent.parent
                / "python"
                / "mlir_tensorrt_tools"
                / "mlir_tensorrt"
                / "tools"
                / "gpu_tools.py"
            )

        if not gpu_tools_path.exists():
            return None

        import importlib.util

        spec = importlib.util.spec_from_file_location("gpu_tools", str(gpu_tools_path))
        gpu_tools = importlib.util.module_from_spec(spec)
        sys.modules["gpu_tools"] = gpu_tools
        spec.loader.exec_module(gpu_tools)
        return gpu_tools
    except Exception as e:
        print(f"Warning: Could not load gpu_tools: {e}", file=sys.stderr)
        return None


# Load gpu_tools module once
_gpu_tools = load_gpu_tools()

ESTIMATED_GPU_MEMORY_PER_JAX_TEST_MB = 2048.0
ESTIMATED_GPU_MEMORY_PER_JAX_MODEL_TEST_MB = 8192.0


def get_tensorrt_compile_time_version():
    """Get TensorRT compile-time version from environment.

    The version should be set by the build system or environment.
    This matches how LIT gets config.mlir_tensorrt_compile_time_version.
    """
    # Check environment variable (can be set by CMake target or manually)
    version_str = os.environ.get("MLIR_TRT_TENSORRT_VERSION", "0.0")
    try:
        parts = version_str.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return (major, minor)
    except Exception:
        return (0, 0)


def in_lit_managed_environment():
    return os.environ.get("MTRT_LIT_MANAGED_ENV", "0") == "1"


def gpu_supports_fp8():
    """Check if all GPUs support FP8."""
    if _gpu_tools is None:
        return False
    try:
        return _gpu_tools.all_gpus_have_fp8_support()
    except Exception:
        return False


def gpu_supports_fp4():
    """Check if all GPUs support FP4."""
    if _gpu_tools is None:
        return False
    try:
        return _gpu_tools.all_gpus_have_fp4_support()
    except Exception:
        return False


def get_device_compute_capability() -> tuple[int, int]:
    """Get the compute capability of the current CUDA device.
    Raises RuntimeError if CUDA device enumeration or property query fails, or if no CUDA devices are found.
    Returns (major, minor) representing the compute capability version.
    """
    from cuda.bindings import runtime  # type: ignore

    status, count = runtime.cudaGetDeviceCount()
    if status != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaGetDeviceCount failed with error {status.name}")
    if count == 0:
        raise RuntimeError("No CUDA devices found")
    status, props = runtime.cudaGetDeviceProperties(0)
    if status != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaGetDeviceProperties failed with error {status.name}")
    return (props.major, props.minor)


def check_compute_capability(operator, major, minor=0):
    """Check compute capability against requirement."""
    cc_major, cc_minor = get_device_compute_capability()

    if operator == "ge":
        return (cc_major, cc_minor) >= (major, minor)
    elif operator == "eq":
        return (cc_major, cc_minor) == (major, minor)
    elif operator == "lt":
        return (cc_major, cc_minor) < (major, minor)
    else:
        raise ValueError(
            f"Invalid comparison operator: {operator}. Expected one of: ge, eq, lt"
        )


def check_tensorrt_version(operator, major, minor=0):
    """Check TensorRT compile-time version against requirement."""
    trt_major, trt_minor = get_tensorrt_compile_time_version()

    if operator == "ge":
        return (trt_major, trt_minor) >= (major, minor)
    elif operator == "eq":
        return (trt_major, trt_minor) == (major, minor)
    elif operator == "lt":
        return (trt_major, trt_minor) < (major, minor)
    else:
        raise ValueError(
            f"Invalid comparison operator: {operator}. Expected one of: ge, eq, lt"
        )


def check_gpu_count(min_gpus):
    """Check if host has at least the specified number of GPUs."""
    if _gpu_tools is None:
        return False
    try:
        return _gpu_tools.get_num_cuda_devices() >= min_gpus
    except Exception:
        return False


def check_ptx_support(min_ptx_version):
    """Check if all GPUs support at least the specified PTX version."""
    if _gpu_tools is None:
        return False
    try:
        supported_ptx_versions = _gpu_tools.get_supported_ptx_versions()
        # Check if any supported PTX version is >= min_ptx_version
        return any(ptx >= min_ptx_version for ptx in supported_ptx_versions)
    except Exception:
        return False


def check_sm_support(min_sm_version):
    """Check if all GPUs support at least the specified SM version."""
    if _gpu_tools is None:
        return False
    try:
        supported_sm_versions = _gpu_tools.get_supported_sm_versions()
        # Check if any supported SM version is >= min_sm_version
        return any(sm >= min_sm_version for sm in supported_sm_versions)
    except Exception:
        return False


# Register custom markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_fp8: test requires GPU with FP8 support (compute capability >= 8.9)",
    )
    config.addinivalue_line(
        "markers",
        "requires_fp4: test requires GPU with FP4 support (compute capability >= 8.6)",
    )
    config.addinivalue_line(
        "markers", "requires_trt_version: test requires specific TensorRT version"
    )
    config.addinivalue_line("markers", "long_test: test takes a long time to run")
    config.addinivalue_line(
        "markers",
        "unsupported_trt_version: test is not supported on specific TensorRT version",
    )
    config.addinivalue_line(
        "markers",
        "requires_compute_capability: test requires specific GPU compute capability",
    )
    config.addinivalue_line(
        "markers",
        "unsupported_compute_capability: test is not supported on specific compute capability",
    )
    config.addinivalue_line(
        "markers",
        "requires_no_asan: test requires ASAN to be disabled (AddressSanitizer conflicts)",
    )
    config.addinivalue_line(
        "markers",
        "requires_at_least_n_gpus: test requires at least N GPUs (specify n=2, n=4, etc.)",
    )
    config.addinivalue_line(
        "markers",
        "requires_minimum_ptx_version: test requires minimum PTX version support (specify version=75, etc.)",
    )
    config.addinivalue_line(
        "markers",
        "requires_minimum_sm_version: test requires minimum SM architecture version (specify version=80, etc.)",
    )
    config.addinivalue_line(
        "markers",
        "requires_nsight_systems: test requires Nsight Systems (nsys) to be available",
    )
    config.addinivalue_line(
        "markers",
        "mlir_trt_flags: set MLIR_TRT_FLAGS environment variable for this test",
    )
    config.addinivalue_line(
        "markers",
        "debug_print: test requires debug/assertions build",
    )

    # Set MLIR_TRT_FLAGS from PJRT_FLAGS environment variable if provided.
    pjrt_flags = os.environ.get("PJRT_FLAGS", "").strip()
    if pjrt_flags:
        os.environ["MLIR_TRT_FLAGS"] = pjrt_flags


def pytest_runtest_setup(item):
    """Check requirements before running each test."""
    if in_lit_managed_environment():
        # In lit managed environment, LIT manages test execution.
        return

    # Check FP8 requirement
    if item.get_closest_marker("requires_fp8"):
        if not gpu_supports_fp8():
            pytest.skip(
                "Test requires GPU with FP8 support (compute capability >= 8.9)"
            )

    # Check FP4 requirement
    if item.get_closest_marker("requires_fp4"):
        if not gpu_supports_fp4():
            pytest.skip(
                "Test requires GPU with FP4 support (compute capability >= 8.6)"
            )

    # Check TensorRT version requirement
    if marker := item.get_closest_marker("requires_trt_version"):
        operator = marker.args[0] if marker.args else "ge"
        major = marker.kwargs.get("major", 0)
        minor = marker.kwargs.get("minor", 0)
        if not check_tensorrt_version(operator, major, minor):
            pytest.skip(f"Test requires TensorRT version {operator} {major}.{minor}")

    # Check unsupported TensorRT version
    if marker := item.get_closest_marker("unsupported_trt_version"):
        operator = marker.args[0] if marker.args else "eq"
        major = marker.kwargs.get("major", 0)
        minor = marker.kwargs.get("minor", 0)
        if check_tensorrt_version(operator, major, minor):
            pytest.skip(
                f"Test is not supported on TensorRT version {operator} {major}.{minor}"
            )

    # Check compute capability requirement
    if marker := item.get_closest_marker("requires_compute_capability"):
        operator = marker.args[0] if marker.args else "ge"
        major = marker.kwargs.get("major", 0)
        minor = marker.kwargs.get("minor", 0)
        if not check_compute_capability(operator, major, minor):
            pytest.skip(f"Test requires compute capability {operator} {major}.{minor}")

    # Check unsupported compute capability
    if marker := item.get_closest_marker("unsupported_compute_capability"):
        operator = marker.args[0] if marker.args else "eq"
        major = marker.kwargs.get("major", 0)
        minor = marker.kwargs.get("minor", 0)
        if check_compute_capability(operator, major, minor):
            pytest.skip(
                f"Test is not supported on compute capability {operator} {major}.{minor}"
            )

    # Check no-ASAN requirement
    if item.get_closest_marker("requires_no_asan"):
        if os.environ.get("ENABLE_ASAN", "False") == "True":
            pytest.skip(
                "Test requires ASAN to be disabled (conflicts with AddressSanitizer)"
            )

    # Check GPU count requirement
    if marker := item.get_closest_marker("requires_at_least_n_gpus"):
        min_gpus = marker.kwargs.get("n", 1)  # Default to 1 if not specified
        if not check_gpu_count(min_gpus):
            pytest.skip(f"Test requires at least {min_gpus} GPUs")

    # Check PTX version requirement
    if marker := item.get_closest_marker("requires_minimum_ptx_version"):
        min_ptx = marker.kwargs.get("version", 70)
        if not check_ptx_support(min_ptx):
            pytest.skip(f"Test requires PTX version >= {min_ptx}")

    # Check SM version requirement
    if marker := item.get_closest_marker("requires_minimum_sm_version"):
        min_sm = marker.kwargs.get("version", 70)
        if not check_sm_support(min_sm):
            pytest.skip(f"Test requires SM architecture version >= {min_sm}")

    # Check Nsight Systems availability
    if item.get_closest_marker("requires_nsight_systems"):
        if not shutil.which("nsys"):
            pytest.skip("Test requires Nsight Systems (nsys) to be available")

    # Skip long tests unless enabled via environment variable
    if item.get_closest_marker("long_test"):
        if os.environ.get("LONG_TESTS", "False") == "False":
            pytest.skip("Long test skipped (set LONG_TESTS=True to run)")

    # If assertions are enabled, enable debug_print
    if item.get_closest_marker("debug_print"):
        if os.environ.get("LLVM_ENABLE_ASSERTIONS", "False") == "False":
            pytest.skip(
                "Test requires debug/assertions build (LLVM_ENABLE_ASSERTIONS=True)"
            )

    # Set MLIR_TRT_FLAGS for tests with mlir_trt_flags marker
    if marker := item.get_closest_marker("mlir_trt_flags"):
        test_flags = marker.args[0] if marker.args else ""
        # Store original value to restore after test
        original_flags = os.environ.get("MLIR_TRT_FLAGS", "")

        # Append test-specific flags to any existing flags
        if original_flags and test_flags:
            combined_flags = f"{original_flags} {test_flags}"
        else:
            combined_flags = test_flags or original_flags

        os.environ["MLIR_TRT_FLAGS"] = combined_flags

        # Store the original flags on the item for restoration in teardown
        item._mlir_trt_original_flags = original_flags


def pytest_runtest_teardown(item, nextitem):
    """Restore environment after test."""
    # Restore MLIR_TRT_FLAGS if it was modified by mlir_trt_flags marker
    if hasattr(item, "_mlir_trt_original_flags"):
        original_flags = item._mlir_trt_original_flags
        if not original_flags:
            os.environ.pop("MLIR_TRT_FLAGS", None)
        else:
            os.environ["MLIR_TRT_FLAGS"] = original_flags


def pytest_ignore_collect(collection_path, config):
    """Ignore non-test files during collection."""
    # Ignore pytest plugin developed for upstream JAX unit tests
    if collection_path.name == "mtrt_jax_pytest_plugin.py":
        return True
    # Ignore lit configuration files
    if collection_path.name.endswith(".cfg"):
        return True
    # Ignore .test files (used by LIT)
    if collection_path.name.endswith(".test"):
        return True
    return None  # Let other hooks decide


@pytest.fixture(scope="session")
def is_batch_mode(request):
    """Detect if running multiple tests (batch mode) vs single test."""
    # If more than 1 test item is collected, we're in batch mode
    return len(request.session.items) > 1


@pytest.fixture(scope="function", autouse=True)
def pick_one_gpu(request):
    """Pick one GPU for the test and set CUDA_VISIBLE_DEVICES."""
    if in_lit_managed_environment():
        # In lit managed environment, LIT manages test execution.
        yield
        return

    if _gpu_tools is None:
        yield
        return

    # Determine required memory (model tests need more)
    required_memory_mb = (
        ESTIMATED_GPU_MEMORY_PER_JAX_MODEL_TEST_MB
        if any(
            request.node.get_closest_marker(m)
            for m in [
                "requires_functional_suite",
                "requires_benchmark_suite",
                "requires_mini_benchmark",
            ]
        )
        else ESTIMATED_GPU_MEMORY_PER_JAX_TEST_MB
    )

    try:
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mlir_tensorrt.tools.gpu_tools",
                "pick-device",
                "--required-memory",
                str(int(required_memory_mb)),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        gpu_id = result.stdout.strip()

        if gpu_id and gpu_id.isdigit():
            original = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            yield
            if original is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original
        else:
            yield
    except Exception:
        yield
