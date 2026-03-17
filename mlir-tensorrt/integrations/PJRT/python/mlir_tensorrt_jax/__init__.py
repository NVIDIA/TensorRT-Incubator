from pathlib import Path
from typing import Optional
import ctypes.util
import logging
import os

import jax._src.xla_bridge as xb
from jax._src.lib import xla_client

__all__ = [
    "mtrt_quantize",
    "mtrt_dequantize",
    "mtrt_dynamic_quantize",
    "mtrt_nvtx_push",
    "mtrt_nvtx_pop",
    "mtrt_nvtx_annotate",
    "NVTX_COLOR",
    "initialize",
    "configure_jax_backend",
]


def configure_jax_backend(
    pjrt_opt_level: int = 3,
    builder_opt_level: int = 3,
    enable_debug: bool = False,
    debug_dir: Optional[str] = None,
    any_other_flags: Optional[str] = None,
):
    """
    Configure JAX backend before JAX is imported.

    MUST be called before any JAX imports!

    Args:
        pjrt_opt_level: MLIR-TensorRT PJRT optimization level
        builder_opt_level: TensorRT builder optimization level
        enable_debug: Enable MLIR IR dumping (generates large files)
        debug_dir: Directory to save MLIR IR after every compiler pass
        any_other_flags: String of space separated flags recognized by MLIR-TensorRT
    """
    os.environ["JAX_PLATFORMS"] = "mlir_tensorrt"
    os.environ["MLIR_TRT_FLAGS"] = (
        f"--mtrt-pjrt-opt-level={pjrt_opt_level} --tensorrt-builder-opt-level={builder_opt_level}"
    )
    if enable_debug:
        if debug_dir is None:
            raise ValueError("debug_dir is required when enable_debug is True")
        os.environ[
            "MLIR_TRT_FLAGS"
        ] += f" --mlir-print-ir-tree-dir={debug_dir} --mlir-print-ir-after-all"
    if any_other_flags is not None:
        os.environ["MLIR_TRT_FLAGS"] += f" {any_other_flags}"


logger = logging.getLogger(__name__)


def _get_library_path():
    installed_path = Path(os.path.dirname(__file__)) / "libmlir-tensorrt-pjrt.so"
    if installed_path.exists():
        return installed_path
    return None


def initialize():
    """Called by JAX during plugin discovery.

    This function is called automatically by JAX when it discovers this plugin
    via the `jax_plugins` entry point defined in pyproject.toml.
    """
    path = _get_library_path()
    if path is None:
        raise RuntimeError("MLIR-TensorRT PJRT plugin library not found")
    c_api = None

    is_initialized = False
    try:
        is_initialized = xla_client.pjrt_plugin_loaded(
            "mlir_tensorrt"
        ) and xla_client.pjrt_plugin_initialized("mlir_tensorrt")
    except Exception as e:
        is_initialized = False

    # Attempt to ensure that the 'libnvinfer.so' library is available.
    # This is 'best effort'. We first try to load the Python package 'tensorrt', since
    # most users of the published Python package will install TensorRT via binary
    # wheels published to Python package indices (e.g. PyPi, see
    # https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html#method-1-python-package-index-pip).

    # However, in development/testing environments and on embedded platforms, TensorRT
    # is not installed via binary wheel files. Instead, the user or build system installs a TensorRT
    # binary package manually and updates the process environment (e.g. LD_LIBRARY_PATH)
    # to ensure that the process can find the dynamic library.
    try:
        # First, try to import the Python package 'tensorrt'.
        import tensorrt

        logger.debug("Successfully imported Python package 'tensorrt'")
    except Exception as e:
        # If the import fails, don't give up.
        logger.debug("Failed to import Python package 'tensorrt': %s", e)

        libnvinfer_path = ctypes.util.find_library("nvinfer")
        if not libnvinfer_path:
            raise Exception(
                "The 'nvinfer' dynamic library was not found. Either TensorRT is not installed or the process environment is not configured to find it."
                "\n - If you are on an x86 platform, try installing TensorRT via the binary wheels published to PyPi ('pip install tensorrt-cu13' for CUDA 13.x or 'pip install tensorrt-cu12' for CUDA 12.x)."
                "\n - If you are on an embedded platform (e.g. NVIDIA Jetson), ensure you have installed TensorRT and that the process' LD_LIBRARY_PATH is pointing to the directory containing the 'libnvinfer.so' dynamic library."
            )

        logger.info(f"Found 'nvinfer' dynamic library at '{libnvinfer_path}'")

    # Force initialization of tvm_ffi
    try:
        import tvm_ffi

        logger.debug("Successfully imported Python package 'tvm_ffi'")
    except Exception as e:
        logger.error(
            "Failed to import Python package 'tvm_ffi'. Ensure 'apache-tvm-ffi' is installed: %s",
            e,
        )

    if not is_initialized:
        c_api = xb.register_plugin(
            "mlir_tensorrt", priority=600, library_path=str(path), options=None
        )
        xla_client.initialize_pjrt_plugin("mlir_tensorrt")

    # Register MLIR lowerings after the platform is registered
    from .mtrt_ops import register_all_lowerings

    register_all_lowerings()

    logger.info("MLIR-TensorRT JAX plugin initialized successfully")
    return c_api


# Lazy import for mtrt_ops functions
def __getattr__(name):
    if name in ["mtrt_quantize", "mtrt_dequantize", "mtrt_dynamic_quantize"]:
        from .mtrt_ops import mtrt_dequantize, mtrt_dynamic_quantize, mtrt_quantize

        globals()[name] = locals()[name]
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
