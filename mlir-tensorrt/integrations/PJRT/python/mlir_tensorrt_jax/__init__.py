import logging
import os
from pathlib import Path
from typing import Optional

import jax._src.xla_bridge as xb
from jax._src.lib import xla_client

__all__ = [
    "mtrt_quantize",
    "mtrt_dequantize",
    "mtrt_dynamic_quantize",
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

    # Force initialization of tensorrt
    try:
        import tensorrt
    except:
        logger.info(
            "TensorRT package not found via Python, looking for libnvinfer.so..."
        )
        nvinfer = ctypes.util.find_library("nvinfer")
        if not nvinfer:
            raise Exception(
                "libnvinfer.so was not found.... recommend install via Python with 'pip install ...'"
            )

    # Force initializtion of tvm_ffi
    try:
        import tvm_ffi
    except Exception as e:
        raise RuntimeError(
            "Error importing tvm_ffi, which is required for MLIR-TensorRT JAX plugin: %s",
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
        from .mtrt_ops import (
            mtrt_quantize,
            mtrt_dequantize,
            mtrt_dynamic_quantize,
        )

        globals()[name] = locals()[name]
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
