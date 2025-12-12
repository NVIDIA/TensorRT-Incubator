# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import sys
from enum import Enum
from pathlib import Path
from typing import Union

import flashinfer_jit_cache
import jax
import jax.numpy as jnp
import numpy as np


def create_flashinfer_import_error() -> ImportError:
    """Create a helpful ImportError message for missing flashinfer_jit_cache.

    Returns
    -------
    ImportError
        An ImportError with installation instructions for flashinfer_jit_cache.
    """
    urls = [
        "https://flashinfer.ai/whl/cu129",
        "https://flashinfer.ai/whl/cu130",
    ]
    error = "flashinfer_jit_cache is not installed. You can install it from the FlashInfer index, for example: \n"
    examples = [
        f" > pip install flashinfer-jit-cache --index-url {url}" for url in urls
    ]
    error += "\n".join(examples)
    return ImportError(error)


def filename_safe_dtype_map(dtype: Union[jnp.dtype, np.dtype]) -> str:
    """Map a numpy/JAX dtype to a filename-safe string representation.

    Parameters
    ----------
    dtype : Union[jnp.dtype, np.dtype]
        The data type to convert.

    Returns
    -------
    str
        Filename-safe string representation of the dtype.

    Raises
    ------
    ValueError
        If the dtype is not supported.
    """
    if dtype == jnp.float16:
        return "f16"
    elif dtype == jnp.bfloat16:
        return "bf16"
    elif dtype == jnp.float8_e4m3fn:
        return "e4m3"
    elif dtype == jnp.float8_e5m2:
        return "e5m2"
    elif dtype == jnp.int8:
        return "i8"
    elif dtype == jnp.uint8:
        return "u8"
    elif dtype == jnp.int32:
        return "i32"
    elif dtype == jnp.uint32:
        return "u32"
    elif dtype == jnp.int64:
        return "i64"
    elif dtype == jnp.uint64:
        return "u64"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def is_float8(x: Union[jnp.dtype, np.dtype, jax.Array]) -> bool:
    """Check if a dtype or array uses FP8 (float8) precision.

    Parameters
    ----------
    x : Union[jnp.dtype, np.dtype, jax.Array]
        Either a dtype or an array to check.

    Returns
    -------
    bool
        True if x is an FP8 dtype (e4m3fn or e5m2), False otherwise.
    """
    if isinstance(x, jnp.dtype):
        return x in [jnp.float8_e4m3fn, jnp.float8_e5m2]
    return is_float8(x.dtype)


class PosEncodingMode(Enum):
    """Position encoding modes for attention mechanisms."""

    NONE = 0
    ROPE_LLAMA = 1
    ALIBI = 2


class TensorLayout(Enum):
    """Tensor layout modes for key/value cache tensors."""

    NHD = 0  # Sequence length, num heads, head dimension
    HND = 1  # Num heads, sequence length, head dimension


class MaskMode(Enum):
    """Mask modes for attention computation."""

    NON_CAUSAL = 0
    CAUSAL = 1
    CUSTOM = 2
    MULTIITEMSCORING = 3


def check_pos_encoding_mode(pos_encoding_mode: str) -> None:
    """Validate position encoding mode string.

    Parameters
    ----------
    pos_encoding_mode : str
        Position encoding mode string to validate.

    Raises
    ------
    KeyError
        If the position encoding mode is invalid.
    ValueError
        If ALIBI mode is specified (not currently supported).
    """
    if not hasattr(PosEncodingMode, pos_encoding_mode):
        raise KeyError("Invalid pos_encoding_mode {}".format(pos_encoding_mode))

    if pos_encoding_mode == PosEncodingMode.ALIBI.name:
        raise ValueError(
            "ALIBI position encoding is not supported due to requiring a strong reference tensor to the TVM FFI call"
        )


def check_kv_layout(kv_layout: str) -> None:
    """Validate key/value tensor layout string.

    Parameters
    ----------
    kv_layout : str
        Layout string to validate.

    Raises
    ------
    KeyError
        If the layout string is invalid.
    ValueError
        If a layout other than NHD is specified (currently only NHD is supported).
    """
    if not hasattr(TensorLayout, kv_layout):
        raise KeyError("Invalid kv_layout {}".format(kv_layout))

    if TensorLayout.NHD.name != kv_layout:
        raise ValueError("only NHD layout is currently supported")


def get_device_compute_capability() -> tuple[int, int]:
    """Get the compute capability of the current CUDA device.

    Returns
    -------
    tuple[int, int]
        A tuple (major, minor) representing the compute capability version.

    Raises
    ------
    RuntimeError
        If CUDA device enumeration or property query fails, or if no CUDA devices
        are found.
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


def match_sm_version(cc: tuple[int, int], sm_version: list[str]) -> bool:
    """Check if compute capability matches any of the specified SM versions.

    Parameters
    ----------
    cc : tuple[int, int]
        Compute capability tuple (major, minor).
    sm_version : list[str]
        List of SM version strings to match against (e.g., ["100", "103"]).

    Returns
    -------
    bool
        True if the device compute capability matches any SM version in the list.
    """
    major, minor = cc
    device_arch = f"{major * 10 + minor}"
    return device_arch in sm_version


def is_sm90a_supported() -> bool:
    """Check if the current device supports SM90a architecture (Hopper).

    Returns
    -------
    bool
        True if the device has compute capability 9.x (Hopper architecture).
    """
    major, _ = get_device_compute_capability()
    return major == 9


def find_flashinfer_lib(uri: str) -> str:
    """Find a FlashInfer FFI library file in the JIT cache.

    Parameters
    ----------
    uri : str
        URI identifier for the library module (without extension).

    Returns
    -------
    str
        Path to the library file (.so on Linux/Mac, .dll on Windows).

    Raises
    ------
    FileNotFoundError
        If the library file cannot be found in the cache directory.
    """
    cache_dir = flashinfer_jit_cache.get_jit_cache_dir()
    libname = uri + ".dll" if sys.platform == "win32" else uri + ".so"
    try:
        lib_path = next(Path(cache_dir).glob(f"**/{libname}"))
    except StopIteration:
        raise FileNotFoundError(
            f"Failed to find cached FFI module {libname} in {cache_dir}.\n"
            "This is caused by one of the following reasons:\n"
            "1. Check to ensure that the correct version of flashinfer_jit_cache is installed.\n"
            "2. This particular op/backend configuration is not yet available as a pre-compiled FFI module.\n"
        )
    return str(lib_path)
