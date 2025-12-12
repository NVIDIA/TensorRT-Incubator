# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from pathlib import Path
from typing import Literal, Optional

import jax
import jax.ffi
import jax.numpy as jnp
import numpy as np

from .utils import (
    find_flashinfer_lib,
    get_device_compute_capability,
    match_sm_version,
)

jax.ffi.register_ffi_target(
    "mtrt_flashinfer_bmm_fp8",
    None,
    platform="mlir_tensorrt",
)


def _get_module_uri(backend: str) -> str:
    """Get the module URI for a given backend.

    Parameters
    ----------
    backend : str
        Backend identifier. Currently only ``"cutlass"`` is supported.

    Returns
    -------
    str
        Module URI string for the backend.

    Raises
    ------
    ValueError
        If the backend is not supported.

    """
    if backend == "cutlass":
        return "fp8_gemm_cutlass"
    raise ValueError(f"Unsupported backend: {backend}")


def get_num_valid_cutlass_tactics() -> int:
    """Get the number of valid CUTLASS FP8 GEMM tactics.

    Uses TVM-FFI to directly execute the required function for returning the number
    of valid tactics available for the current device.

    Returns
    -------
    int
        The number of valid CUTLASS FP8 GEMM tactics.
    """
    from tvm_ffi import load_module

    libpath = find_flashinfer_lib(_get_module_uri("cutlass"))
    lib = load_module(libpath)
    return lib.get_function("fp8_gemm_tactic_num")()


def _validate_fp8_output_dtype(dtype: jnp.dtype) -> None:
    """Validate that the output dtype is either bfloat16 or float16.

    Parameters
    ----------
    dtype : jnp.dtype
        Output data type to validate.

    Raises
    ------
    ValueError
        If the dtype is not bfloat16 or float16.
    """
    if dtype != jnp.bfloat16 and dtype != jnp.float16:
        raise ValueError(
            f"Unsupported output dtype: {dtype}. "
            f"Only bf16 and fp16 are supported for FP8 GEMM operations."
        )


def fp8_gemm_cutlass(
    a: jax.Array,
    b: jax.Array,
    scale_a: jax.Array,
    scale_b: jax.Array,
    output_dtype: jnp.dtype,
    tactic: Optional[int] = None,
) -> jax.Array:
    """Perform FP8 batched matrix multiplication using CUTLASS backend.

    Parameters
    ----------
    a : jax.Array
        Input tensor A, shape: ``(batch, m, k)``, dtype: fp8_e4m3fn or fp8_e5m2.
    b : jax.Array
        Input tensor B, shape: ``(batch, k, n)``, dtype: fp8_e4m3fn or fp8_e5m2.
        Will be transposed to ``(batch, n, k)`` internally.
    scale_a : jax.Array
        Scale tensor for A, dtype: float32.
    scale_b : jax.Array
        Scale tensor for B, dtype: float32.
    output_dtype : jnp.dtype
        Output data type, must be either bfloat16 or float16.
    tactic : Optional[int]
        CUTLASS tactic index to use. If None, defaults to 0. Must be between 0 and
        ``get_num_valid_cutlass_tactics() - 1``.

    Returns
    -------
    jax.Array
        Output tensor, shape: ``(batch, m, n)``, dtype: output_dtype.

    Raises
    ------
    ValueError
        If input dtypes are invalid, backend is unsupported, or tactic is out of range.
    """
    is_e5m2 = a.dtype == jnp.float8_e5m2 or b.dtype == jnp.float8_e5m2
    is_e4m3 = a.dtype == jnp.float8_e4m3fn or b.dtype == jnp.float8_e4m3fn
    if not is_e5m2 and not is_e4m3:
        raise ValueError("a and b must be fp8 e4m3 or fp8 e5m2")
    if scale_a.dtype != jnp.float32 or scale_b.dtype != jnp.float32:
        raise ValueError("scale_a and scale_b must be float32")

    cc = get_device_compute_capability()
    is_sm100 = match_sm_version(cc, ["100", "103", "110"])
    is_sm120 = match_sm_version(cc, ["120", "121"])

    # Validate CUTLASS backend choice
    if is_e5m2:
        raise ValueError("e5m2 is not supported for cutlass backend")
    if not is_sm100 and not is_sm120:
        raise ValueError(
            "cutlass backend is only supported on SM100, SM103, SM110, SM120, SM121"
        )
    if is_sm120:
        k_dim = a.shape[-1] if a.ndim == 2 else a.shape[2]
        if k_dim < 128:
            raise ValueError(
                "cutlass backend is only supported on SM120, SM121 with k_dim >= 128"
            )
    if tactic is None:
        tactic = 0
    if tactic < 0 or tactic >= get_num_valid_cutlass_tactics():
        raise ValueError(
            f"Invalid CUTLASS FP8 GEMM tactic number: {tactic}. Must be between 0 and {get_num_valid_cutlass_tactics() - 1}"
        )

    uri = _get_module_uri("cutlass")
    libpath = find_flashinfer_lib(uri)

    call = jax.ffi.ffi_call(
        "mtrt_flashinfer_bmm_fp8",
        jax.ShapeDtypeStruct([a.shape[0], a.shape[1], b.shape[2]], output_dtype),
    )

    arguments_spec = [
        "args.0",  # a
        "args.1",  # b.transpose
        "args.2",  # scale_a
        "args.3",  # scale_b
        "rets.0",  # out
        "args.4",  # workspace
        "attrs.tactic",  # tactic
    ]
    b = jnp.transpose(b, (0, 2, 1))
    workspace = jnp.zeros(32 * 1024 * 1024 // 2, dtype=jnp.float16)
    return call(
        a,
        b,
        scale_a,
        scale_b,
        workspace,
        tactic=tactic,
        func="fp8_gemm",
        plugin=str(libpath),
        arg_spec=";".join(arguments_spec),
        mtrt_ffi_backend="tvm_ffi",
    )  # type: ignore


def bmm_fp8(
    A: jax.Array,
    B: jax.Array,
    A_scale: jax.Array,
    B_scale: jax.Array,
    dtype: jnp.dtype,
    backend: Literal["cutlass"] = "cutlass",
    tactic: Optional[int] = None,
) -> jax.Array:
    r"""Perform batched matrix multiplication with FP8 inputs.

    Performs batched matrix multiplication using FP8 (float8) precision inputs with
    per-tensor or per-batch scaling, producing outputs in bfloat16 or float16.

    Parameters
    ----------
    A : jax.Array
        Input tensor A, shape: ``(batch, m, k)``, dtype: fp8_e4m3fn or fp8_e5m2.
    B : jax.Array
        Input tensor B, shape: ``(batch, k, n)``, dtype: fp8_e4m3fn or fp8_e5m2.
        Will be transposed internally for column-major layout.
    A_scale : jax.Array
        Scale tensor for A, dtype: float32.
    B_scale : jax.Array
        Scale tensor for B, dtype: float32.
    dtype : jnp.dtype
        Output data type, must be either bfloat16 or float16.
    backend : Literal["cutlass"]
        The backend to use for the operation. Currently only ``"cutlass"`` is supported.
        Defaults to ``"cutlass"``.
    tactic : Optional[int]
        CUTLASS tactic index to use. If None, defaults to 0. Must be between 0 and
        ``get_num_valid_cutlass_tactics() - 1``.

    Returns
    -------
    jax.Array
        Output tensor, shape: ``(batch, m, n)``, dtype: output_dtype.

    Raises
    ------
    ValueError
        If input shapes don't match, dtypes are invalid, backend is unsupported,
        or tactic is out of range.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import mlir_tensorrt_jax.flashinfer as flashinfer
    >>>
    >>> def to_float8(x, dtype=jnp.float8_e4m3fn):
    ...     finfo = jnp.finfo(dtype)
    ...     min_val, max_val = jnp.min(x), jnp.max(x)
    ...     amax = jnp.maximum(jnp.abs(min_val), jnp.abs(max_val))
    ...     amax = jnp.maximum(amax, 1e-12)
    ...     scale = finfo.max / amax
    ...     x_scl_sat = jnp.clip(x * scale, finfo.min, finfo.max)
    ...     return x_scl_sat.astype(dtype), (1.0 / scale).astype(jnp.float32)
    >>>
    >>> input = jnp.randn(jax.random.PRNGKey(0), (16, 48, 64), dtype=jnp.bfloat16)
    >>> input_fp8, input_inv_s = to_float8(input, dtype=jnp.float8_e4m3fn)
    >>> # Column-major weight (transposed)
    >>> weight = jnp.randn(jax.random.PRNGKey(1), (16, 80, 64), dtype=jnp.bfloat16)
    >>> weight = jnp.transpose(weight, (0, 2, 1))
    >>> weight_fp8, weight_inv_s = to_float8(weight, dtype=jnp.float8_e4m3fn)
    >>> out = flashinfer.bmm_fp8(input_fp8, weight_fp8, input_inv_s, weight_inv_s, jnp.bfloat16)
    >>> out.shape
    (16, 48, 80)
    >>> out.dtype
    bfloat16
    """
    _validate_fp8_output_dtype(dtype)

    if len(A.shape) != 3 or len(B.shape) != 3:
        raise ValueError("A and B must be 3D tensors")
    if A.shape[2] != B.shape[1]:
        raise ValueError("A.shape[2] must be equal to B.shape[1]")
    if A.shape[0] != B.shape[0]:
        raise ValueError("A.shape[0] must be equal to B.shape[0]")
    if A_scale.dtype != jnp.float32 or B_scale.dtype != jnp.float32:
        raise ValueError("A_scale and B_scale must be float32")

    if backend == "cutlass":
        return fp8_gemm_cutlass(A, B, A_scale, B_scale, dtype, tactic)

    raise ValueError(f"Unsupported backend: {backend}")
