# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""FlashInfer attention operations for MLIR TensorRT JAX integration."""

# Import tvm_ffi just to ensure that the shared library is loaded.
import tvm_ffi  # type: ignore

from .decode import single_decode_with_kv_cache
from .gemm import bmm_fp8, get_num_valid_cutlass_tactics
from .prefill import single_prefill_with_kv_cache

__all__ = [
    "bmm_fp8",
    "get_num_valid_cutlass_tactics",
    "single_decode_with_kv_cache",
    "single_prefill_with_kv_cache",
]
