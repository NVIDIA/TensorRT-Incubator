# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import math
from typing import Literal, Optional, Tuple

import jax
import jax.ffi
import jax.numpy as jnp

from .utils import (
    PosEncodingMode,
    TensorLayout,
    check_kv_layout,
    check_pos_encoding_mode,
    filename_safe_dtype_map,
    find_flashinfer_lib,
)

jax.ffi.register_ffi_target(
    "mtrt_flashinfer_single_decode_with_kv_cache",
    None,
    platform="mlir_tensorrt",
)


def _get_uri(
    dtype_q: jnp.dtype,
    dtype_kv: jnp.dtype,
    dtype_o: jnp.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
) -> str:
    """Generate a URI string identifying a specific decode kernel configuration.

    Parameters
    ----------
    dtype_q : jnp.dtype
        Data type of query tensor.
    dtype_kv : jnp.dtype
        Data type of key/value tensors.
    dtype_o : jnp.dtype
        Data type of output tensor.
    head_dim_qk : int
        Head dimension for query/key tensors.
    head_dim_vo : int
        Head dimension for value/output tensors.
    pos_encoding_mode : int
        Position encoding mode enum value.
    use_sliding_window : bool
        Whether sliding window attention is enabled.
    use_logits_soft_cap : bool
        Whether logits soft capping is enabled.

    Returns
    -------
    str
        URI string uniquely identifying the kernel configuration.
    """
    return (
        f"single_decode_with_kv_cache_dtype_q_{filename_safe_dtype_map(dtype_q)}_"
        f"dtype_kv_{filename_safe_dtype_map(dtype_kv)}_"
        f"dtype_o_{filename_safe_dtype_map(dtype_o)}_"
        f"head_dim_qk_{head_dim_qk}_"
        f"head_dim_vo_{head_dim_vo}_"
        f"posenc_{pos_encoding_mode}_"
        f"use_swa_{use_sliding_window}_"
        f"use_logits_cap_{use_logits_soft_cap}"
    )


def single_decode_with_kv_cache(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_layout: Literal["NHD"] = "NHD",
    pos_encoding_mode: Literal["NONE", "ROPE_LLAMA", "ALIBI"] = "NONE",
    q_scale: float = 1.0,
    k_scale: float = 1.0,
    window_left: int = -1,
    logits_soft_cap: float = 0.0,
    sm_scale: Optional[float] = None,
    rope_scale: float = 1.0,
    rope_theta: float = 1e4,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    r"""Single decode attention with KV cache.

    Performs attention computation for a single decode step using cached key and value
    tensors. This function is optimized for autoregressive decoding where queries are
    processed one token at a time.

    Parameters
    ----------
    q : jax.Array
        The query tensor, shape: ``[num_qo_heads, head_dim]``.
    k : jax.Array
        The key cache tensor, shape: ``[kv_len, num_kv_heads, head_dim]``.
        This tensor will be updated in-place during the operation.
    v : jax.Array
        The value cache tensor, shape: ``[kv_len, num_kv_heads, head_dim]``.
        This tensor will be updated in-place during the operation.
    kv_layout : Literal["NHD"]
        The layout of the input k/v tensors. Currently only ``"NHD"`` is supported.
        Defaults to ``"NHD"``.
    pos_encoding_mode : Literal["NONE", "ROPE_LLAMA", "ALIBI"]
        The position encoding mode applied inside attention kernels.
        ``"NONE"``: No position encoding.
        ``"ROPE_LLAMA"``: LLAMA-style rotary position embedding.
        ``"ALIBI"``: Attention with Linear Biases (not currently supported).
        Defaults to ``"NONE"``.
    q_scale : float
        Scale factor for query tensor quantization. Defaults to ``1.0``.
    k_scale : float
        Scale factor for key tensor quantization. Defaults to ``1.0``.
    window_left : int
        The left (inclusive) window size for sliding window attention.
        When set to ``-1``, no sliding window is used. Defaults to ``-1``.
    logits_soft_cap : float
        The attention logits soft capping value. If greater than 0, logits will be
        capped according to the formula: ``logits_soft_cap * tanh(x / logits_soft_cap)``,
        where ``x`` is the input logits. Defaults to ``0.0``.
    sm_scale : Optional[float]
        The scale used in softmax computation. If not provided, will be set to
        ``1.0 / sqrt(head_dim)`` and multiplied by ``q_scale * k_scale``.
    rope_scale : float
        The scale used in RoPE interpolation. Defaults to ``1.0``.
    rope_theta : float
        The theta (base frequency) parameter used in RoPE. Defaults to ``1e4``.

    Returns
    -------
    Tuple[jax.Array, jax.Array, jax.Array]
        A tuple containing:
        - The attention output tensor, shape: ``[num_qo_heads, head_dim]``.
        - The updated K cache tensor, shape: ``[kv_len, num_kv_heads, head_dim]``.
        - The updated V cache tensor, shape: ``[kv_len, num_kv_heads, head_dim]``.

    Note
    ----
    The K and V cache tensors are updated in-place. The ``num_qo_heads`` must be a
    multiple of ``num_kv_heads`` for grouped query attention support.
    """

    check_pos_encoding_mode(pos_encoding_mode)
    check_kv_layout(kv_layout)

    if len(q.shape) != 2:
        raise ValueError("q must be a 2D tensor")
    if len(k.shape) != 3:
        raise ValueError("k must be a 3D tensor")
    if len(v.shape) != 3:
        raise ValueError("v must be a 3D tensor")

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    sm_scale *= q_scale * k_scale

    workspace = jnp.zeros(32 * 1024 * 1024 // 2, dtype=jnp.float16)

    # TODO: check preconditions on data types and shapes of q/k/v for better error messages.
    head_dim = q.shape[-1]
    use_sliding_window = window_left != -1
    use_logits_soft_cap = logits_soft_cap > 0

    # Generate the library name (flashinfer URI) based on configuration.
    uri = _get_uri(
        jnp.dtype(q.dtype),
        jnp.dtype(k.dtype),
        jnp.dtype(v.dtype),
        head_dim,
        head_dim,
        PosEncodingMode[pos_encoding_mode].value,
        use_sliding_window,
        use_logits_soft_cap,
    )

    # Locate the .so file.
    lib_path = find_flashinfer_lib(uri)

    call = jax.ffi.ffi_call(
        "mtrt_flashinfer_single_decode_with_kv_cache",
        [
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            jax.ShapeDtypeStruct(k.shape, k.dtype),
            jax.ShapeDtypeStruct(v.shape, v.dtype),
        ],
        # The K/V input tensors are aliases of the K/V output tensors.
        input_output_aliases={1: 1, 2: 2},
    )

    # This specifies how the generated `stablehlo.custom_call` operands/results map to the
    # TVM FFI call arguments.
    # fmt: off
    arguments_spec = [
        "args.0", # q,
        "args.1", # k,
        "args.2", # v,
        "args.3", # workspace
        "rets.0", # out,
        "none",   # lse (not currently supported),
        "attrs.kv_layout", # TensorLayout[kv_layout].value,
        "attrs.window_left", # window_left,
        "none", # alibi slopes,
        "attrs.logits_soft_cap", # logits_soft_cap,
        "attrs.sm_scale", # sm_scale,
        "attrs.rope_scale", # rope_scale,
        "attrs.rope_theta", # rope_theta,
    ]  # fmt: on
    assert (
        len(arguments_spec) == 13
    ), f"Expected spec for 13 arguments, got {len(arguments_spec)}"

    # fmt: off
    return call(
        q, k, v,
        workspace,
        kv_layout=TensorLayout[kv_layout].value,
        window_left=window_left,
        logits_soft_cap=logits_soft_cap,
        sm_scale=sm_scale,
        rope_scale=1.0/rope_scale,
        rope_theta=1.0/rope_theta,
        func="run",
        plugin=str(lib_path),
        arg_spec=";".join(arguments_spec),
        mtrt_ffi_backend="tvm_ffi",
    ) # type: ignore
    # fmt: on
