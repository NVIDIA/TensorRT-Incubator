# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import math
from typing import Optional, Tuple, Union

import jax
import jax.ffi
import jax.numpy as jnp

from .utils import (
    MaskMode,
    PosEncodingMode,
    TensorLayout,
    check_kv_layout,
    check_pos_encoding_mode,
    filename_safe_dtype_map,
    find_flashinfer_lib,
    is_float8,
    is_sm90a_supported,
)

jax.ffi.register_ffi_target(
    "mtrt_flashinfer_single_prefill_with_kv_cache",
    None,
    platform="mlir_tensorrt",
)


def get_uri(
    backend: str,
    dtype_q: jnp.dtype,
    dtype_kv: jnp.dtype,
    dtype_o: jnp.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
    use_fp16_qk_reduction: bool,
) -> str:
    """Generate a URI string identifying a specific prefill kernel configuration.

    Parameters
    ----------
    backend : str
        Backend identifier (e.g., "fa2", "fa3").
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
    use_fp16_qk_reduction : bool
        Whether FP16 QK reduction is enabled.

    Returns
    -------
    str
        URI string uniquely identifying the kernel configuration.
    """
    return (
        f"single_prefill_with_kv_cache_dtype_q_{filename_safe_dtype_map(dtype_q)}_"
        f"dtype_kv_{filename_safe_dtype_map(dtype_kv)}_"
        f"dtype_o_{filename_safe_dtype_map(dtype_o)}_"
        f"head_dim_qk_{head_dim_qk}_"
        f"head_dim_vo_{head_dim_vo}_"
        f"posenc_{pos_encoding_mode}_"
        f"use_swa_{use_sliding_window}_"
        f"use_logits_cap_{use_logits_soft_cap}_"
        f"f16qk_{use_fp16_qk_reduction}" + ("_sm90" if backend == "fa3" else "")
    )


def _is_fa3_backend_supported(
    pos_encoding_mode: int,
    use_fp16_qk_reductions: bool,
    use_custom_mask: bool,
) -> bool:
    """Check if FA3 backend is supported for the given configuration.

    Parameters
    ----------
    pos_encoding_mode : int
        Position encoding mode enum value.
    use_fp16_qk_reductions : bool
        Whether FP16 QK reduction is enabled.
    use_custom_mask : bool
        Whether custom mask is being used.

    Returns
    -------
    bool
        True if FA3 backend is supported, False otherwise.
    """
    if use_custom_mask:
        return False
    if pos_encoding_mode != PosEncodingMode.NONE.value:
        return False
    if use_fp16_qk_reductions:
        return False
    return True


def _determine_backend(
    pos_encoding_mode: int,
    use_fp16_qk_reductions: bool,
    use_custom_mask: bool,
) -> str:
    """Determine the appropriate backend for prefill attention.

    Automatically selects between FA2 and FA3 backends based on device capabilities
    and kernel configuration requirements.

    Parameters
    ----------
    pos_encoding_mode : int
        Position encoding mode enum value.
    use_fp16_qk_reductions : bool
        Whether FP16 QK reduction is enabled.
    use_custom_mask : bool
        Whether custom mask is being used.

    Returns
    -------
    str
        Backend identifier ("fa2" or "fa3").
    """
    if is_sm90a_supported() and _is_fa3_backend_supported(
        pos_encoding_mode,
        use_fp16_qk_reductions,
        use_custom_mask,
    ):
        return "fa3"
    else:
        return "fa2"


def single_prefill_with_kv_cache(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    scale_q: Optional[jax.Array] = None,
    scale_k: Optional[jax.Array] = None,
    scale_v: Optional[jax.Array] = None,
    custom_mask: Optional[jax.Array] = None,
    packed_custom_mask: Optional[jax.Array] = None,
    causal: bool = False,
    kv_layout: str = "NHD",
    pos_encoding_mode: str = "NONE",
    use_fp16_qk_reduction: bool = False,
    sm_scale: Optional[float] = None,
    window_left: int = -1,
    logits_soft_cap: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    backend: str = "auto",
    return_lse: bool = False,
) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
    r"""Prefill/Append attention with KV cache for single request, return the attention
    output.

    Parameters
    ----------
    q : jax.Array
        The query tensor, shape: ``[qo_len, num_qo_heads, head_dim_qk]``.
    k : jax.Array
        The key tensor, shape: ``[kv_len, num_kv_heads, head_dim_qk]`` if :attr:`kv_layout`
        is ``NHD``, or ``[num_kv_heads, kv_len, head_dim_qk]`` if :attr:`kv_layout` is
        ``HND``.
    v : jax.Array
        The key tensor, shape: ``[kv_len, num_kv_heads, head_dim_vo]`` if :attr:`kv_layout`
        is ``NHD``, ``[num_kv_heads, kv_len, head_dim_vo]`` if :attr:`kv_layout` is
        ``HND``.
    scale_q : Optional[jax.Array]
        The scale tensor for query, per-head quantization with shape: ``[num_qo_heads]``.
        Used with FP8 Quantization. If not provided, will be set to ``1.0``.
    scale_k : Optional[jax.Array]
        The scale tensor for key, per-head quantization with shape: ``[num_kv_heads]``.
        Used with FP8 Quantization. If not provided, will be set to ``1.0``.
    scale_v : Optional[jax.Array]
        The scale tensor for value, per-head quantization with shape: ``[num_kv_heads]``.
        Used with FP8 Quantization. If not provided, will be set to ``1.0``.

    custom_mask : Optional[jax.Array]
        The custom boolean mask tensor, shape: ``[qo_len, kv_len]``.
        The elements in the mask tensor should be either ``True`` or ``False``,
        where ``False`` means the corresponding element in the attention matrix will be
        masked out.

        When :attr:`custom_mask` is provided, and :attr:`packed_custom_mask` is not, the
        function will pack the custom mask tensor into a 1D packed mask tensor, which introduces
        additional overhead.
    packed_custom_mask : Optional[jax.Array]
        The 1D packed uint8 mask tensor, if provided, the :attr:`custom_mask` will be ignored.
        The packed mask tensor is generated by :func:`flashinfer.quantization.packbits`.
    causal : bool
        Whether to apply causal mask to the attention matrix.
        This is only effective when :attr:`custom_mask` is not provided.
    kv_layout : str
        The layout of the input k/v tensors. Currently only ``"NHD"`` is supported.
    pos_encoding_mode : str
        The position encoding applied inside attention kernels, could be
        ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
        Default is ``NONE``.
    use_fp16_qk_reduction : bool
        Whether to use f16 for qk reduction (faster at the cost of slight precision
        loss).
    window_left : int
        The left (inclusive) window size for the attention window, when set to ``-1``, the window
        size will be set to the full length of the sequence. Defaults to ``-1``.
    logits_soft_cap : Optional[float]
        The attention logits soft capping value (used in Gemini, Grok and Gemma-2, etc.), if not
        provided, will be set to ``0``. If greater than 0, the logits will be capped according to
        formula:
        :math:`\texttt{logits_soft_cap} \times \mathrm{tanh}(x / \texttt{logits_soft_cap})`,
        where :math:`x` is the input logits.
    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim_qk)``.
    rope_scale : Optional[float]
        The scale used in RoPE interpolation, if not provided, will be set to 1.0.
    rope_theta : Optional[float]
        The theta used in RoPE, if not provided, will be set to 1e4.
    backend : str
        The implementation backend, could be ``auto``/``fa2`` or ``fa3``. Defaults to ``auto``.
        If set to ``auto``, the function will automatically choose the backend based on the
        device architecture and kernel availability.
    return_lse : bool
        Whether to return the log sum exp value of the attention logits.

    Returns
    -------
    Union[jax.Array, Tuple[jax.Array, jax.Array]]
        If :attr:`return_lse` is ``False``, the attention output, shape: ``[qo_len, num_qo_heads, head_dim_vo]``.
        If :attr:`return_lse` is ``True``, a tuple of two tensors:

        * The attention output, shape: ``[qo_len, num_qo_heads, head_dim_vo]``.
        * The log sum exp value, shape: ``[qo_len, num_qo_heads]``.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import mlir_tensorrt_jax.flashinfer as flashinfer
    >>>
    >>> qo_len = 128
    >>> kv_len = 4096
    >>> num_qo_heads = 32
    >>> num_kv_heads = 4
    >>> head_dim = 128
    >>>
    >>> key = jax.random.PRNGKey(0)
    >>> q = jax.random.normal(key, (qo_len, num_qo_heads, head_dim), dtype=jnp.float16)
    >>> k = jax.random.normal(key, (kv_len, num_kv_heads, head_dim), dtype=jnp.float16)
    >>> v = jax.random.normal(key, (kv_len, num_kv_heads, head_dim), dtype=jnp.float16)
    >>>
    >>> o = flashinfer.single_prefill_with_kv_cache(
    ...     q, k, v, causal=True, use_fp16_qk_reduction=True
    ... )
    >>> o.shape
    (128, 32, 128)

    Note
    ----
    The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads`` is
    not equal to ``num_kv_heads``, the function will use
    `grouped query attention <https://arxiv.org/abs/2305.13245>`_.
    """
    check_pos_encoding_mode(pos_encoding_mode)
    check_kv_layout(kv_layout)
    if logits_soft_cap is None:
        logits_soft_cap = 0.0
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    if rope_scale is None:
        rope_scale = 1.0
    if rope_theta is None:
        rope_theta = 1e4
    if custom_mask is not None and packed_custom_mask is None:
        raise ValueError(
            "packed_custom_mask is not yet supported due to FFI encoding limitations"
        )

    if packed_custom_mask is not None:
        mask_mode = MaskMode.CUSTOM.value
    else:
        if causal:
            mask_mode = MaskMode.CAUSAL.value
        else:
            mask_mode = MaskMode.NON_CAUSAL.value

    if return_lse:
        raise ValueError("return_lse is not yet supported")

    if k.dtype != v.dtype:
        raise ValueError("K and V arrays must have the same dtype")

    if q.dtype not in [jnp.float16, jnp.bfloat16]:
        raise ValueError(
            "Q data type must be float16 or bfloat16. Only K/V data type can be an fp8 type"
        )
    if not is_float8(k):
        if k.dtype != q.dtype:
            raise ValueError(
                "if K is not an fp8 type, it must have the same dtype as Q"
            )

    if backend == "auto":
        backend = _determine_backend(
            PosEncodingMode[pos_encoding_mode].value,
            use_fp16_qk_reduction,
            use_custom_mask=packed_custom_mask is not None,
        )

    call = jax.ffi.ffi_call(
        "mtrt_flashinfer_single_prefill_with_kv_cache",
        [
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            jax.ShapeDtypeStruct(k.shape, k.dtype),
            jax.ShapeDtypeStruct(v.shape, v.dtype),
        ],
        # The K/V input tensors are aliases of the K/V output tensors.
        input_output_aliases={1: 1, 2: 2},
    )

    workspace = jnp.zeros(32 * 1024 * 1024 // 2, dtype=jnp.float16)
    encoded_kv_layout = TensorLayout[kv_layout].value
    o_dtype = q.dtype

    uri = get_uri(
        backend,
        q.dtype,
        k.dtype,
        o_dtype,
        q.shape[-1],  # head_dim_qk
        v.shape[-1],  # head_dim_vo
        PosEncodingMode[pos_encoding_mode].value,
        window_left >= 0,
        logits_soft_cap > 0,
        use_fp16_qk_reduction,
    )
    libpath = find_flashinfer_lib(uri)

    def ffi_config(arg_spec):
        return {
            "arg_spec": ";".join(arg_spec),
            "plugin": str(libpath),
            "func": "run",
            "mtrt_ffi_backend": "tvm_ffi",
        }

    if backend == "fa3":
        if not is_float8(q):
            spec = [
                "args.0",  # q
                "args.1",  # k
                "args.2",  # v
                "args.3",  # workspace
                "rets.4",  # o
                "none",  # maybe_lse
                "attrs.mask_mode",  # mask_mode
                "attrs.kv_layout",  # layout
                "attrs.window_left",  # window_left
                "attrs.logits_soft_cap",  # logits_soft_cap
                "attrs.sm_scale",  # sm_scale
            ]
            return call(
                q,
                k,
                v,
                workspace,
                mask_mode=mask_mode,
                kv_layout=encoded_kv_layout,
                window_left=window_left,
                logits_soft_cap=logits_soft_cap,
                sm_scale=sm_scale,
                **ffi_config(spec),
            )  # type: ignore
        else:
            # FP8 enabled
            spec = [
                "args.0",  # q
                "args.1",  # k
                "args.2",  # v
                "args.3",  # workspace
                "rets.4",  # o
                "none",  # maybe_lse
                "attrs.mask_mode",  # mask_mode
                "attrs.kv_layout",  # layout
                "attrs.window_left",  # window_left,
                "attrs.scale_q",  # scale_q,
                "attrs.scale_k",  #   scale_k,
                "attrs.scale_v",  # scale_v,
                "attrs.sm_scale",  # sm_scale,
            ]
            return call(
                q,
                k,
                v,
                workspace,
                mask_mode=mask_mode,
                kv_layout=encoded_kv_layout,
                window_left=window_left,
                scale_q=scale_q,
                scale_k=scale_k,
                scale_v=scale_v,
                sm_scale=sm_scale,
                **ffi_config(spec),
            )  # type: ignore
    if backend == "fa2":
        spec = [
            "args.0",  # q
            "args.1",  # k
            "args.2",  # v
            "args.3",  # workspace
            "rets.0",  # o
            "none",  # maybe_lse
            "attrs.mask_mode",
            "attrs.kv_layout",
            "attrs.window_left",
            "none",  # maybe_packed_custom_mask
            "none",  # maybe_alibi_slopes
            "attrs.logits_soft_cap",
            "attrs.sm_scale",
            "attrs.rope_scale",
            "attrs.rope_theta",
        ]
        return call(
            q,
            k,
            v,
            workspace,
            mask_mode=mask_mode,
            kv_layout=encoded_kv_layout,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            sm_scale=sm_scale,
            rope_scale=1.0 / rope_scale,
            rope_theta=1.0 / rope_theta,
            **ffi_config(spec),
        )  # type: ignore

    raise ValueError("Invalid backend: {}".format(backend))
