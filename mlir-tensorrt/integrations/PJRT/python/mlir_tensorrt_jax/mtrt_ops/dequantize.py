# Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
"""Custom JAX primitive for TensorRT dequantization operation."""

from typing import Literal, List, Optional
import jax
import jax.numpy as jnp
from jax._src.typing import ArrayLike, DTypeLike
from jax._src.interpreters.mlir import LoweringRuleContext, Value
from jax.interpreters import mlir
from jax.interpreters.mlir import ir

from .utils import declare_primitive, default_layouts, JAX_VERSION_0_6_0_OR_GREATER

if JAX_VERSION_0_6_0_OR_GREATER:
    from jax._src.core import ShapedArray
else:
    from jax.core import ShapedArray

DequantizationType = Literal["tensorrt.pt_dq", "tensorrt.pc_dq", "tensorrt.block_dq"]

_HAS_FLOAT8_E4M3FN = hasattr(jnp, "float8_e4m3fn")
_HAS_FLOAT4_E2M1FN = hasattr(jnp, "float4_e2m1fn")
_HAS_INT4 = hasattr(jnp, "int4")

# Based on the availability of the dtypes for installed JAX version, build the list of supported input dtypes for block dequantization.
_SUPPORTED_BLOCK_DEQUANT_INPUT_DTYPES = []
if _HAS_FLOAT8_E4M3FN:
    _SUPPORTED_BLOCK_DEQUANT_INPUT_DTYPES.append(jnp.float8_e4m3fn)
if _HAS_INT4:
    _SUPPORTED_BLOCK_DEQUANT_INPUT_DTYPES.append(jnp.int4)
if _HAS_FLOAT4_E2M1FN:
    _SUPPORTED_BLOCK_DEQUANT_INPUT_DTYPES.append(jnp.float4_e2m1fn)


# Define the primitive
dequantize_p = declare_primitive("mtrt_dequantize")


# Abstract evaluation
def _dequantize_abstract_eval(
    input: ShapedArray,
    scale: ShapedArray,
    *,
    mode: str,
    axis: Optional[int],
    output_dtype: DTypeLike,
) -> ShapedArray:
    """Output has input's shape and output_dtype.

    Validates scale shape based on dequantization mode.
    """
    if mode == "tensorrt.pt_dq":
        assert axis is None, "axis is only valid for tensorrt.pc_dq mode"
        assert (
            scale.shape == ()
        ), f"tensorrt.pt_dq (per-tensor) mode expects scalar scale, got shape {scale.shape}"
    elif mode == "tensorrt.pc_dq":
        assert axis is not None, "tensorrt.pc_dq mode requires axis"
        assert axis >= 0 and axis < len(input.shape), (
            f"tensorrt.pc_dq mode requires axis to be in range [0, {len(input.shape)}), "
            f"got {axis}"
        )
        assert (
            len(scale.shape) == 1
        ), f"tensorrt.pc_dq (per-channel) mode expects 1-D scale tensor, got shape {scale.shape}"
        assert scale.shape[0] == input.shape[axis], (
            f"tensorrt.pc_dq (per-channel) mode expects scale size {input.shape[axis]} "
            f"(matching axis {axis}), got {scale.shape[0]}"
        )
    elif mode == "tensorrt.block_dq":
        assert axis is None, "axis is only valid for tensorrt.pc_dq mode"
        assert len(scale.shape) == len(input.shape), (
            f"tensorrt.block_dq mode expects scale with same rank as input ({len(input.shape)}), "
            f"got rank {len(scale.shape)}"
        )

        diff_dims = [
            i for i in range(len(input.shape)) if input.shape[i] != scale.shape[i]
        ]
        assert len(diff_dims) == 1, (
            f"tensorrt.block_dq mode expects exactly one blocking dimension, "
            f"but found {len(diff_dims)} blocking dimensions"
        )
        blocking_dim = diff_dims[0]
        assert blocking_dim in [len(input.shape) - 1, len(input.shape) - 2], (
            f"tensorrt.block_dq mode requires blocking dimension to be last or second-to-last, "
            f"got dimension {blocking_dim} in shape of rank {len(input.shape)}"
        )

        assert (
            input.dtype in _SUPPORTED_BLOCK_DEQUANT_INPUT_DTYPES
        ), f"tensorrt.block_dq mode supports input dtype to be one of {_SUPPORTED_BLOCK_DEQUANT_INPUT_DTYPES}, got {input.dtype}"
    else:
        raise ValueError(f"Unknown dequantization mode: {mode}")

    assert output_dtype in [
        jnp.float16,
        jnp.bfloat16,
        jnp.float32,
    ], f"Unsupported output dtype for dequantization: {output_dtype}"

    return ShapedArray(input.shape, output_dtype)


dequantize_p.def_abstract_eval(_dequantize_abstract_eval)


# Eager mode implementation (raises error to force JIT)
dequantize_p.def_impl(
    lambda *a, **k: (_ for _ in ()).throw(
        NotImplementedError("mtrt_dequantize requires jit compilation")
    )
)


# MLIR lowering
def _dequantize_lowering(
    ctx: LoweringRuleContext,
    input: Value,
    scale: Value,
    *,
    mode: str,
    axis: Optional[int],
    output_dtype: DTypeLike,
) -> List[Value]:
    """Lower to TensorRT custom call."""
    extra_attributes = {
        "mode": ir.StringAttr.get(mode),
    }
    if axis is not None:
        extra_attributes["axis"] = ir.IntegerAttr.get(
            ir.IntegerType.get_signless(32), axis
        )

    result = mlir.custom_call(
        "tensorrt.dequantize",
        result_types=[mlir.aval_to_ir_type(ctx.avals_out[0])],
        operands=[input, scale],
        backend_config="",
        operand_layouts=default_layouts(input.type.shape, scale.type.shape),
        result_layouts=default_layouts(ctx.avals_out[0].shape),
        extra_attributes=extra_attributes,
    )
    return result.results


def register_dequantize_lowering():
    """Register the dequantize lowering for the mlir_tensorrt platform."""
    mlir.register_lowering(dequantize_p, _dequantize_lowering, platform="mlir_tensorrt")


def mtrt_dequantize(
    input: ArrayLike,
    scale: ArrayLike,
    *,
    mode: DequantizationType,
    axis: Optional[int] = None,
    output_dtype: DTypeLike = jnp.float32,
) -> jax.Array:
    """Dequantize a tensor using TensorRT.

    Args:
        input: Quantized tensor to dequantize
        scale: Scale factor tensor (must be build-time constant):
            - tensorrt.pt_dq (per-tensor): scalar
            - tensorrt.pc_dq (per-channel): 1-D tensor with size matching axis dimension
            - tensorrt.block_dq (block): same rank as input, with exactly one dimension different
                               (the blocking dimension, which must be last or second-to-last)
        mode: Dequantization mode - "tensorrt.pt_dq", "tensorrt.pc_dq", or "tensorrt.block_dq"
        axis: Axis along which to apply dequantization. Axis is valid only for "tensorrt.pc_dq" mode.
        output_dtype: Output dtype (default: float32)

    Returns:
        Dequantized tensor

    Examples:
        # Per-tensor dequantization (scalar scale).
        scale = jnp.array(0.5)
        dequantized = mtrt_dequantize(quantized, scale=scale, mode="tensorrt.pt_dq")

        # Per-channel dequantization along axis 1 (1-D scale).
        # For input shape (32, 64, 128), scale shape should be (64,)
        scales = jnp.ones(64)
        dequantized = mtrt_dequantize(quantized, scale=scales, mode="tensorrt.pc_dq", axis=1)

        # Block dequantization with axis 0 as blocking dimension.
        # For quantized input shape (64, 128) with scale shape (4, 128).
        # The block size (for example, 16 in this case) is inferred from the scale shape.
        block_scales = jnp.ones((4, 128))
        dequantized = mtrt_dequantize(quantized, scale=block_scales, mode="tensorrt.block_dq")

    """
    return dequantize_p.bind(
        input,
        scale,
        mode=mode,
        axis=axis,
        output_dtype=output_dtype,
    )
