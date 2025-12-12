# Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
"""Custom JAX primitive for TensorRT dynamic quantization operation."""

from typing import List
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

_HAS_FLOAT8_E4M3FN = hasattr(jnp, "float8_e4m3fn")
_HAS_FLOAT4_E2M1FN = hasattr(jnp, "float4_e2m1fn")

# Based on the availability of the dtypes for installed JAX version, build the list of supported dtypes for dynamic quantization.
_SUPPORTED_DYNAMIC_QUANT_OUTPUT_DTYPES = []
if _HAS_FLOAT4_E2M1FN:
    _SUPPORTED_DYNAMIC_QUANT_OUTPUT_DTYPES.append(jnp.float4_e2m1fn)

_SUPPORTED_DYNAMIC_QUANT_SCALE_DTYPES = []
if _HAS_FLOAT8_E4M3FN:
    _SUPPORTED_DYNAMIC_QUANT_SCALE_DTYPES.append(jnp.float8_e4m3fn)

# Default dtypes (None if not available)
_DEFAULT_OUTPUT_DTYPE = jnp.float4_e2m1fn if _HAS_FLOAT4_E2M1FN else None
_DEFAULT_SCALE_DTYPE = jnp.float8_e4m3fn if _HAS_FLOAT8_E4M3FN else None

# Define the primitive
dynamic_quantize_p = declare_primitive("mtrt_dynamic_quantize")
dynamic_quantize_p.multiple_results = True


# Abstract evaluation
def _dynamic_quantize_abstract_eval(
    input: ShapedArray,
    double_quant_scale: ShapedArray,
    *,
    axis: int,
    block_size: int,
    output_dtype: DTypeLike,
    scale_dtype: DTypeLike,
) -> List[ShapedArray]:
    """Compute output shapes and dtypes.

    Returns two outputs:
    1. `quantized_data` has same shape as input and dtype `output_dtype`
    2. `scale` has same shape as input but with axis dimension reduced by `block_size` and dtype `scale_dtype`
    """
    assert axis is not None, "axis is required for dynamic quantization"
    assert axis in [len(input.shape) - 1, len(input.shape) - 2], (
        f"axis must be the last dimension or the second to last dimension, "
        f"got {axis}"
    )
    assert block_size == 16, "currently only blocks of 16 elements are supported"
    assert (
        double_quant_scale.shape == ()
    ), f"`double_quant_scale` is expected to be scalar, got shape {double_quant_scale.shape}"
    assert (
        input.shape[axis] % block_size == 0
    ), f"Input shape[{axis}]={input.shape[axis]} must be divisible by block_size={block_size}"

    assert (
        output_dtype in _SUPPORTED_DYNAMIC_QUANT_OUTPUT_DTYPES
    ), f"dynamic quantization supports output_dtype to be one of {_SUPPORTED_DYNAMIC_QUANT_OUTPUT_DTYPES}, got {output_dtype}"

    assert (
        scale_dtype in _SUPPORTED_DYNAMIC_QUANT_SCALE_DTYPES
    ), f"dynamic quantization supports scale_dtype to be one of {_SUPPORTED_DYNAMIC_QUANT_SCALE_DTYPES}, got {scale_dtype}"

    assert input.dtype in [
        jnp.float16,
        jnp.bfloat16,
        jnp.float32,
    ], f"Unsupported input dtype for dynamic quantization: {input.dtype}"

    # Quantized scale shape: axis dimension reduced by block_size
    scale_shape = (
        input.shape[:axis]
        + (input.shape[axis] // block_size,)
        + input.shape[axis + 1 :]
    )

    return [
        ShapedArray(input.shape, output_dtype),
        ShapedArray(scale_shape, scale_dtype),
    ]


dynamic_quantize_p.def_abstract_eval(_dynamic_quantize_abstract_eval)


# Eager mode implementation (raises error to force JIT)
dynamic_quantize_p.def_impl(
    lambda *a, **k: (_ for _ in ()).throw(
        NotImplementedError("mtrt_dynamic_quantize requires jit compilation")
    )
)


# MLIR lowering
def _dynamic_quantize_lowering(
    ctx: LoweringRuleContext,
    input: Value,
    double_quant_scale: Value,
    *,
    axis: int,
    block_size: int,
    output_dtype: DTypeLike,
    scale_dtype: DTypeLike,
) -> List[Value]:
    """Lower to TensorRT custom call."""
    extra_attributes = {
        "axis": ir.IntegerAttr.get(ir.IntegerType.get_signless(32), axis),
        "block_size": ir.IntegerAttr.get(ir.IntegerType.get_signless(32), block_size),
    }
    result = mlir.custom_call(
        "tensorrt.dynamic_quantize",
        result_types=[
            mlir.aval_to_ir_type(ctx.avals_out[0]),
            mlir.aval_to_ir_type(ctx.avals_out[1]),
        ],
        operands=[input, double_quant_scale],
        backend_config="",
        operand_layouts=default_layouts(
            input.type.shape, double_quant_scale.type.shape
        ),
        result_layouts=default_layouts(
            ctx.avals_out[0].shape,
            ctx.avals_out[1].shape,
        ),
        extra_attributes=extra_attributes,
    )
    return result.results


def register_dynamic_quantize_lowering():
    """Register the dynamic quantize lowering for the mlir_tensorrt platform."""
    mlir.register_lowering(
        dynamic_quantize_p, _dynamic_quantize_lowering, platform="mlir_tensorrt"
    )


def mtrt_dynamic_quantize(
    input: ArrayLike,
    double_quant_scale: ArrayLike,
    *,
    axis: int,
    block_size: int = 16,
    output_dtype: DTypeLike = None,
    scale_dtype: DTypeLike = None,
) -> tuple[jax.Array, jax.Array]:
    """Dynamically quantize a tensor using TensorRT.
    Dynamic quantization is always block quantization.

    Args:
        input: Tensor to quantize
        double_quant_scale: Double quantization scale factor
        axis: Axis along which to apply block quantization
        block_size: Size of blocks for quantization (default: 16)
        output_dtype: Data type for quantized output (default: float4_e2m1fn if available)
        scale_dtype: Data type for scale values (default: float8_e4m3fn if available)

    Returns:
        Tuple of (quantized_data, scale) where:
        - `quantized_data` has same shape as input and dtype `output_dtype`
        - `scale` has same shape as input but with axis dimension reduced by `block_size` and dtype `scale_dtype`

    Note:
        This function requires JAX version with support for float4_e2m1fn and float8_e4m3fn dtypes.
        Using this function with older JAX versions will raise a ValueError.
    """
    # Set default dtypes if not provided
    if output_dtype is None:
        if _DEFAULT_OUTPUT_DTYPE is None:
            raise ValueError(
                "Dynamic quantization requires JAX version with support for float4_e2m1fn dtype. "
                "Please upgrade JAX or explicitly specify output_dtype."
            )
        output_dtype = _DEFAULT_OUTPUT_DTYPE

    if scale_dtype is None:
        if _DEFAULT_SCALE_DTYPE is None:
            raise ValueError(
                "Dynamic quantization requires JAX version with support for float8_e4m3fn dtype. "
                "Please upgrade JAX or explicitly specify scale_dtype."
            )
        scale_dtype = _DEFAULT_SCALE_DTYPE

    return dynamic_quantize_p.bind(
        input,
        double_quant_scale,
        axis=axis,
        block_size=block_size,
        output_dtype=output_dtype,
        scale_dtype=scale_dtype,
    )
