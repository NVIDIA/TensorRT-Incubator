# Copyright (c) 2026 NVIDIA CORPORATION. All rights reserved.
"""Custom JAX primitives for NVTX range annotations."""

import functools
from dataclasses import dataclass
from typing import List

import jax.numpy as jnp
from jax._src.interpreters.mlir import LoweringRuleContext, Value
from jax.interpreters import mlir
from jax.interpreters.mlir import ir

from .utils import declare_primitive, default_layouts, JAX_VERSION_0_6_0_OR_GREATER

__all__ = [
    "mtrt_nvtx_push",
    "mtrt_nvtx_pop",
    "mtrt_nvtx_annotate",
    "NVTX_COLOR",
    "register_nvtx_range_lowering",
]

if JAX_VERSION_0_6_0_OR_GREATER:
    from jax._src.core import ShapedArray
else:
    from jax.core import ShapedArray


@dataclass(frozen=True)
class NvtxColor:
    """ARGB color values compatible with the NVTX C API."""

    GREEN: int = 0xFF00FF00
    BLUE: int = 0xFF0000FF
    RED: int = 0xFFFF0000
    YELLOW: int = 0xFFFFFF00
    CYAN: int = 0xFF00FFFF
    MAGENTA: int = 0xFFFF00FF
    ORANGE: int = 0xFFFF8800
    PURPLE: int = 0xFF8800FF
    WHITE: int = 0xFFFFFFFF
    GRAY: int = 0xFF888888


NVTX_COLOR = NvtxColor()

# ──────────────────────────── nvtx_push primitive ────────────────────────────

nvtx_push_p = declare_primitive("mtrt_nvtx_push")
nvtx_push_p.multiple_results = True


def _nvtx_push_abstract_eval(
    *inputs: ShapedArray, name: str, color: int
) -> List[ShapedArray]:
    tensor_results = [ShapedArray(inp.shape, inp.dtype) for inp in inputs]
    range_id = ShapedArray((), jnp.int64)
    return tensor_results + [range_id]


nvtx_push_p.def_abstract_eval(_nvtx_push_abstract_eval)
nvtx_push_p.def_impl(
    lambda *a, **k: (_ for _ in ()).throw(
        NotImplementedError("mtrt_nvtx_push requires jit compilation")
    )
)


def _nvtx_push_lowering(
    ctx: LoweringRuleContext, *inputs: Value, name: str, color: int
) -> List[Value]:
    # Operands: N tensors
    # Results: N tensors (same as inputs) + 1 tensor<i64> (range_id)
    result_types = [inp.type for inp in inputs] + [
        ir.RankedTensorType.get([], ir.IntegerType.get_signless(64))
    ]
    result_layouts = default_layouts(*(a.shape for a in ctx.avals_out[:-1]))
    result_layouts.append(range(0))  # scalar i64 layout

    result = mlir.custom_call(
        "nvtx.push",
        result_types=result_types,
        operands=list(inputs),
        has_side_effect=True,
        backend_config="",
        operand_layouts=default_layouts(*(a.shape for a in ctx.avals_in)),
        result_layouts=result_layouts,
        extra_attributes={
            "name": ir.StringAttr.get(name),
            "color": ir.IntegerAttr.get(ir.IntegerType.get_signless(32), color),
        },
    )
    return result.results


# ──────────────────────────── nvtx_pop primitive ─────────────────────────────

nvtx_pop_p = declare_primitive("mtrt_nvtx_pop")
nvtx_pop_p.multiple_results = True


def _nvtx_pop_abstract_eval(*inputs_and_range_id: ShapedArray) -> List[ShapedArray]:
    assert (
        len(inputs_and_range_id) >= 2
    ), "mtrt_nvtx_pop expects at least 1 tensor + 1 range_id"
    range_id = inputs_and_range_id[-1]
    assert range_id.shape == () and range_id.dtype == jnp.int64, (
        f"mtrt_nvtx_pop last input must be scalar i64 range_id, "
        f"got shape={range_id.shape} dtype={range_id.dtype}"
    )
    return [ShapedArray(inp.shape, inp.dtype) for inp in inputs_and_range_id[:-1]]


nvtx_pop_p.def_abstract_eval(_nvtx_pop_abstract_eval)
nvtx_pop_p.def_impl(
    lambda *a, **k: (_ for _ in ()).throw(
        NotImplementedError("mtrt_nvtx_pop requires jit compilation")
    )
)


def _nvtx_pop_lowering(
    ctx: LoweringRuleContext, *inputs_and_range_id: Value
) -> List[Value]:
    # Operands: N tensors + 1 i64 (range_id)
    # Results: N tensors (passthrough, excluding range_id)
    tensor_inputs = list(inputs_and_range_id[:-1])
    result_types = [inp.type for inp in tensor_inputs]

    operand_layouts = default_layouts(*(a.shape for a in ctx.avals_in[:-1]))
    operand_layouts.append(range(0))  # scalar i64 layout

    result = mlir.custom_call(
        "nvtx.pop",
        result_types=result_types,
        operands=list(inputs_and_range_id),
        has_side_effect=True,
        backend_config="",
        operand_layouts=operand_layouts,
        result_layouts=default_layouts(*(a.shape for a in ctx.avals_out)),
    )
    return result.results


# ──────────────────────────── Registration ───────────────────────────────────


def register_nvtx_range_lowering():
    """Register the nvtx range lowerings for the mlir_tensorrt platform."""
    mlir.register_lowering(nvtx_push_p, _nvtx_push_lowering, platform="mlir_tensorrt")
    mlir.register_lowering(nvtx_pop_p, _nvtx_pop_lowering, platform="mlir_tensorrt")


# ──────────────────────────── Public API ─────────────────────────────────────


def mtrt_nvtx_push(*inputs, name: str, color: int = NVTX_COLOR.GREEN):
    """Start an NVTX range, returning passthrough tensors and a range token.

    The range token (an i64 correlation ID) must be passed to
    `mtrt_nvtx_pop` to end the range. This pairing supports overlapping
    ranges correctly.

    Args:
        *inputs: Tensor(s) to pass through.
        name:  NVTX range name visible in Nsight Systems.
        color: ARGB color for the range (default: green).

    Returns:
        If single input: (tensor, range_id)
        If multiple inputs: (tuple_of_tensors, range_id)
    """
    if not inputs:
        raise ValueError(
            "mtrt_nvtx_push requires at least one tensor input. "
            "Pass the tensor(s) whose data flow should be marked by this range."
        )
    results = nvtx_push_p.bind(*inputs, name=name, color=color)
    range_id = results[-1]
    tensors = results[:-1]
    if len(tensors) == 1:
        return tensors[0], range_id
    return tuple(tensors), range_id


def mtrt_nvtx_pop(*inputs, range_id):
    """End an NVTX range using the correlation ID from `mtrt_nvtx_push`.

    Args:
        *inputs: Tensor(s) to pass through. At least one tensor is required
            to maintain a data dependency that prevents dead code elimination.
        range_id: The correlation ID returned by `mtrt_nvtx_push`

    Returns:
        Single tensor or tuple of tensors (passthrough).
    """
    if not inputs:
        raise ValueError(
            "mtrt_nvtx_pop requires at least one tensor input. "
            "Pass the tensor(s) whose data flow should be marked by this range."
        )
    results = nvtx_pop_p.bind(*inputs, range_id)
    if len(results) == 1:
        return results[0]
    return tuple(results)


def mtrt_nvtx_annotate(*, name: str, color: int = NVTX_COLOR.GREEN):
    """Decorator that wraps a function with paired nvtx start/end markers.

    This structurally guarantees that start and end are always paired — there
    is no way to emit one without the other. The range token is handled
    internally.

    Args:
        name:  NVTX range name visible in Nsight Systems.
        color: ARGB color for the range (default: green).

    Example:

        @mtrt_nvtx_annotate(name="my_softmax", color=NVTX_COLOR.MAGENTA)
        def exp_normalize(x):
            e = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
            return e / jnp.sum(e, axis=-1, keepdims=True)

        @jax.jit
        def forward(x):
            return exp_normalize(x)
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args):
            pushed = nvtx_push_p.bind(*args, name=name, color=color)
            range_id = pushed[-1]
            tensor_args = pushed[:-1]
            results = fn(*tensor_args)
            is_tuple = isinstance(results, (list, tuple))
            flat = list(results) if is_tuple else [results]
            popped = nvtx_pop_p.bind(*flat, range_id)
            return tuple(popped) if is_tuple else popped[0]

        return wrapper

    return decorator
