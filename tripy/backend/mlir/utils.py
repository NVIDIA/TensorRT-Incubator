import contextlib
import os
from typing import Dict

from mlir import dialects, ir

from tripy import utils
from tripy.common import ShapeInfo


def make_ir_context() -> ir.Context:
    context = ir.Context()

    context.enable_multithreading(False)
    # Allow unregistered dialects to assign trt shape_profile attribute to stablehlo program.
    context.allow_unregistered_dialects = True
    dialects.stablehlo.register_dialect(context)
    return context


def get_mlir_dtype(dtype: "tripy.dtype"):
    """
    Converts a tripy data type to an MLIR data type.
    """
    return {
        "float32": ir.F32Type.get(),
        "float16": ir.F16Type.get(),
        "float8e4m3fn": ir.Float8E4M3FNType.get(),
        "bfloat16": ir.BF16Type.get(),
        "int4": ir.IntegerType.get_signless(4),
        "int8": ir.IntegerType.get_signless(8),
        "int32": ir.IntegerType.get_signless(32),
        "int64": ir.IntegerType.get_signless(64),
        "uint8": ir.IntegerType.get_unsigned(8),
        "bool": ir.IntegerType.get_signless(1),
    }[dtype.name]


def make_mlir_tensor(shape: ShapeInfo, dtype: "tripy.common.dtype") -> ir.RankedTensorType:
    return ir.RankedTensorType.get(
        [ir.ShapedType.get_dynamic_size() if s.is_dynamic_dim() else s.min for s in utils.make_list(shape)],
        get_mlir_dtype(dtype),
    )


def remove_constants(mlir_text) -> str:
    lines = mlir_text.split("\n")

    def replace_dense_data(text):
        const_start_index = text.find("<") + 1
        const_end_index = text.find(">") - 1
        start_index = text.find(": tensor<") + 9

        substr = text[start_index:]
        dims = substr.split("x")
        dims = [int(dim) for dim in dims if dim.isdigit()]

        if utils.should_omit_constant_in_str(dims):
            return text[:const_start_index] + "..." + text[const_end_index + 1 :]
        return text

    replaced = [replace_dense_data(line) if "stablehlo.constant dense" in line else line for line in lines]
    return "\n".join(replaced)
