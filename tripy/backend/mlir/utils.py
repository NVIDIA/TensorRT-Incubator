import subprocess

from mlir import dialects, ir

from tripy.flat_ir import FlatIR
from tripy.logging import G_LOGGER
from tripy.ops import Storage
from tripy.util import log_time


def make_ir_context() -> ir.Context:
    context = ir.Context()

    context.enable_multithreading(False)

    dialects.stablehlo.register_dialect(context)
    return context


@log_time
def execute_binary(bin_path):
    result = subprocess.Popen(
        bin_path, shell=True, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    output, _ = result.communicate()
    if result.returncode != 0:
        G_LOGGER.error(f"Command failed with return code {result.returncode}")
    print(output)


def convert_dtype(dtype: "tripy.frontend.DataType"):
    """
    Converts a tripy data type to an MLIR data type.
    """
    return {
        "float32": ir.F32Type.get(),
        "float16": ir.F16Type.get(),
        "float8e4m3fn": ir.Float8E4M3FNType.get(),
        "bfloat16": ir.BF16Type.get(),
        "int4": ir.IntegerType.get_signed(4),
        "int8": ir.IntegerType.get_signed(8),
        "int32": ir.IntegerType.get_signed(32),
        "int64": ir.IntegerType.get_signed(64),
        "uint8": ir.IntegerType.get_unsigned(8),
        # TODO (#26): Figure out how to make boolean types work.
    }[dtype.name]
