from typing import Any, Dict, List
from mlir import ir
from mlir.dialects import func as func_dialect

from tripy.common.logging import G_LOGGER
from tripy.backend.mlir.utils import make_ir_context


def lower_flat_ir_to_mlir(flat_ir: "FlatIR") -> ir.Module:
    """
    Lowers FlatIR representation of a program into its equivalent StableHLO representation.
    Args:
        flat_ir: FlatIR representation of a program.
    Returns:
        mlir Module which is functionally equivalent to the input FlatIR program.
    """
    return flat_ir.to_mlir()
